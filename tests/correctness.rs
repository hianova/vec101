use vec101::{vec101_block, vec101_compute, vec101_context};

struct XorShift32(u32);
impl XorShift32 {
    fn new(seed: u32) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn next_u64(&mut self) -> u64 {
        ((self.next() as u64) << 32) | (self.next() as u64)
    }
    fn next_i8(&mut self) -> i8 {
        self.next() as i8
    }
    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u32::MAX as f32)
    }
}

fn naive_fp32_compute(
    num_blocks: usize,
    w_stream: &[vec101_block],
    x_stream: &[i8],
    i_stream: &[u32],
    s_stream: &[f32],
    out_buffer: &mut [f32],
) {
    for i in 0..num_blocks {
        let mut block_sum = 0.0;
        let w_block = &w_stream[i];

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            for b in 0..32 {
                let x_val = x_stream[i * 256 + sub * 32 + b] as f32;
                let is_pos = (w_pos_32 & (1 << b)) != 0;
                let is_neg = (w_neg_32 & (1 << b)) != 0;
                let weight = (if is_pos { 1.0 } else { 0.0 }) - (if is_neg { 1.0 } else { 0.0 });
                block_sum += weight * x_val;
            }
        }

        let target_idx = i_stream[i] as usize;
        let scale = s_stream[i];
        out_buffer[target_idx] += block_sum * scale;
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[test]
fn test_vec101_correctness() {
    let mut rng = XorShift32::new(42);
    let num_blocks = 100;
    let num_outputs = 10;

    let mut w_stream = vec![vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; num_blocks];
    for block in &mut w_stream {
        for w in &mut block.w_pos_bits {
            *w = rng.next_u64();
        }
        for w in &mut block.w_neg_bits {
            *w = rng.next_u64();
        }
    }

    let mut x_stream = vec![0i8; num_blocks * 256];
    for x in &mut x_stream {
        *x = rng.next_i8();
    }

    let mut i_stream = vec![0u32; num_blocks];
    for i in &mut i_stream {
        *i = rng.next() % (num_outputs as u32);
    }

    let mut s_stream = vec![0f32; num_blocks];
    for s in &mut s_stream {
        *s = 0.1 + rng.next_f32() * 1.4;
    }

    let mut out_expected = vec![0f32; num_outputs];
    naive_fp32_compute(
        num_blocks,
        &w_stream,
        &x_stream,
        &i_stream,
        &s_stream,
        &mut out_expected,
    );

    let mut out_actual = vec![0f32; num_outputs];
    let ctx = vec101_context {
        w_stream: w_stream.as_ptr(),
        x_stream: x_stream.as_ptr(),
        i_stream: i_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        num_blocks,
    };

    unsafe {
        vec101_compute(&ctx);
    }

    let sim = cosine_similarity(&out_expected, &out_actual);
    println!("Cosine Similarity: {}", sim);
    assert!(sim > 0.99, "Similarity {} is below threshold", sim);
}
