use vec101::{vec101_block, vec101_compute, vec101_context};
use vec101::types::{Vec101SuperBlock, EngineState};

mod common;
use common::{XorShift32, naive_fp32_compute};

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
    let batch_size = 2;
    let num_rows = 10;
    let blocks_per_row = 1; // 1 SuperBlock = 2048 features
    let total_blocks = num_rows * blocks_per_row;

    let mut w_stream = vec![Vec101SuperBlock { scales: [0x3C00; 8], offsets: [0; 8], _padding: [0; 32], blocks: [vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 8] }; total_blocks];
    for super_block in &mut w_stream {
        for block in &mut super_block.blocks {
            for w in &mut block.w_pos_bits {
                *w = rng.next_u64();
            }
            for w in &mut block.w_neg_bits {
                *w = rng.next_u64();
            }
        }
    }

    let mut x_stream = vec![0i8; batch_size * blocks_per_row * 2048];
    for x in &mut x_stream {
        *x = rng.next_i8();
    }

    let mut s_stream = vec![0f32; num_rows];
    for s in &mut s_stream {
        *s = 0.1 + rng.next_f32() * 1.4;
    }

    let mut out_expected = vec![0f32; batch_size * num_rows];
    naive_fp32_compute(
        batch_size,
        num_rows,
        blocks_per_row,
        &w_stream,
        &x_stream,
        &s_stream,
        &mut out_expected,
    );

    let mut out_actual = vec![0f32; batch_size * num_rows];
    let ctx = vec101_context {
        w_stream: w_stream.as_ptr(),
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        batch_size,
        num_rows,
        blocks_per_row,
        num_threads: 4,
        state: EngineState::Drafting { target_tokens: 1 },
    };

    unsafe {
        vec101_compute(&ctx);
    }

    let sim = cosine_similarity(&out_expected, &out_actual);
    println!("Cosine Similarity: {}", sim);
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    let threshold = 0.75;
    #[cfg(not(any(feature = "gpu-metal", feature = "gpu-cuda")))]
    let threshold = 0.99;
    assert!(sim > threshold, "Similarity {} is below threshold {}", sim, threshold);
}
