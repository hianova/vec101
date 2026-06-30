use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vec101::{vec101_block, vec101_compute, vec101_context};
use vec101::types::{Vec101SuperBlock, f16_to_f32};

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
    batch_size: usize,
    num_rows: usize,
    blocks_per_row: usize,
    w_stream: &[Vec101SuperBlock],
    x_stream: &[i8],
    s_stream: &[f32],
    out_buffer: &mut [f32],
) {
    let in_features = blocks_per_row * 2048;
    for b in 0..batch_size {
        for r in 0..num_rows {
            let mut row_sum = 0.0;
            for c in 0..blocks_per_row {
                let w_super = &w_stream[r * blocks_per_row + c];
                for sub_blk in 0..8 {
                    let micro_scale = f16_to_f32(w_super.scales[sub_blk]);
                    let w_block = &w_super.blocks[sub_blk];
                    let mut micro_sum = 0.0;
                    for sub in 0..8 {
                        let u64_idx = sub / 2;
                        let shift_amt = (sub % 2) * 32;
                        let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
                        let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

                        for bit in 0..32 {
                            let x_val = x_stream[b * in_features + c * 2048 + sub_blk * 256 + sub * 32 + bit] as f32;
                            let is_pos = (w_pos_32 & (1 << bit)) != 0;
                            let is_neg = (w_neg_32 & (1 << bit)) != 0;
                            let weight = (if is_pos { 1.0 } else { 0.0 }) - (if is_neg { 1.0 } else { 0.0 });
                            micro_sum += weight * x_val;
                        }
                    }
                    row_sum += micro_sum * micro_scale;
                }
            }
            let scale = s_stream[r];
            out_buffer[b * num_rows + r] += row_sum * scale;
        }
    }
}

fn bench_vec101(c: &mut Criterion) {
    let mut rng = XorShift32::new(777);
    let batch_size = 1;
    let num_rows = 1000;
    let blocks_per_row = 1; // 1 * 2048 = 2048 features

    let mut w_stream = vec![Vec101SuperBlock { scales: [0x3C00; 8], offsets: [0; 8], _padding: [0; 32], blocks: [vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 8] }; num_rows * blocks_per_row];
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
    let mut out_actual = vec![0f32; batch_size * num_rows];

    let ctx = vec101_context {
        quant_type: vec101::types::QuantType::Bit1_58,
        w_stream: w_stream.as_ptr() as *const u8,
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        kv_blocks: core::ptr::null(),
        num_blocks: 0,
        block_size: 0,
        batch_size,
        num_rows,
        blocks_per_row,
        num_threads: 1, // Measure single-thread CPU core logic time for micro-bench
        tree_mask: core::ptr::null(),
        tree_size: 0,
    };

    let mut group = c.benchmark_group("Compute Comparison (Batch=1)");

    group.bench_function("naive_fp32", |b| {
        b.iter(|| {
            naive_fp32_compute(
                black_box(batch_size),
                black_box(num_rows),
                black_box(blocks_per_row),
                black_box(&w_stream),
                black_box(&x_stream),
                black_box(&s_stream),
                black_box(&mut out_expected),
            );
        })
    });

    group.bench_function("vec101_simd", |b| {
        b.iter(|| unsafe {
            vec101_compute(black_box(&ctx));
        })
    });

    group.finish();

    let mut ctx_gemm = ctx;
    ctx_gemm.batch_size = 16;
    let mut x_stream_gemm = vec![0i8; 16 * blocks_per_row * 2048];
    for x in &mut x_stream_gemm {
        *x = rng.next_i8();
    }
    let mut out_gemm = vec![0f32; 16 * num_rows];
    ctx_gemm.x_stream = x_stream_gemm.as_ptr();
    ctx_gemm.out_buffer = out_gemm.as_mut_ptr();

    let mut group_gemm = c.benchmark_group("Compute Comparison (Batch=16 GEMM)");
    group_gemm.bench_function("vec101_simd_gemm", |b| {
        b.iter(|| unsafe {
            vec101_compute(black_box(&ctx_gemm));
        })
    });
    group_gemm.finish();
}

criterion_group!(benches, bench_vec101);
criterion_main!(benches);
