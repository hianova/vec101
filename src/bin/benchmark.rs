use std::time::Instant;
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

fn main() {
    let mut rng = XorShift32::new(12345);
    
    // BitNet b1.58 3B model has roughly 3 Billion parameters.
    // 1 block = 256 weights.
    // Let's assume in_features = 4096, so blocks_per_row = 16.
    let blocks_per_row = 16;
    let num_rows = 11_718_750 / 16; // ~732,421 rows to simulate 3B parameters
    let num_blocks = num_rows * blocks_per_row;
    let max_batch_size = 128; // For TTFT prefill

    println!("Generating {} blocks of data (3B parameters) for performance test...", num_blocks);

    let mut w_stream = vec![vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; num_blocks];
    for block in &mut w_stream {
        for w in &mut block.w_pos_bits { *w = rng.next_u64(); }
        for w in &mut block.w_neg_bits { *w = rng.next_u64(); }
    }

    let mut x_stream = vec![0i8; max_batch_size * blocks_per_row * 256];
    for x in &mut x_stream { *x = rng.next_i8(); }

    let mut s_stream = vec![0f32; num_rows];
    for s in &mut s_stream { *s = 0.1 + rng.next_f32() * 1.4; }

    let mut out_actual = vec![0f32; max_batch_size * num_rows];

    println!("Data generation complete. Starting Decode (TPS) benchmark...");

    let ctx_decode = vec101_context {
        w_stream: w_stream.as_ptr(),
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        batch_size: 1,
        num_rows,
        blocks_per_row,
        num_threads: 8,
    };

    let start_simd = Instant::now();
    unsafe { vec101_compute(&ctx_decode); }
    let decode_duration = start_simd.elapsed();
    let tps = 1.0 / decode_duration.as_secs_f64();

    println!("Starting Prefill (TTFT) benchmark for batch size {}...", max_batch_size);
    let ctx_prefill = vec101_context {
        w_stream: w_stream.as_ptr(),
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        batch_size: max_batch_size,
        num_rows,
        blocks_per_row,
        num_threads: 8,
    };

    let start_prefill = Instant::now();
    unsafe { vec101_compute(&ctx_prefill); }
    let prefill_duration = start_prefill.elapsed();

    println!("\n=== vec101 Engine Benchmark Metrics (BitNet b1.58 3B Scale) ===");
    println!("Hardware Acceleration: M1 NEON + Custom Spin-Latch Multi-threading + GEMM Batching");
    println!("生成速度 (TPS, Decode 1 token): {:.2} tokens/sec (Time: {:?})", tps, decode_duration);
    println!("首字延遲 (TTFT, {} tokens prefill): {:.2} seconds", max_batch_size, prefill_duration.as_secs_f64());
    println!("===============================================================");
}
