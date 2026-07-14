#![allow(unused)]
use std::time::Instant;
use vec101::core::{QuantType, Vec101SuperBlock, vec101_context};
use vec101::{vec101_block, vec101_compute};

struct XorShift32(u32);
impl XorShift32 {
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }
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
    fn next_i32(&mut self) -> i32 {
        self.next() as i32
    }
}

// Result struct to hold benchmark output
struct BenchResult {
    name: String,
    quant_type: String,
    batch_size: usize,
    threads: usize,
    latency_ms: f64,
    throughput: f64,
}

fn run_engine_benchmark(
    name: &str,
    quant_type: QuantType,
    batch_size: usize,
    threads: usize,
    num_rows: usize,
    blocks_per_row: usize,
    iters: usize,
) -> BenchResult {
    let mut rng = XorShift32::new(12345);
    let num_blocks = num_rows * blocks_per_row;
    let in_features = blocks_per_row * 2048;

    let mut w_stream = vec![
        Vec101SuperBlock {
            scales: [1; 8],
            offsets: [0; 8],
            blocks: [vec101_block {
                w_pos_bits: [0; 4],
                w_neg_bits: [0; 4]
            }; 8]
        };
        num_blocks
    ];
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

    let mut x_stream = vec![0i8; batch_size * in_features];
    for x in &mut x_stream {
        *x = rng.next_i8();
    }

    let mut s_stream = vec![0i32; num_rows];
    for s in &mut s_stream {
        *s = rng.next_i32();
    }

    let mut out_buffer = vec![0i32; batch_size * num_rows];

    let ctx = vec101_context {
        quant_type,
        w_stream: w_stream.as_ptr() as *const u8,
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_buffer.as_mut_ptr(),
        kv_blocks: core::ptr::null(),
        num_blocks: 0,
        block_size: 0,
        batch_size,
        num_rows,
        blocks_per_row,
        num_threads: threads,
        tree_mask: core::ptr::null(),
        tree_size: 0,
        hardware_handle: core::ptr::null_mut(),
    };

    // Warmup
    unsafe {
        vec101_compute(&ctx);
    }

    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            vec101_compute(&ctx);
        }
    }
    let duration = start.elapsed() / iters as u32;
    let latency_ms = duration.as_secs_f64() * 1000.0;

    // Throughput is batch_size tokens processed per duration
    let throughput = (batch_size as f64) / duration.as_secs_f64();

    let quant_str = match quant_type {
        QuantType::Bit1_58 => "BitNet (1.58b)",
        QuantType::Q4_0 => "GGUF (Q4_0)",
        _ => "Unknown",
    };

    BenchResult {
        name: name.to_string(),
        quant_type: quant_str.to_string(),
        batch_size,
        threads,
        latency_ms,
        throughput,
    }
}

fn main() {
    println!("# Vec101 Engine Integrated Benchmark\n");

    // We simulate a slice of a 3B parameter model to keep bench times reasonable.
    // 1 SuperBlock = 2048 weights.
    let blocks_per_row = 2; // 4096 input features
    let num_rows = 10000; // 10000 output features (40.9M parameters total)

    println!(
        "* **Model Scale**     : {} Parameters per layer ({} x {})",
        num_rows * blocks_per_row * 2048,
        num_rows,
        blocks_per_row * 2048
    );
    println!("* **Hardware Backend**: ARM NEON / Generic CPU Fallback\n");

    let configs = vec![
        // (Name, QuantType, Batch Size, Threads, Iters)
        ("Framework Overhead", QuantType::Bit1_58, 1, 1, 0, 0, 10000), // Zero features to test pure overhead
        (
            "Decode (Single-Thread)",
            QuantType::Bit1_58,
            1,
            1,
            num_rows,
            blocks_per_row,
            20,
        ),
        (
            "Decode (Multi-Thread)",
            QuantType::Bit1_58,
            1,
            8,
            num_rows,
            blocks_per_row,
            20,
        ),
        (
            "Prefill TTFT (Batch=128)",
            QuantType::Bit1_58,
            128,
            8,
            num_rows,
            blocks_per_row,
            5,
        ),
        (
            "Decode (Single-Thread)",
            QuantType::Q4_0,
            1,
            1,
            num_rows,
            blocks_per_row,
            20,
        ),
        (
            "Decode (Multi-Thread)",
            QuantType::Q4_0,
            1,
            8,
            num_rows,
            blocks_per_row,
            20,
        ),
        (
            "Prefill TTFT (Batch=128)",
            QuantType::Q4_0,
            128,
            8,
            num_rows,
            blocks_per_row,
            5,
        ),
    ];

    println!("| Scenario | Quantization | Batch | Threads | Latency (ms) | Throughput (tok/s) |");
    println!("|----------|--------------|-------|---------|--------------|--------------------|");

    for (name, q_type, batch, threads, rows, blocks, iters) in configs {
        let res = run_engine_benchmark(name, q_type, batch, threads, rows, blocks, iters);

        if rows == 0 {
            // Special format for overhead
            println!(
                "| {:<24} | {:<12} | {:>5} | {:>7} | {:>12.6} | {:>18} |",
                res.name, res.quant_type, res.batch_size, res.threads, res.latency_ms, "-"
            );
        } else {
            println!(
                "| {:<24} | {:<12} | {:>5} | {:>7} | {:>12.3} | {:>18.1} |",
                res.name,
                res.quant_type,
                res.batch_size,
                res.threads,
                res.latency_ms,
                res.throughput
            );
        }
    }

    println!("\n*Note: Throughput is measured in tokens/sec for the configured layer size.*");
}
