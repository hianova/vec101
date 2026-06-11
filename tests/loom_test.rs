#![allow(unused)]
use vec101::{vec101_compute, vec101_context, vec101_block};
use vec101::types::{Vec101SuperBlock, EngineState};

#[cfg(loom)]
#[test]
fn test_vec101_loom_concurrency() {
    loom::model(|| {
        let batch_size = 1;
        let num_rows = 4;
        let blocks_per_row = 1; // 1 SuperBlock
        let num_threads = 2; // Test with 2 threads

        let x_stream = vec![0i8; blocks_per_row * 2048];
        let s_stream = vec![1.0f32; num_rows];
        let w_stream = vec![Vec101SuperBlock { scales: [0x3C00; 8], offsets: [0; 8], _padding: [0; 32], blocks: [vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 8] }; blocks_per_row * num_rows];
        let mut out_buffer = vec![0.0f32; num_rows];

        let ctx = vec101_context {
            x_stream: x_stream.as_ptr(),
            s_stream: s_stream.as_ptr(),
            w_stream: w_stream.as_ptr(),
            out_buffer: out_buffer.as_mut_ptr(),
            num_rows,
            blocks_per_row,
            batch_size,
            num_threads,
            state: EngineState::Drafting { target_tokens: 1 },
        };

        unsafe {
            // This will dispatch to SIMD logic (avx2/neon/scalar) depending on the target arch.
            // Loom will track the `Arc<AtomicUsize>` remaining counter inside `vec101_compute`.
            vec101_compute(&ctx);
        }
        
        // Basic check to ensure the threads complete and memory is consistent.
        assert_eq!(out_buffer[0], 0.0);
    });
}
