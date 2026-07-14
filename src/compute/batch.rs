#![cfg(feature = "std")]

use crate::core::engine::Vec101Engine;
use alloc::string::String;
use alloc::vec::Vec;
use rayon::prelude::*;

impl Vec101Engine {
    /// Parallely generates responses for a batch of prompts using rayon.
    /// This method is gated behind the `std` feature because it requires threads.
    pub fn batch_generate(&mut self, prompts: Vec<String>) -> Vec<String> {
        // In a real implementation, we would:
        // 1. Parallel Tokenize the prompts using rayon
        // 2. Setup the x_stream for the batch
        // 3. Dispatch the batched GEMM compute via `self.compute()`
        // 4. Decode the resulting tokens back to strings in parallel

        // Mock parallel tokenization and decoding steps to demonstrate rayon usage.
        let tokenized: Vec<_> = prompts
            .par_iter()
            .map(|p| {
                // Mock tokenization
                p.len()
            })
            .collect();

        // Normally we'd update self.ctx.batch_size and execute self.compute() here.
        // For demonstration, we'll just run it.
        self.ctx.batch_size = prompts.len();
        self.compute();

        // Mock decoding
        let results: Vec<String> = tokenized
            .par_iter()
            .map(|&len| {
                // Mock output generation based on compute results
                alloc::format!("Generated output for prompt of length {}", len)
            })
            .collect();

        results
    }
}
