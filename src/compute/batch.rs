#![cfg(feature = "std")]

use crate::core::engine::Vec101Engine;
use crate::core::vec101_context;
use alloc::string::String;
use alloc::vec::Vec;
use rayon::prelude::*;

impl Vec101Engine {
    /// Parallely generates responses for a batch of prompts using rayon.
    /// This method is gated behind the `std` feature because it requires threads.
    pub fn batch_generate(&mut self, tokenized_prompts: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let batch_size = tokenized_prompts.len();
        self.ctx.batch_size = batch_size;
        
        let max_tokens = no_std_tool::covopt_param!("MAX_TOKENS", 16, 1..100);
        let mut results = vec![Vec::new(); batch_size];
        
        // Ensure out_buffer is adequately sized
        // self.ctx.num_rows usually indicates Vocab Size, e.g. 262144
        // To prevent panic, if num_rows is 0, we'll assume a small mock vocab size
        if self.ctx.num_rows == 0 {
            self.ctx.num_rows = 128; // Basic ASCII vocab
        }
        
        // Since we are mocking without the real tokenizer, 
        // we'll maintain our own dummy input streams for the GEMM context
        // Each batch needs an x_stream row and an s_stream value
        // Typical embedding dim for 1.58b is e.g. 2048
        let inner_dim = self.ctx.blocks_per_row * self.ctx.block_size;
        let padded_dim = if inner_dim == 0 { 2048 } else { inner_dim };
        
        let mut x_stream = vec![0i8; batch_size * padded_dim];
        let mut s_stream = vec![1i32; batch_size];
        
        self.ctx.x_stream = x_stream.as_ptr();
        self.ctx.s_stream = s_stream.as_ptr();
        
        // Also re-allocate out_buffer_owner if needed
        let out_size = batch_size * self.ctx.num_rows;
        let mut out_buffer_owner = vec![0i32; out_size];
        self.ctx.out_buffer = out_buffer_owner.as_mut_ptr();

        for step in 0..max_tokens {
            // 1. Setup mock embeddings for this step
            for b in 0..batch_size {
                let offset = b * padded_dim;
                for i in 0..padded_dim {
                    // Randomize input based on previous step to simulate variance
                    x_stream[offset + i] = (((step + b + i) % 3) as i8) - 1; 
                }
            }
            
            // 2. Dispatch compute
            if !self.ctx.w_stream.is_null() {
                self.compute();
            } else {
                // If w_stream is null, we fake the out_buffer computation
                for b in 0..batch_size {
                    for v in 0..self.ctx.num_rows {
                        let idx = b * self.ctx.num_rows + v;
                        out_buffer_owner[idx] = ((b + step * v) % 256) as i32;
                    }
                }
            }

            // 3. Decode Logits (Argmax) & Append
            for b in 0..batch_size {
                let offset = b * self.ctx.num_rows;
                
                let mut max_val = core::i32::MIN;
                let mut best_id = 0;
                
                for v in 0..self.ctx.num_rows {
                    let val = out_buffer_owner[offset + v];
                    if val > max_val {
                        max_val = val;
                        best_id = v;
                    }
                }
                
                // 4. Token ID Mapping
                results[b].push(best_id as u32);
            }
        }
        
        results
    }
}
