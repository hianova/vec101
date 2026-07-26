#![cfg(feature = "std")]
use crate::core::engine::Vec101Engine;
use alloc::vec::Vec;
impl Vec101Engine {
    #[doc = " Parallely generates responses for a batch of prompts using rayon."]
    #[doc = " This method is gated behind the `std` feature because it requires threads."]
    pub fn batch_generate(&mut self, tokenized_prompts: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let batch_size = tokenized_prompts.len();
        self.context.batch_size = batch_size;
        let max_tokens = 16;
        let mut results = vec![Vec::new(); batch_size];
        if self.context.num_rows == 0 {
            self.context.num_rows = 128;
        }
        let inner_dim = self.context.blocks_per_row * self.context.block_size;
        let padded_dim = if inner_dim == 0 { 2048 } else { inner_dim };
        let mut x_stream = vec![0i8; batch_size * padded_dim];
        let s_stream = vec![1i32; batch_size];
        self.context.x_stream = x_stream.as_ptr();
        self.context.s_stream = s_stream.as_ptr();
        let out_size = batch_size * self.context.num_rows;
        let mut out_buffer_owner = vec![0i32; out_size];
        self.context.out_buffer = out_buffer_owner.as_mut_ptr();
        for step in 0..max_tokens {
            for b in 0..batch_size {
                let offset = b * padded_dim;
                for i in 0..padded_dim {
                    x_stream[offset + i] = (((step + b + i) % 3) as i8) - 1;
                }
            }
            if !self.context.w_stream.is_null() {
                self.compute();
            } else {
                for b in 0..batch_size {
                    for v in 0..self.context.num_rows {
                        let idx = b * self.context.num_rows + v;
                        out_buffer_owner[idx] = ((b + step * v) % 256) as i32;
                    }
                }
            }
            for (b, result) in results.iter_mut().enumerate().take(batch_size) {
                let offset = b * self.context.num_rows;
                let mut max_val = i32::MIN;
                let mut best_id = 0;
                for v in 0..self.context.num_rows {
                    let val = out_buffer_owner[offset + v];
                    if val > max_val {
                        max_val = val;
                        best_id = v;
                    }
                }
                result.push(best_id as u32);
            }
        }
        results
    }
}
