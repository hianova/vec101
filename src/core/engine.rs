use crate::compute::vec101_compute;
use crate::core::vec101_context;
use alloc::vec::Vec;
use core::ptr;

pub struct Vec101Engine {
    pub(crate) ctx: vec101_context,
    out_buffer_owner: Vec<i32>,
    kv_blocks_owner: Vec<*const i32>,
}

impl Vec101Engine {
    pub(crate) fn new(mut ctx: vec101_context) -> Self {
        // Output buffer size is batch_size * num_rows for a full forward pass
        // If num_rows is 0, we'll just allocate a small buffer to avoid issues
        let out_size = if ctx.batch_size * ctx.num_rows == 0 {
            1
        } else {
            ctx.batch_size * ctx.num_rows
        };
        let mut out_buffer_owner = alloc::vec![0; out_size];
        ctx.out_buffer = out_buffer_owner.as_mut_ptr();

        let mut kv_blocks_owner = alloc::vec![ptr::null(); ctx.num_blocks];
        ctx.kv_blocks = kv_blocks_owner.as_ptr();

        Self {
            ctx,
            out_buffer_owner,
            kv_blocks_owner,
        }
    }

    /// Set a specific KV block pointer for shared prefix caching.
    pub fn set_kv_block(&mut self, index: usize, block_ptr: *const i32) {
        if index < self.kv_blocks_owner.len() {
            self.kv_blocks_owner[index] = block_ptr;
        }
    }

    pub fn set_w_stream(&mut self, ptr: *const u8) {
        self.ctx.w_stream = ptr;
    }

    pub fn set_quant_type(&mut self, q: crate::core::QuantType) {
        self.ctx.quant_type = q;
    }

    pub fn set_num_rows(&mut self, rows: usize) {
        self.ctx.num_rows = rows;
    }

    /// Run the computation safely
    pub fn compute(&mut self) {
        unsafe {
            vec101_compute(&self.ctx);
        }
    }

    /// Get the raw output buffer
    pub fn get_output(&self) -> &[i32] {
        &self.out_buffer_owner
    }

    /// Zero-Token Logit Classification
    /// Returns raw logits for a specific set of target token IDs.
    pub fn forward_logits_only(&mut self, targets: &[u32]) -> Vec<f32> {
        // Extract specific logits from the current out_buffer.
        // Assuming out_buffer maps to [batch_size, num_rows], and each target corresponds to a row index.
        let mut results = Vec::with_capacity(targets.len());
        for &target_id in targets {
            let idx = target_id as usize;
            if idx < self.out_buffer_owner.len() {
                results.push(self.out_buffer_owner[idx] as f32);
            } else {
                results.push(0.0);
            }
        }
        results
    }
}
