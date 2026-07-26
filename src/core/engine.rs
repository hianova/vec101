use crate::compute::vec101_compute;
use crate::core::vec101_context;
use alloc::vec::Vec;
use core::ptr;

#[repr(C, align(64))]
pub struct Vec101Engine {
    pub(crate) context: vec101_context,
    out_buffer_owner: Vec<i32>,
    kv_blocks_owner: Vec<*const i32>,
}

impl Vec101Engine {
    pub(crate) fn new(mut context: vec101_context) -> Self {
        let out_size = if context.batch_size * context.num_rows == 0 {
            1
        } else {
            context.batch_size * context.num_rows
        };
        let mut out_buffer_owner = alloc::vec![0; out_size];
        context.out_buffer = out_buffer_owner.as_mut_ptr();
        let kv_blocks_owner = alloc::vec![ptr::null(); context.num_blocks];
        context.kv_blocks = kv_blocks_owner.as_ptr();
        Self {
            context,
            out_buffer_owner,
            kv_blocks_owner,
        }
    }
    #[doc = " Set a specific KV block pointer for shared prefix caching."]
    pub fn set_kv_block(&mut self, index: usize, block_ptr: *const i32) {
        if index < self.kv_blocks_owner.len() {
            self.kv_blocks_owner[index] = block_ptr;
        }
    }
    pub fn set_w_stream(&mut self, ptr: *const u8) {
        self.context.w_stream = ptr;
    }
    pub fn set_quant_type(&mut self, q: crate::core::QuantType) {
        self.context.quant_type = q;
    }
    pub fn set_num_rows(&mut self, rows: usize) {
        self.context.num_rows = rows;
    }
    pub fn set_x_stream(&mut self, ptr: *const i8) {
        self.context.x_stream = ptr;
    }
    pub fn set_s_stream(&mut self, ptr: *const i32) {
        self.context.s_stream = ptr;
    }
    pub fn set_batch_size(&mut self, size: usize) {
        self.context.batch_size = size;
    }
    pub fn set_blocks_per_row(&mut self, blocks: usize) {
        self.context.blocks_per_row = blocks;
    }
    #[doc = " Run the computation safely"]
    pub fn compute(&mut self) {
        unsafe {
            vec101_compute(&self.context);
        }
    }
    #[doc = " Get the raw output buffer"]
    pub fn get_output(&self) -> &[i32] {
        &self.out_buffer_owner
    }
    #[doc = " Zero-Token Logit Classification"]
    #[doc = " Returns raw logits for a specific set of target token IDs."]
    pub fn forward_logits_only(&mut self, targets: &[u32]) -> Vec<f32> {
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
