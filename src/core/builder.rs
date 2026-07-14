use crate::core::vec101_context;
use crate::core::QuantType;
use crate::core::engine::Vec101Engine;
use core::ptr;

pub struct ComputeContextBuilder {
    quant_type: QuantType,
    w_stream: *const u8,
    x_stream: *const i8,
    s_stream: *const i32,
    num_blocks: usize,
    block_size: usize,
    batch_size: usize,
    num_rows: usize,
    blocks_per_row: usize,
    num_threads: usize,
    tree_mask: *const u32,
    tree_size: usize,
    hardware_handle: *mut core::ffi::c_void,
}

impl Default for ComputeContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeContextBuilder {
    pub fn new() -> Self {
        Self {
            quant_type: QuantType::Bit1_58,
            w_stream: ptr::null(),
            x_stream: ptr::null(),
            s_stream: ptr::null(),
            num_blocks: 0,
            block_size: 16,
            batch_size: 1,
            num_rows: 0,
            blocks_per_row: 0,
            num_threads: 1,
            tree_mask: ptr::null(),
            tree_size: 0,
            hardware_handle: ptr::null_mut(),
        }
    }

    pub fn quant_type(mut self, q: QuantType) -> Self {
        self.quant_type = q;
        self
    }

    pub fn w_stream(mut self, ptr: *const u8) -> Self {
        self.w_stream = ptr;
        self
    }

    pub fn x_stream(mut self, ptr: *const i8) -> Self {
        self.x_stream = ptr;
        self
    }

    pub fn s_stream(mut self, ptr: *const i32) -> Self {
        self.s_stream = ptr;
        self
    }

    pub fn num_blocks(mut self, n: usize) -> Self {
        self.num_blocks = n;
        self
    }

    pub fn block_size(mut self, n: usize) -> Self {
        self.block_size = n;
        self
    }

    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn num_rows(mut self, n: usize) -> Self {
        self.num_rows = n;
        self
    }

    pub fn blocks_per_row(mut self, n: usize) -> Self {
        self.blocks_per_row = n;
        self
    }

    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    pub fn tree_mask(mut self, ptr: *const u32, size: usize) -> Self {
        self.tree_mask = ptr;
        self.tree_size = size;
        self
    }

    pub fn hardware_handle(mut self, ptr: *mut core::ffi::c_void) -> Self {
        self.hardware_handle = ptr;
        self
    }

    pub fn build(self) -> Vec101Engine {
        let ctx = vec101_context {
            quant_type: self.quant_type,
            w_stream: self.w_stream,
            x_stream: self.x_stream,
            s_stream: self.s_stream,
            out_buffer: ptr::null_mut(),
            kv_blocks: ptr::null(),
            num_blocks: self.num_blocks,
            block_size: self.block_size,
            batch_size: self.batch_size,
            num_rows: self.num_rows,
            blocks_per_row: self.blocks_per_row,
            num_threads: self.num_threads,
            tree_mask: self.tree_mask,
            tree_size: self.tree_size,
            hardware_handle: self.hardware_handle,
        };
        Vec101Engine::new(ctx)
    }
}
