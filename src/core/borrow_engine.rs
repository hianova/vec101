use crate::compute::vec101_compute;
use crate::core::vec101_context;
use crate::core::QuantType;
use core::marker::PhantomData;
use core::ptr;

/// A zero-allocation lifetime-bound engine for execution in `no_alloc` environments.
pub struct Vec101EngineBorrow<'a> {
    pub(crate) ctx: vec101_context,
    _marker_w: PhantomData<&'a [u8]>,
    _marker_x: PhantomData<&'a [i8]>,
    _marker_s: PhantomData<&'a [i32]>,
    _marker_out: PhantomData<&'a mut [i32]>,
}

impl<'a> Vec101EngineBorrow<'a> {
    /// Create a new borrowed engine from provided slices.
    pub fn new(
        w_stream: &'a [u8],
        x_stream: &'a [i8],
        s_stream: &'a [i32],
        out_buffer: &'a mut [i32],
        batch_size: usize,
        num_rows: usize,
        blocks_per_row: usize,
    ) -> Result<Self, &'static str> {
        let expected_out_size = if batch_size * num_rows == 0 {
            1
        } else {
            batch_size * num_rows
        };

        if out_buffer.len() < expected_out_size {
            return Err("Output buffer is too small for the given batch_size and num_rows");
        }

        let ctx = vec101_context {
            quant_type: QuantType::Bit1_58,
            w_stream: w_stream.as_ptr(),
            x_stream: x_stream.as_ptr(),
            s_stream: s_stream.as_ptr(),
            out_buffer: out_buffer.as_mut_ptr(),
            kv_blocks: ptr::null(),
            num_blocks: 0,
            block_size: 16,
            batch_size,
            num_rows,
            blocks_per_row,
            num_threads: 1,
            tree_mask: ptr::null(),
            tree_size: 0,
            hardware_handle: ptr::null_mut(),
        };

        Ok(Self {
            ctx,
            _marker_w: PhantomData,
            _marker_x: PhantomData,
            _marker_s: PhantomData,
            _marker_out: PhantomData,
        })
    }

    /// Set quantization type (default is Bit1_58)
    pub fn set_quant_type(&mut self, q: QuantType) {
        self.ctx.quant_type = q;
    }

    /// Set number of threads for computation
    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.ctx.num_threads = num_threads;
    }

    /// Set hardware handle for backend acceleration
    pub fn set_hardware_handle(&mut self, handle: *mut core::ffi::c_void) {
        self.ctx.hardware_handle = handle;
    }

    /// Set tree mask and tree size for speculative decoding
    pub fn set_tree_mask(&mut self, mask: *const u32, size: usize) {
        self.ctx.tree_mask = mask;
        self.ctx.tree_size = size;
    }

    /// Set KV blocks and block size for paged attention
    pub fn set_kv_blocks(&mut self, kv_blocks: *const *const i32, num_blocks: usize, block_size: usize) {
        self.ctx.kv_blocks = kv_blocks;
        self.ctx.num_blocks = num_blocks;
        self.ctx.block_size = block_size;
    }

    /// Run the computation safely
    pub fn compute(&mut self) {
        unsafe {
            vec101_compute(&self.ctx);
        }
    }
}
