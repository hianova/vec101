/// The fundamental compute block for vec101, precisely aligned to 32 bytes 
/// to match a single AVX2 register. Contains 256 bits of highly compressed 1-bit weights.
/// 0 represents weight = -1, 1 represents weight = +1.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct vec101_block {
    pub w_pos_bits: [u64; 4],
    pub w_neg_bits: [u64; 4],
}

/// The runtime context for the vec101 engine.
/// Holds pointers to perfectly aligned memory streams to avoid any allocations.
#[repr(C)]
pub struct vec101_context {
    /// Highly compressed 1-bit weights stream. (num_rows * blocks_per_row)
    pub w_stream: *const vec101_block,
    /// Continuous activation values stream. (batch_size * blocks_per_row * 256)
    pub x_stream: *const i8,
    /// Quantization scaling factor stream per row. (num_rows)
    pub s_stream: *const f32,
    /// Output buffer. (batch_size * num_rows)
    pub out_buffer: *mut f32,
    /// Number of tokens processed simultaneously (GEMM Batch Dimension)
    pub batch_size: usize,
    /// Number of rows in the weight matrix
    pub num_rows: usize,
    /// Number of 256-bit blocks per row
    pub blocks_per_row: usize,
    /// Number of parallel threads to use (0 or 1 = sequential)
    pub num_threads: usize,
}

// Ensure the structs are Sync and Send if required for multi-threading.
// Raw pointers are not automatically Send/Sync.
unsafe impl Send for vec101_context {}
unsafe impl Sync for vec101_context {}
