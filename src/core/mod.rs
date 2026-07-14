pub mod llm_traits;
pub use llm_traits::*;

pub mod builder;
pub mod engine;
pub mod kv_cache;
pub mod sensor;
pub use builder::ComputeContextBuilder;
pub use engine::Vec101Engine;
pub use kv_cache::SharedPrefixManager;
pub use sensor::ContinuousInput;
pub mod math;
pub mod topk;
/// The fundamental compute block for vec101.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct vec101_block {
    pub w_pos_bits: [u64; 4],
    pub w_neg_bits: [u64; 4],
}

/// 完美對齊 64-Byte，且維持 256 維度的終極設計！
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vec101SuperBlock {
    // 第 1 個 Cache Line (64 Bytes)：專門放 Metadata
    // 支援 8 個 Block 的 Scale 和 Offset
    pub scales: [i16; 8],
    pub offsets: [i16; 8],
    // 其餘 8 個 Cache Line：存放實際的 bit 流 (每列對應一個 Block)
    pub blocks: [vec101_block; 8], 
}

/// Gemma Q4_0 Block (32 weights packed into 16 bytes + 1 f16 scale = 18 bytes)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: i16,           // Block Scale (Delta)
    pub qs: [u8; 16],     // 32 個 4-bit 權重打包
}



/// Supported quantization types for Dual Engine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    Bit1_58,
    Q4_0,
}

/// The runtime context for the vec101 engine.
#[repr(C)]
pub struct vec101_context {
    /// Quantization type of the current stream
    pub quant_type: QuantType,
    /// Highly compressed SuperBlocks stream or Q4_0 blocks stream (Zero-Copy Archived)
    pub w_stream: *const u8,
    /// Continuous activation values stream.
    pub x_stream: *const i8,
    /// Quantization scaling factor stream per row.
    pub s_stream: *const i32,
    /// Output buffer.
    pub out_buffer: *mut i32,
    /// Pointers to Paged Attention KV blocks.
    pub kv_blocks: *const *const i32,
    /// Number of valid blocks in the kv_blocks array.
    pub num_blocks: usize,
    /// Number of tokens per block (e.g. 16 or 64).
    pub block_size: usize,
    /// Number of tokens processed simultaneously (GEMM Batch Dimension)
    pub batch_size: usize,
    /// Number of rows in the weight matrix
    pub num_rows: usize,
    /// Number of SuperBlocks per row
    pub blocks_per_row: usize,
    /// Number of parallel threads to use
    pub num_threads: usize,
    /// Tree mask for speculative decoding (1D array of parent indices)
    pub tree_mask: *const u32,
    /// Number of nodes in the speculative decoding tree
    pub tree_size: usize,
    /// Opaque pointer to the hardware backend (e.g., CudaDevice or Metal Device)
    /// The application layer is responsible for its allocation and lifecycle
    pub hardware_handle: *mut core::ffi::c_void,
}

unsafe impl Send for vec101_context {}
unsafe impl Sync for vec101_context {}

/// The Heterogeneous Compute Hub interface (Dual Engine)
pub struct DualEngineContext<'a> {
    /// 1. 唯一的一份物理記憶體映射 (Zero-copy)
    pub shared_kv_cache: &'a mut [u8], 
    pub shared_weights_1_58b: &'a [u8],
    pub shared_weights_4b: &'a [u8],

    /// 2. 你發明的無鎖信箱 (Lock-free Mailbox)
    pub auto_fill_mailbox: crate::sync::AtomicMailboxU32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_structs_debug() {
        let block1 = vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] };
        let _ = alloc::format!("{:?}", block1);

        let sb = Vec101SuperBlock {
            scales: [0; 8],
            offsets: [0; 8],
            blocks: [block1; 8],
        };
        let _ = alloc::format!("{:?}", sb);

        let q4 = BlockQ4_0 { d: 0, qs: [0; 16] };
        let _ = alloc::format!("{:?}", q4);
        
        let _ = alloc::format!("{:?}", QuantType::Bit1_58);
    }

    #[test]
    fn test_mailbox_default() {
        let _ = crate::sync::AtomicMailboxU32::default();
    }
}
