/// An f16 type represented as u16 for strict no_std compatibility.
#[allow(non_camel_case_types)]
pub type f16 = u16;

#[inline(always)]
pub fn f16_to_f32(h: f16) -> f32 {
    let mut bits = (h as u32 & 0x7fff) << 13;
    let exp = bits >> 23;
    if exp == 0 {
        if bits != 0 {
            bits |= 0x00800000;
            let mut e = 113;
            while (bits & 0x00800000) == 0 {
                e -= 1;
                bits <<= 1;
            }
            bits &= !0x00800000;
            bits |= e << 23;
        }
    } else {
        bits += 0x38000000;
    }
    bits |= (h as u32 & 0x8000) << 16;
    f32::from_bits(bits)
}

#[inline(always)]
pub fn f32_to_f16(f: f32) -> f16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let frac = (bits >> 13) & 0x3ff;
    
    if exp <= 0 {
        sign as f16 // flush subnormals to zero
    } else if exp >= 31 {
        (sign | 0x7c00) as f16 // infinity
    } else {
        (sign | ((exp as u32) << 10) | frac) as f16
    }
}

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
    pub scales: [f16; 8],
    pub offsets: [i16; 8],
    pub _padding: [u8; 32],
    // 其餘 8 個 Cache Line：存放實際的 bit 流 (每列對應一個 Block)
    pub blocks: [vec101_block; 8], 
}

/// Gemma Q4_0 Block (32 weights packed into 16 bytes + 1 f16 scale = 18 bytes)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: f16,           // Block Scale (Delta)
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
    pub s_stream: *const f32,
    /// Output buffer.
    pub out_buffer: *mut f32,
    /// Pointers to Paged Attention KV blocks.
    pub kv_blocks: *const *const f32,
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
}

unsafe impl Send for vec101_context {}
unsafe impl Sync for vec101_context {}
