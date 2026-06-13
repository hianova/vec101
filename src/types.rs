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
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct vec101_block {
    pub w_pos_bits: [u64; 4],
    pub w_neg_bits: [u64; 4],
}

/// 完美對齊 64-Byte，且維持 256 維度的終極設計！
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Vec101SuperBlock {
    // 第 1 個 Cache Line (64 Bytes)：專門放 Metadata
    // 支援 8 個 Block 的 Scale 和 Offset
    pub scales: [f16; 8],      // 16 Bytes
    pub offsets: [i16; 8],     // 16 Bytes
    pub _padding: [u8; 32],    // 32 Bytes (保留給未來擴充，或放靜態啟動值)

    // 第 2 到 第 9 個 Cache Line (8 * 64 = 512 Bytes)：純粹的權重
    // 8 個 Block * 256 維度 = 2048 維度！
    pub blocks: [vec101_block; 8], 
}

/// 執行狀態與投機解碼控制
#[derive(Debug, Clone, Copy)]
pub enum EngineState {
    /// 草稿模式：跳過特定層數（例如 stride=2 也就是跳過偶數層），利用 MTP 預測後續 Token
    Drafting { target_tokens: usize, layer_skip_stride: usize },
    /// 驗證模式：動態長度的 Draft Token 驗證，補齊被跳過的運算
    Verifying { draft_tokens: [u32; 8], draft_len: usize },
    /// Markdown 擴散模式，Batch Size = N
    CanvasDiffusion { blocks: usize },
}

/// The runtime context for the vec101 engine.
#[repr(C)]
pub struct vec101_context {
    /// Highly compressed 1.58-bit SuperBlocks stream.
    pub w_stream: *const Vec101SuperBlock,
    /// Continuous activation values stream.
    pub x_stream: *const i8,
    /// Quantization scaling factor stream per row.
    pub s_stream: *const f32,
    /// Output buffer.
    pub out_buffer: *mut f32,
    /// Number of tokens processed simultaneously (GEMM Batch Dimension)
    pub batch_size: usize,
    /// Number of rows in the weight matrix
    pub num_rows: usize,
    /// Number of SuperBlocks per row
    pub blocks_per_row: usize,
    /// Number of parallel threads to use
    pub num_threads: usize,
    /// Current execution state (Speculative decoding routing)
    pub state: EngineState,
}

unsafe impl Send for vec101_context {}
unsafe impl Sync for vec101_context {}
