use alloc::vec::Vec;
use crate::types::{vec101_context, EngineState};
use crate::compute::vec101_compute;

/// Surprisal Index (Cognitive Telemetry)
pub struct SurprisalIndex {
    pub score: f32,
    pub is_outlier: bool,
}

/// A speculative engine for vec101 Dual-Mode execution.
#[cfg(not(feature = "std"))]
pub struct Vec101Engine;

#[cfg(feature = "std")]
use crate::loader::ZeroCopyModelLoader;
#[cfg(feature = "std")]
use crate::tokenizer::TrieTokenizer;
#[cfg(feature = "std")]
use alloc::string::String;

#[cfg(feature = "std")]
pub struct Vec101Engine {
    pub loader: ZeroCopyModelLoader,
    pub tokenizer: TrieTokenizer,
}

#[cfg(feature = "std")]
impl Vec101Engine {
    pub fn new(model_path: &str) -> std::io::Result<Self> {
        let loader = ZeroCopyModelLoader::new(model_path)?;
        let mut tokenizer = TrieTokenizer::new(0);
        // Default init for fallback
        tokenizer.vocab_size = 262144;
        
        Ok(Self { loader, tokenizer })
    }

    /// CanvasDiffusion: Markdown Parallel Generation
    pub fn generate_parallel(&mut self, prompts: &[String]) -> Vec<String> {
        let batch_size = prompts.len();
        
        let mut out_buffer = vec![0.0f32; batch_size * 4096];
        let mut x_stream = vec![0i8; batch_size * 16 * 2048];
        let mut s_stream = vec![1.0f32; batch_size];
        
        // 1. Prepare Batch Context
        let mut ctx = vec101_context {
            quant_type: crate::types::QuantType::Bit1_58, // Default, will update based on data
            w_stream: core::ptr::null(), // Will be linked to loader.model_weights.layers[..]
            x_stream: x_stream.as_ptr(),
            s_stream: s_stream.as_ptr(),
            out_buffer: out_buffer.as_mut_ptr(),
            batch_size,
            num_rows: 4096, // Example hidden dim
            blocks_per_row: 16,
            num_threads: 4,
            state: EngineState::CanvasDiffusion { blocks: batch_size },
        };

        // 2. Map safetensors zero-copy layers dynamically to ctx
        if let Some(first_layer) = self.loader.model_weights.layers.first() {
            match &first_layer.data {
                crate::types::ArchivedLayerData::Bit1_58(blocks) => {
                    ctx.quant_type = crate::types::QuantType::Bit1_58;
                    ctx.w_stream = blocks.as_ptr() as *const u8;
                },
                crate::types::ArchivedLayerData::Q4_0(blocks) => {
                    ctx.quant_type = crate::types::QuantType::Q4_0;
                    ctx.w_stream = blocks.as_ptr() as *const u8;
                }
            }
            // The zero-copy magic happens here: ctx.w_stream directly points to mmap'd physical memory!
        }

        // 3. Execute Batch Inference
        unsafe {
            if !ctx.w_stream.is_null() {
                vec101_compute(&ctx);
            }
        }

        // 4. Return results
        let mut results = Vec::with_capacity(batch_size);
        for (i, p) in prompts.iter().enumerate() {
            let start = i * 4096;
            let logits = &out_buffer[start..start + 4096];
            // Since we don't have the full decoding pipeline here, we just use a sample token id.
            // A true engine would append the decoded string.
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            for (idx, &v) in logits.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = idx;
                }
            }
            results.push(format!("{}\n\n[vec101 Batch {} Generated Content utilizing Zero-Copy engine. Max Logit Idx: {}]", p, i, max_idx));
        }
        
        results
    }
}

#[cfg(not(feature = "std"))]
impl Vec101Engine {
    /// # Safety
    /// Bypasses memory checks; requires perfectly constructed `vec101_context`.
    pub unsafe fn forward_draft(ctx: &mut vec101_context, target_tokens: usize) -> Vec<u32> {
        ctx.state = EngineState::Drafting { target_tokens, layer_skip_stride: 2 };
        
        vec101_compute(ctx);
        
        let logits = core::slice::from_raw_parts(ctx.out_buffer, ctx.num_rows);
        let mut drafted = alloc::vec::Vec::with_capacity(target_tokens);
        
        let (token, _) = Self::sample_with_telemetry(logits);
        for _ in 0..target_tokens {
            drafted.push(token); // For a real MTP it would decode multiple outputs, here we duplicate the single pass logit.
        }
        drafted
    }

    /// # Safety
    /// Bypasses memory checks; requires perfectly constructed `vec101_context`.
    pub unsafe fn forward_verify(ctx: &mut vec101_context, draft_tokens: &[u32]) -> bool {
        let mut tokens_buf = [0u32; 8];
        let len = core::cmp::min(draft_tokens.len(), 8);
        tokens_buf[..len].copy_from_slice(&draft_tokens[..len]);
        
        ctx.state = EngineState::Verifying { draft_tokens: tokens_buf, draft_len: len };
        
        // The batch size is scaled to verify all drafted tokens simultaneously
        let _original_batch = ctx.batch_size;
        ctx.batch_size = len + 1; // 1 (last accepted token) + len (draft tokens)
        
        vec101_compute(ctx);
        
        let mut all_match = true;
        for i in 0..len {
            let logits = core::slice::from_raw_parts(ctx.out_buffer.add(i * ctx.num_rows), ctx.num_rows);
            let (token, _) = Self::sample_with_telemetry(logits);
            if token != draft_tokens[i] {
                all_match = false;
                break;
            }
        }
        all_match
    }

    /// Sampling with Cognitive Telemetry.
    pub fn sample_with_telemetry(logits: &[f32]) -> (u32, SurprisalIndex) {
        // Simplified fallback for telemetry
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        
        (max_idx as u32, SurprisalIndex {
            score: max_val,
            is_outlier: max_val < 0.1, // mock threshold
        })
    }
}
