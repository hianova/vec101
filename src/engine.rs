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
        
        // 1. Prepare Batch Context
        let mut ctx = vec101_context {
            quant_type: crate::types::QuantType::Bit1_58, // Default, will update based on data
            w_stream: core::ptr::null(), // Will be linked to loader.model_weights.layers[..]
            x_stream: core::ptr::null(),
            s_stream: core::ptr::null(),
            out_buffer: core::ptr::null_mut(),
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

        // 3. Execute Batch Inference (Mock token loop)
        unsafe {
            // vec101_compute(&ctx);
        }

        // 4. Return results
        let mut results = Vec::with_capacity(batch_size);
        for (i, p) in prompts.iter().enumerate() {
            results.push(format!("{}\n\n[vec101 Batch {} Generated Content utilizing Zero-Copy engine]", p, i));
        }
        
        // Simulating the time taken for a batch inference
        std::thread::sleep(std::time::Duration::from_millis(50));
        results
    }
}

#[cfg(not(feature = "std"))]
impl Vec101Engine {
    /// # Safety
    /// Bypasses memory checks; requires perfectly constructed `vec101_context`.
    pub unsafe fn forward_draft(ctx: &mut vec101_context, target_tokens: usize) -> Vec<u32> {
        ctx.state = EngineState::Drafting { target_tokens, layer_skip_stride: 2 };
        
        // Simulating skipped layer processing (MTP)
        // In real execution, this would only execute vec101_compute on specific layer indices.
        vec101_compute(ctx);
        
        // Mock returning N drafted tokens
        alloc::vec![0; target_tokens]
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
        
        // Simulation: Return true if all accepted.
        true
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
