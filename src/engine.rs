use alloc::vec::Vec;
use crate::types::{vec101_context, EngineState};
use crate::compute::vec101_compute;

/// Surprisal Index (Cognitive Telemetry)
pub struct SurprisalIndex {
    pub score: f32,
    pub is_outlier: bool,
}

/// A simplified speculative engine for vec101 Dual-Mode execution.
pub struct Vec101Engine;

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
