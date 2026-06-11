use alloc::vec::Vec;
use crate::types::{Vec101SuperBlock};

/// Defines the input streams supplied to the layer.
pub struct Vec101LayerStreams {
    pub w_stream: *const Vec101SuperBlock,
    pub s_stream: *const f32,
    // Add extra context if needed by the caller.
}

/// A JIT interface for decoding weights directly from files (e.g. safetensors)
/// and converting them into vec101 cache-aligned `Vec101SuperBlock` streams.
pub trait WeightProvider {
    /// Given a layer index, dynamically provide the packed Dual-Rail streams.
    fn get_layer_streams(&self, layer_idx: usize) -> Vec101LayerStreams;
}

/// Represents a node in the distributed Daisy Chain pipeline.
pub trait PipelineNode {
    /// Process a stage (part of the model layers) and output the transformed hidden state.
    fn process_stage(&self, hidden_state: &[i8]) -> Vec<i8>;

    /// Tail node execution. Decodes logits and broadcasts the final TokenID.
    fn decode_and_broadcast(&self, final_state: &[i8]) -> u32;
}
