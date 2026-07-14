//! Generic traits for constructing LLM pipelines without allocations or I/O.
//! This ensures vec101 stays focused strictly on computation.

use core::result::Result;

/// Provides read-only access to model weights.
/// Implemented by the application layer to feed safetensors or mmapped files into the core.
pub trait WeightProvider {
    type WeightType;

    /// Retrieves a specific weight slice for a given layer and module name.
    /// Returns `None` if the weight is not found.
    fn get_weights(&self, layer_id: usize, module_name: &str) -> Option<&[Self::WeightType]>;
}

/// Represents a single layer (e.g., a Transformer Block) in a neural network.
pub trait LlmLayer<W: WeightProvider> {
    type Error;

    /// Executes the forward pass for this layer.
    ///
    /// All buffers must be pre-allocated by the caller to ensure `no_std` compatibility.
    ///
    /// - `layer_id`: The index of this layer.
    /// - `weights`: The weight provider to fetch projection weights.
    /// - `hidden_states`: The in/out feature vector for the sequence.
    /// - `kv_cache_k`: The key cache buffer for this layer.
    /// - `kv_cache_v`: The value cache buffer for this layer.
    /// - `scratch_buffer`: A temporary working buffer to avoid allocations.
    fn forward(
        &self,
        layer_id: usize,
        weights: &W,
        hidden_states: &mut [i8],
        kv_cache_k: &mut [i8],
        kv_cache_v: &mut [i8],
        scratch_buffer: &mut [i8],
    ) -> Result<(), Self::Error>;
}

/// Represents the full end-to-end LLM generation pipeline.
pub trait LlmPipeline<W: WeightProvider, L: LlmLayer<W>> {
    type Error;

    /// Executes a single generation step for the entire pipeline.
    ///
    /// - `input_token`: The token ID to process.
    /// - `weights`: The weight provider.
    /// - `kv_cache`: The global KV cache (must be sliced/managed by the implementation for each layer).
    /// - `scratch_buffer`: A temporary working buffer for intermediate states (e.g. logits).
    ///
    /// Returns a slice to the computed logits, which will be inside the `scratch_buffer` or another pre-allocated area.
    fn generate_step<'a>(
        &self,
        input_token: u32,
        weights: &W,
        kv_cache: &mut [i8],
        scratch_buffer: &'a mut [i8],
    ) -> Result<&'a [f32], Self::Error>;
}
