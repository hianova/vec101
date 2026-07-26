use crate::compute::vec101_compute;
use crate::core::vec101_context;

/// A safe wrapper for running multi-layer sequential evaluations.
/// Encapsulates the pointer-swapping logic inside an iterator or loop,
/// reducing alignment and out-of-bounds risks for callers.
pub struct LayerSequenceEvaluator<'a> {
    context: vec101_context,
    layer_weights: &'a [*const u8],
}

impl<'a> LayerSequenceEvaluator<'a> {
    /// Create a new evaluator from a base context and an array of layer weight pointers.
    pub fn new(context: vec101_context, layer_weights: &'a [*const u8]) -> Self {
        Self { context, layer_weights }
    }

    /// Iteratively evaluates all layers in sequence.
    /// Mutates the context's `w_stream` pointer safely for each layer.
    pub fn evaluate_all(&mut self) {
        for &w_ptr in self.layer_weights {
            self.context.w_stream = w_ptr;
            unsafe {
                vec101_compute(&self.context);
            }
        }
    }
}
