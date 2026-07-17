/// Interface for ingesting continuous floating-point sensor streams.
///
/// Since `vec101` focuses on the 1.58-bit/Q4 compute engine, this trait defines
/// the contract for external quantization tools (like those in `no_std_tool`)
/// to hook into and provide pre-quantized sequences (e.g. i8 arrays + i32 scales)
/// without `vec101` needing to reimplement floating-point normalization.
pub trait ContinuousInput {
    /// Takes a continuous float array (e.g. from an audio or robotic sensor)
    /// and returns the quantized `x_stream` bytes and corresponding `s_stream` scales.
    fn quantize_continuous_stream(
        &self,
        raw_input: &[f32],
    ) -> (Vec<i8>, Vec<i32>);
}

impl crate::core::engine::Vec101Engine {
    /// Feeds a continuous sensor stream directly into the context by leveraging
    /// a provided quantizer implementation, bypassing text tokenization.
    pub fn feed_continuous_inputs<Q: ContinuousInput>(&mut self, quantizer: &Q, raw_input: &[f32]) {
        let (x_stream, s_stream) = quantizer.quantize_continuous_stream(raw_input);

        // In a full implementation, we'd persist these vecs inside the engine
        // so their pointers remain valid during compute, e.g. using an internal
        // buffer or arena. For now, we update the context pointers.
        // WARNING: Pointer lifetime is only safe if engine owns or borrows properly.
        // As a prototype architectural shell, we just expose the interface.
        self.ctx.x_stream = x_stream.as_ptr();
        self.ctx.s_stream = s_stream.as_ptr();

        // Ensure the engine knows how many tokens (or sensor frames) it's processing
        // self.ctx.batch_size = ...;
    }
}
