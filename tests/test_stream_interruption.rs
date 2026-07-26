use vec101::core::llm_traits::{LlmLayer, LlmPipeline, WeightProvider};
use core::ops::ControlFlow;

struct MockWeightProvider;
impl WeightProvider for MockWeightProvider {
    type WeightType = u8;
    fn get_weights(&self, _layer_id: usize, _module_name: &str) -> Option<&[Self::WeightType]> {
        None
    }
}

struct MockLayer;
impl LlmLayer<MockWeightProvider> for MockLayer {
    type Error = ();
    fn forward(
        &self,
        _layer_id: usize,
        _weights: &MockWeightProvider,
        _hidden_states: &mut [i8],
        _kv_cache_k: &mut [i8],
        _kv_cache_v: &mut [i8],
        _scratch_buffer: &mut [i8],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct MockPipeline;
impl LlmPipeline<MockWeightProvider, MockLayer> for MockPipeline {
    type Error = ();

    fn generate_step<'a>(
        &self,
        _input_token: u32,
        _weights: &MockWeightProvider,
        _kv_cache: &mut [i8],
        scratch_buffer: &'a mut [i8],
    ) -> Result<&'a [f32], Self::Error> {
        // Return dummy logits pointing to scratch_buffer casted to f32 slice
        let ptr = scratch_buffer.as_mut_ptr() as *mut f32;
        let len = scratch_buffer.len() / 4;
        unsafe { Ok(core::slice::from_raw_parts(ptr, len)) }
    }
}

#[test]
fn test_stream_interruption() {
    let pipeline = MockPipeline;
    let weights = MockWeightProvider;
    let mut kv_cache = [0i8; 1024];
    let mut scratch_buffer = [0i8; 1024];
    
    let mut tokens_generated = 0;
    
    // Sampler always returns token ID 42
    let sampler = |_logits: &[f32]| -> u32 {
        42
    };
    
    use vec101::core::llm_traits::ExecutionBuffers;

    let result = pipeline.generate_stream(
        &[1, 2, 3],
        &weights,
        ExecutionBuffers {
            kv_cache: &mut kv_cache,
            scratch_buffer: &mut scratch_buffer,
        },
        100,
        sampler,
        |_next_token| {
            tokens_generated += 1;
            if tokens_generated == 5 {
                ControlFlow::Break(()) // Interrupt early
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    
    assert!(result.is_ok());
    assert_eq!(tokens_generated, 5, "Generation should stop exactly at 5 tokens");
}
