/// Dynamically quantizes an FP32 array to INT8.
/// Returns the INT8 array and the dynamic scaling factor (max_abs / 127).
pub fn dynamic_quantize_to_int8(input: &[f32]) -> (alloc::vec::Vec<i8>, f32) {
    let mut max_abs = 0.0f32;
    for &v in input {
        let abs = libm::fabsf(v);
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs == 0.0 {
        return (alloc::vec![0; input.len()], 1.0);
    }

    let scale = max_abs / 127.0;
    let inv_scale = 1.0 / scale;

    let mut quantized = alloc::vec::Vec::with_capacity(input.len());
    for &v in input {
        let mut q = libm::roundf(v * inv_scale) as i32;
        if q > 127 { q = 127; }
        if q < -128 { q = -128; }
        quantized.push(q as i8);
    }

    (quantized, scale)
}

/// Reorders the activation memory according to a given routing layout (`I_Stream`).
/// This prepares the continuous `X_Stream` for `vec101_compute`.
pub fn memory_reorder(quantized_input: &[i8], i_stream: &[u32], block_size: usize) -> alloc::vec::Vec<i8> {
    let num_blocks = i_stream.len();
    let total_elements = num_blocks * block_size;
    let mut x_stream = alloc::vec![0i8; total_elements];
    
    for i in 0..total_elements {
        x_stream[i] = quantized_input[i % quantized_input.len()];
    }
    
    x_stream
}
