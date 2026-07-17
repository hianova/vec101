/// Dynamically quantizes an INT32 array to INT8 using pure integer arithmetic.
/// Returns the INT8 array and the dynamic scaling factor.
pub fn dynamic_quantize_to_int8(input: &[i32]) -> (Vec<i8>, i32) {
    let mut max_abs = 0i32;
    for &v in input {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs == 0 {
        return (alloc::vec![0; input.len()], 1);
    }

    let scale_factor = max_abs / 127;
    let mut quantized = Vec::with_capacity(input.len());

    if scale_factor == 0 {
        for &v in input {
            quantized.push(v.clamp(-128, 127) as i8);
        }
        return (quantized, 1);
    }

    for &v in input {
        let mut q = v / scale_factor;
        q = q.clamp(-128, 127);
        quantized.push(q as i8);
    }

    (quantized, scale_factor)
}

/// Reorders the activation memory according to a given routing layout (`I_Stream`).
/// This prepares the continuous `X_Stream` for `vec101_compute`.
pub fn memory_reorder(
    quantized_input: &[i8],
    i_stream: &[u32],
    block_size: usize,
) -> Vec<i8> {
    let num_blocks = i_stream.len();
    let total_elements = num_blocks * block_size;
    let mut x_stream = alloc::vec![0i8; total_elements];

    for (b_idx, &i_val) in i_stream.iter().enumerate() {
        let src_offset = (i_val as usize) * block_size;
        let dst_offset = b_idx * block_size;

        for i in 0..block_size {
            x_stream[dst_offset + i] = quantized_input[(src_offset + i) % quantized_input.len()];
        }
    }
    x_stream
}

use crate::core::{Vec101SuperBlock, vec101_block};

/// Packs INT32 model weights into the highly optimized Dual-Rail `Vec101SuperBlock` format.
/// Applications should use this to load model weights into vec101.
///
/// # Arguments
/// * `weights` - The flattened continuous INT32 weight array. Length must be a multiple of 2048.
///
/// # Panics
/// Panics if `weights.len()` is not a multiple of 2048.
pub fn pack_weights_to_superblocks(weights: &[i32]) -> Vec<Vec101SuperBlock> {
    assert_eq!(
        weights.len() % 2048,
        0,
        "Weight length must be a multiple of 2048 for SuperBlock packing"
    );
    let num_superblocks = weights.len() / 2048;
    let mut superblocks = Vec::with_capacity(num_superblocks);

    for sb_idx in 0..num_superblocks {
        let sb_weights = &weights[sb_idx * 2048..(sb_idx + 1) * 2048];
        let mut sb = Vec101SuperBlock {
            scales: [0; 8],
            offsets: [0; 8],
            blocks: [vec101_block {
                w_pos_bits: [0; 4],
                w_neg_bits: [0; 4],
            }; 8],
        };

        for b_idx in 0..8 {
            let block_w = &sb_weights[b_idx * 256..(b_idx + 1) * 256];

            // BitNet 1.58-bit Quantization: Scale = Mean Absolute Value
            let mut sum_abs = 0i64;
            for &w in block_w {
                sum_abs += (w as i64).abs();
            }
            let mean_abs = sum_abs / 256;
            let scale = if mean_abs == 0 { 1 } else { mean_abs as i32 };

            sb.scales[b_idx] = scale.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

            let mut pos_bits = [0u64; 4];
            let mut neg_bits = [0u64; 4];

            for (i, &w) in block_w.iter().enumerate() {
                let q = w / scale;
                let ternary = q.clamp(-1, 1);

                let u64_idx = i / 64;
                let bit_shift = i % 64;

                if ternary == 1 {
                    pos_bits[u64_idx] |= 1 << bit_shift;
                } else if ternary == -1 {
                    neg_bits[u64_idx] |= 1 << bit_shift;
                }
            }

            sb.blocks[b_idx].w_pos_bits = pos_bits;
            sb.blocks[b_idx].w_neg_bits = neg_bits;
        }

        superblocks.push(sb);
    }

    superblocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_dynamic_quantize() {
        let input = vec![100, 200, -1000, 0];
        let (quantized, scale) = dynamic_quantize_to_int8(&input);

        assert_eq!(scale, 1000 / 127);
        assert_eq!(quantized[2], -128); // Note: -1000 / 7 = -142 -> clamped to -128
        assert_eq!(quantized[3], 0);
    }

    #[test]
    fn test_memory_reorder() {
        let input = vec![1, 2, 3];
        let i_stream = vec![0, 1]; // 2 blocks
        let out = memory_reorder(&input, &i_stream, 2);
        assert_eq!(out, vec![1, 2, 3, 1]);
    }

    #[test]
    fn test_pack_weights_to_superblocks() {
        let mut weights = vec![0i32; 2048];
        weights[0] = 200; // Will be +1
        weights[1] = -200; // Will be -1
        weights[64] = 200; // Next u64 pos_bits

        let superblocks = pack_weights_to_superblocks(&weights);
        assert_eq!(superblocks.len(), 1);

        let sb = &superblocks[0];
        let block0 = &sb.blocks[0];

        assert_eq!(block0.w_pos_bits[0] & 1, 1);
        assert_eq!(block0.w_neg_bits[0] & 2, 2);
        assert_eq!(block0.w_pos_bits[1] & 1, 1);
    }

    #[test]
    fn test_dynamic_quantize_edge_cases() {
        let input_zeros = vec![0, 0, 0];
        let (q_zeros, s_zeros) = dynamic_quantize_to_int8(&input_zeros);
        assert_eq!(s_zeros, 1);

        let input_small = vec![10, -10];
        let (q_small, s_small) = dynamic_quantize_to_int8(&input_small);
        assert_eq!(s_small, 1);
        assert_eq!(q_small[0], 10);
    }
}
