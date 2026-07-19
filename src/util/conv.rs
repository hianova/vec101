use crate::core::vec101_context;
use crate::compute::vec101_compute;
use crate::util::feeder::pack_weights_to_superblocks;
use crate::core::Vec101SuperBlock;
use crate::core::QuantType;
use alloc::vec::Vec;
use core::ptr;

/// Converts a 4D image tensor into a 2D column matrix, with native padding for vec101 SuperBlocks.
/// 
/// `input` shape: `(batch_size, in_channels, height, width)`
/// `output` buffer size: `batch_size * out_h * out_w * padded_inner_dim`
/// 
/// To remain `no_alloc` compliant, this function requires the caller to provide
/// a pre-allocated mutable slice `output` of the exact required size. The inner
/// dimension (`in_channels * kernel_h * kernel_w`) is padded automatically to `padded_inner_dim`.
pub fn im2col(
    input: &[i8],
    batch_size: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    padded_inner_dim: usize,
    output: &mut [i8],
) {
    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    let expected_len = batch_size * out_h * out_w * padded_inner_dim;
    assert_eq!(output.len(), expected_len, "Output buffer size mismatch");

    let mut out_idx = 0;
    let inner_dim = in_channels * kernel_h * kernel_w;
    assert!(padded_inner_dim >= inner_dim, "Padded inner dim is too small");

    for b in 0..batch_size {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut local_inner = 0;
                for c in 0..in_channels {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = (oh * stride + kh) as isize - padding as isize;
                            let iw = (ow * stride + kw) as isize - padding as isize;
                            
                            let val = if ih >= 0 && ih < height as isize && iw >= 0 && iw < width as isize {
                                let in_idx = b * (in_channels * height * width) 
                                           + c * (height * width) 
                                           + (ih as usize) * width 
                                           + (iw as usize);
                                input[in_idx]
                            } else {
                                0 // Padding
                            };
                            
                            output[out_idx] = val;
                            out_idx += 1;
                            local_inner += 1;
                        }
                    }
                }
                
                // Pad the rest of the inner dimension with 0 to match SuperBlock requirements
                while local_inner < padded_inner_dim {
                    output[out_idx] = 0;
                    out_idx += 1;
                    local_inner += 1;
                }
            }
        }
    }
}

/// Packs standard convolution weights `(C_out, C_in, K_h, K_w)` into `Vec101SuperBlock`s.
/// 
/// The inner dimension `C_in * K_h * K_w` is automatically padded with zeros to be a 
/// multiple of 2048 elements (1 SuperBlock), which is required by `vec101_compute`.
pub fn pack_conv_weights(
    weights: &[i32],
    out_channels: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> Vec<Vec101SuperBlock> {
    let inner_dim = in_channels * kernel_h * kernel_w;
    let padded_inner_dim = inner_dim.div_ceil(2048) * 2048;
    
    let mut padded_weights = Vec::with_capacity(out_channels * padded_inner_dim);
    
    for oc in 0..out_channels {
        let start_idx = oc * inner_dim;
        let end_idx = start_idx + inner_dim;
        
        // Copy actual weights
        padded_weights.extend_from_slice(&weights[start_idx..end_idx]);
        
        // Pad with zeros to a multiple of 2048
        padded_weights.resize(padded_weights.len() + (padded_inner_dim - inner_dim), 0);
    }
    
    // Delegate to existing highly optimized packer
    pack_weights_to_superblocks(&padded_weights)
}

/// Dispatches a Convolution operation down to the `vec101` GEMM engine.
/// 
/// This acts as the operator bridge separating CNN mathematics from LLM logic.
/// The function strictly avoids hidden heap allocations by requiring pre-allocated buffers.
pub fn conv2d_compute(
    input: &[i8],
    packed_weights: &[Vec101SuperBlock],
    batch_size: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    s_stream: &[i32],
    im2col_buffer: &mut [i8],
    out_buffer: &mut [i32],
) {
    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
    
    let inner_dim = in_channels * kernel_h * kernel_w;
    let padded_inner_dim = inner_dim.div_ceil(2048) * 2048;
    
    // 1. Image to Column memory expansion (padded directly into the buffer)
    im2col(
        input, batch_size, in_channels, height, width, 
        kernel_h, kernel_w, stride, padding, padded_inner_dim, im2col_buffer
    );
    
    let num_rows_x = batch_size * out_h * out_w;
    let blocks_per_row = padded_inner_dim / 2048;

    // 2. Build the context manually to ensure no_alloc output buffer handling
    let ctx = vec101_context {
        quant_type: QuantType::Bit1_58,
        w_stream: packed_weights.as_ptr() as *const u8,
        x_stream: im2col_buffer.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_buffer.as_mut_ptr(),
        kv_blocks: ptr::null(),
        num_blocks: 0,
        block_size: 256, // Fixed block size for SuperBlocks
        batch_size: num_rows_x,
        num_rows: out_channels,
        blocks_per_row,
        num_threads: 1, // Single threaded for standalone standard execution
        tree_mask: ptr::null(),
        tree_size: 0,
        hardware_handle: ptr::null_mut(),
    };

    // 3. Dispatch to vec101 Core (GEMM)
    unsafe {
        vec101_compute(&ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_im2col_basic() {
        let input = vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ];
        // 1 batch, 1 channel, 3x3 image, 2x2 kernel, stride 1, pad 0
        let padded_dim = 2048; // Must pad to 2048
        let out_h = 2;
        let out_w = 2;
        let mut im2col_buf = vec![0i8; out_h * out_w * padded_dim];

        im2col(&input, 1, 1, 3, 3, 2, 2, 1, 0, padded_dim, &mut im2col_buf);

        // Window 1: [1, 2, 4, 5]
        assert_eq!(&im2col_buf[0..4], &[1, 2, 4, 5]);
        // Following elements in the row should be padded to 0
        assert_eq!(im2col_buf[4], 0);

        // Window 2: [2, 3, 5, 6]
        assert_eq!(&im2col_buf[padded_dim..padded_dim+4], &[2, 3, 5, 6]);
    }

    #[test]
    fn test_pack_conv_weights() {
        // 1 out_channel, 1 in_channel, 2x2 kernel
        let weights = vec![1, -1, 0, 1]; // Length 4
        let packed = pack_conv_weights(&weights, 1, 1, 2, 2);
        
        assert_eq!(packed.len(), 1); // 1 SuperBlock
        let block0 = packed[0].blocks[0];
        
        assert_eq!(block0.w_pos_bits[0] & 1, 1); // index 0 is 1
        assert_eq!(block0.w_neg_bits[0] & 2, 2); // index 1 is -1
        assert_eq!(block0.w_pos_bits[0] & 8, 8); // index 3 is 1
    }
}
