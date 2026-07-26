use crate::compute::vec101_compute;
use crate::core::QuantType;
use crate::core::Vec101SuperBlock;
use crate::core::vec101_context;
use crate::util::feeder::pack_weights_to_superblocks;
use alloc::vec::Vec;
use core::ptr;

#[doc = " Parameters for image-to-column conversion."]
pub struct Im2ColParams<'a> {
    pub input: &'a [i8],
    pub batch_size: usize,
    pub in_channels: usize,
    pub height: usize,
    pub width: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
    pub padded_inner_dim: usize,
    pub output: &'a mut [i8],
}

#[doc = " Parameters for 2D convolution computation."]
pub struct Conv2dParams<'a> {
    pub input: &'a [i8],
    pub packed_weights: &'a [Vec101SuperBlock],
    pub batch_size: usize,
    pub in_channels: usize,
    pub height: usize,
    pub width: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: usize,
    pub padding: usize,
    pub s_stream: &'a [i32],
    pub im2col_buffer: &'a mut [i8],
    pub out_buffer: &'a mut [i32],
}

#[doc = " Converts a 4D image tensor into a 2D column matrix, with native padding for vec101 SuperBlocks."]
pub fn im2col(p: Im2ColParams<'_>) {
    let out_h = (p.height + 2 * p.padding - p.kernel_h) / p.stride + 1;
    let out_w = (p.width + 2 * p.padding - p.kernel_w) / p.stride + 1;
    let expected_len = p.batch_size * out_h * out_w * p.padded_inner_dim;
    assert_eq!(p.output.len(), expected_len, "Output buffer size mismatch");
    let mut out_idx = 0;
    let inner_dim = p.in_channels * p.kernel_h * p.kernel_w;
    assert!(
        p.padded_inner_dim >= inner_dim,
        "Padded inner dim is too small"
    );
    for b in 0..p.batch_size {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut local_inner = 0;
                for c in 0..p.in_channels {
                    for kh in 0..p.kernel_h {
                        for kw in 0..p.kernel_w {
                            let ih = (oh * p.stride + kh) as isize - p.padding as isize;
                            let iw = (ow * p.stride + kw) as isize - p.padding as isize;
                            let val = if ih >= 0
                                && ih < p.height as isize
                                && iw >= 0
                                && iw < p.width as isize
                            {
                                let in_idx = b * (p.in_channels * p.height * p.width)
                                    + c * (p.height * p.width)
                                    + (ih as usize) * p.width
                                    + (iw as usize);
                                p.input[in_idx]
                            } else {
                                0
                            };
                            p.output[out_idx] = val;
                            out_idx += 1;
                            local_inner += 1;
                        }
                    }
                }
                while local_inner < p.padded_inner_dim {
                    p.output[out_idx] = 0;
                    out_idx += 1;
                    local_inner += 1;
                }
            }
        }
    }
}
#[doc = " Packs standard convolution weights `(C_out, C_in, K_h, K_w)` into `Vec101SuperBlock`s."]
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
        padded_weights.extend_from_slice(&weights[start_idx..end_idx]);
        padded_weights.resize(padded_weights.len() + (padded_inner_dim - inner_dim), 0);
    }
    pack_weights_to_superblocks(&padded_weights)
}
#[doc = " Dispatches a Convolution operation down to the `vec101` GEMM engine."]
pub fn conv2d_compute(p: Conv2dParams<'_>) {
    let out_h = (p.height + 2 * p.padding - p.kernel_h) / p.stride + 1;
    let out_w = (p.width + 2 * p.padding - p.kernel_w) / p.stride + 1;
    let inner_dim = p.in_channels * p.kernel_h * p.kernel_w;
    let padded_inner_dim = inner_dim.div_ceil(2048) * 2048;
    im2col(Im2ColParams {
        input: p.input,
        batch_size: p.batch_size,
        in_channels: p.in_channels,
        height: p.height,
        width: p.width,
        kernel_h: p.kernel_h,
        kernel_w: p.kernel_w,
        stride: p.stride,
        padding: p.padding,
        padded_inner_dim,
        output: p.im2col_buffer,
    });
    let num_rows_x = p.batch_size * out_h * out_w;
    let blocks_per_row = padded_inner_dim / 2048;
    let ctx = vec101_context {
        quant_type: QuantType::Bit1_58,
        w_stream: p.packed_weights.as_ptr() as *const u8,
        x_stream: p.im2col_buffer.as_ptr(),
        s_stream: p.s_stream.as_ptr(),
        out_buffer: p.out_buffer.as_mut_ptr(),
        kv_blocks: ptr::null(),
        num_blocks: 0,
        block_size: 256,
        batch_size: num_rows_x,
        num_rows: p.out_channels,
        blocks_per_row,
        num_threads: 1,
        tree_mask: ptr::null(),
        tree_size: 0,
        hardware_handle: ptr::null_mut(),
        enable_liquid: false,
        dt: 0.0,
        liquid_state: core::ptr::null_mut(),
        liquid_tau: core::ptr::null(),
        liquid_out_buffer: core::ptr::null_mut(),
        scratch_buffer: core::ptr::null_mut(),
        scratch_size: 0,
    };
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
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let padded_dim = 2048;
        let out_h = 2;
        let out_w = 2;
        let mut im2col_buf = vec![0i8; out_h * out_w * padded_dim];
        im2col(Im2ColParams {
            input: &input,
            batch_size: 1,
            in_channels: 1,
            height: 3,
            width: 3,
            kernel_h: 2,
            kernel_w: 2,
            stride: 1,
            padding: 0,
            padded_inner_dim: padded_dim,
            output: &mut im2col_buf,
        });
        assert_eq!(&im2col_buf[0..4], &[1, 2, 4, 5]);
        assert_eq!(im2col_buf[4], 0);
        assert_eq!(&im2col_buf[padded_dim..padded_dim + 4], &[2, 3, 5, 6]);
    }
    #[test]
    fn test_pack_conv_weights() {
        let weights = vec![1, -1, 0, 1];
        let packed = pack_conv_weights(&weights, 1, 1, 2, 2);
        assert_eq!(packed.len(), 1);
        let block0 = packed[0].blocks[0];
        assert_eq!(block0.w_pos_bits[0] & 1, 1);
        assert_eq!(block0.w_neg_bits[0] & 2, 2);
        assert_eq!(block0.w_pos_bits[0] & 8, 8);
    }
}
