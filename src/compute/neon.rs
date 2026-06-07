use crate::types::vec101_context;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn expand_bits_to_mask_neon(w_16: u16) -> int8x16_t {
    let mut mask_arr = [0i8; 16];
    for b in 0..16 {
        if (w_16 & (1 << b)) != 0 {
            mask_arr[b] = -1; // 0xFF
        } else {
            mask_arr[b] = 0x00;
        }
    }
    vld1q_s8(mask_arr.as_ptr())
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn process_row_neon_gemv(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut block_sum_pos = 0i32;
    let mut block_sum_neg = 0i32;
    let mut acc_pos = vdupq_n_s32(0);
    let mut acc_neg = vdupq_n_s32(0);

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_block = &(*ctx.w_stream.add(block_idx));

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let mask_pos_lo = expand_bits_to_mask_neon((w_pos_32 & 0xFFFF) as u16);
            let mask_pos_hi = expand_bits_to_mask_neon((w_pos_32 >> 16) as u16);
            let mask_neg_lo = expand_bits_to_mask_neon((w_neg_32 & 0xFFFF) as u16);
            let mask_neg_hi = expand_bits_to_mask_neon((w_neg_32 >> 16) as u16);

            let x_ptr = ctx.x_stream.add(col * 256 + sub * 32);

            let x_val_lo = vld1q_s8(x_ptr);
            let x_val_hi = vld1q_s8(x_ptr.add(16));

            let x_pos_lo = vandq_s8(x_val_lo, mask_pos_lo);
            let x_pos_hi = vandq_s8(x_val_hi, mask_pos_hi);
            let x_neg_lo = vandq_s8(x_val_lo, mask_neg_lo);
            let x_neg_hi = vandq_s8(x_val_hi, mask_neg_hi);

            let mut tmp_pos = vpaddlq_s16(vpaddlq_s8(x_pos_lo));
            acc_pos = vaddq_s32(acc_pos, tmp_pos);
            tmp_pos = vpaddlq_s16(vpaddlq_s8(x_pos_hi));
            acc_pos = vaddq_s32(acc_pos, tmp_pos);
            
            let mut tmp_neg = vpaddlq_s16(vpaddlq_s8(x_neg_lo));
            acc_neg = vaddq_s32(acc_neg, tmp_neg);
            tmp_neg = vpaddlq_s16(vpaddlq_s8(x_neg_hi));
            acc_neg = vaddq_s32(acc_neg, tmp_neg);
        }
    }

    let sum_pos = vaddvq_s32(acc_pos);
    let sum_neg = vaddvq_s32(acc_neg);
    
    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += ((sum_pos - sum_neg) as f32) * scale;
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn process_row_neon_gemm(row: usize, ctx: &vec101_context, x_t: &[i8], padded_batch: usize, row_sums: &mut [i32]) {
    let scale = *ctx.s_stream.add(row);
    
    let mut pos_indices = [0u16; 4096];
    let mut neg_indices = [0u16; 4096];
    let mut pos_cnt = 0;
    let mut neg_cnt = 0;

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_block = &(*ctx.w_stream.add(block_idx));
        for sub in 0..4 {
            let mut pos_bits = w_block.w_pos_bits[sub];
            while pos_bits != 0 {
                let tz = pos_bits.trailing_zeros();
                pos_bits &= pos_bits - 1;
                pos_indices[pos_cnt] = (col * 256 + sub * 64 + tz as usize) as u16;
                pos_cnt += 1;
            }
            let mut neg_bits = w_block.w_neg_bits[sub];
            while neg_bits != 0 {
                let tz = neg_bits.trailing_zeros();
                neg_bits &= neg_bits - 1;
                neg_indices[neg_cnt] = (col * 256 + sub * 64 + tz as usize) as u16;
                neg_cnt += 1;
            }
        }
    }

    let mut b_start = 0;
    while b_start + 64 <= ctx.batch_size {
        let mut s0 = vdupq_n_s32(0);
        let mut s1 = vdupq_n_s32(0);
        let mut s2 = vdupq_n_s32(0);
        let mut s3 = vdupq_n_s32(0);
        let mut s4 = vdupq_n_s32(0);
        let mut s5 = vdupq_n_s32(0);
        let mut s6 = vdupq_n_s32(0);
        let mut s7 = vdupq_n_s32(0);
        let mut s8 = vdupq_n_s32(0);
        let mut s9 = vdupq_n_s32(0);
        let mut s10 = vdupq_n_s32(0);
        let mut s11 = vdupq_n_s32(0);
        let mut s12 = vdupq_n_s32(0);
        let mut s13 = vdupq_n_s32(0);
        let mut s14 = vdupq_n_s32(0);
        let mut s15 = vdupq_n_s32(0);

        for chunk in pos_indices[..pos_cnt].chunks(256) {
            let mut a0 = vdupq_n_s16(0);
            let mut a1 = vdupq_n_s16(0);
            let mut a2 = vdupq_n_s16(0);
            let mut a3 = vdupq_n_s16(0);
            let mut a4 = vdupq_n_s16(0);
            let mut a5 = vdupq_n_s16(0);
            let mut a6 = vdupq_n_s16(0);
            let mut a7 = vdupq_n_s16(0);

            for &f in chunk {
                let x_ptr = x_t.as_ptr().add((f as usize) * padded_batch + b_start);

                let xa = vld1q_s8(x_ptr);
                a0 = vaddw_s8(a0, vget_low_s8(xa));
                a1 = vaddw_s8(a1, vget_high_s8(xa));

                let xb = vld1q_s8(x_ptr.add(16));
                a2 = vaddw_s8(a2, vget_low_s8(xb));
                a3 = vaddw_s8(a3, vget_high_s8(xb));

                let xc = vld1q_s8(x_ptr.add(32));
                a4 = vaddw_s8(a4, vget_low_s8(xc));
                a5 = vaddw_s8(a5, vget_high_s8(xc));

                let xd = vld1q_s8(x_ptr.add(48));
                a6 = vaddw_s8(a6, vget_low_s8(xd));
                a7 = vaddw_s8(a7, vget_high_s8(xd));
            }

            s0 = vaddw_s16(s0, vget_low_s16(a0));
            s1 = vaddw_s16(s1, vget_high_s16(a0));
            s2 = vaddw_s16(s2, vget_low_s16(a1));
            s3 = vaddw_s16(s3, vget_high_s16(a1));

            s4 = vaddw_s16(s4, vget_low_s16(a2));
            s5 = vaddw_s16(s5, vget_high_s16(a2));
            s6 = vaddw_s16(s6, vget_low_s16(a3));
            s7 = vaddw_s16(s7, vget_high_s16(a3));

            s8 = vaddw_s16(s8, vget_low_s16(a4));
            s9 = vaddw_s16(s9, vget_high_s16(a4));
            s10 = vaddw_s16(s10, vget_low_s16(a5));
            s11 = vaddw_s16(s11, vget_high_s16(a5));

            s12 = vaddw_s16(s12, vget_low_s16(a6));
            s13 = vaddw_s16(s13, vget_high_s16(a6));
            s14 = vaddw_s16(s14, vget_low_s16(a7));
            s15 = vaddw_s16(s15, vget_high_s16(a7));
        }

        for chunk in neg_indices[..neg_cnt].chunks(256) {
            let mut a0 = vdupq_n_s16(0);
            let mut a1 = vdupq_n_s16(0);
            let mut a2 = vdupq_n_s16(0);
            let mut a3 = vdupq_n_s16(0);
            let mut a4 = vdupq_n_s16(0);
            let mut a5 = vdupq_n_s16(0);
            let mut a6 = vdupq_n_s16(0);
            let mut a7 = vdupq_n_s16(0);

            for &f in chunk {
                let x_ptr = x_t.as_ptr().add((f as usize) * padded_batch + b_start);

                let xa = vld1q_s8(x_ptr);
                a0 = vaddw_s8(a0, vget_low_s8(xa));
                a1 = vaddw_s8(a1, vget_high_s8(xa));

                let xb = vld1q_s8(x_ptr.add(16));
                a2 = vaddw_s8(a2, vget_low_s8(xb));
                a3 = vaddw_s8(a3, vget_high_s8(xb));

                let xc = vld1q_s8(x_ptr.add(32));
                a4 = vaddw_s8(a4, vget_low_s8(xc));
                a5 = vaddw_s8(a5, vget_high_s8(xc));

                let xd = vld1q_s8(x_ptr.add(48));
                a6 = vaddw_s8(a6, vget_low_s8(xd));
                a7 = vaddw_s8(a7, vget_high_s8(xd));
            }

            s0 = vsubw_s16(s0, vget_low_s16(a0));
            s1 = vsubw_s16(s1, vget_high_s16(a0));
            s2 = vsubw_s16(s2, vget_low_s16(a1));
            s3 = vsubw_s16(s3, vget_high_s16(a1));

            s4 = vsubw_s16(s4, vget_low_s16(a2));
            s5 = vsubw_s16(s5, vget_high_s16(a2));
            s6 = vsubw_s16(s6, vget_low_s16(a3));
            s7 = vsubw_s16(s7, vget_high_s16(a3));

            s8 = vsubw_s16(s8, vget_low_s16(a4));
            s9 = vsubw_s16(s9, vget_high_s16(a4));
            s10 = vsubw_s16(s10, vget_low_s16(a5));
            s11 = vsubw_s16(s11, vget_high_s16(a5));

            s12 = vsubw_s16(s12, vget_low_s16(a6));
            s13 = vsubw_s16(s13, vget_high_s16(a6));
            s14 = vsubw_s16(s14, vget_low_s16(a7));
            s15 = vsubw_s16(s15, vget_high_s16(a7));
        }

        let sum_ptr = row_sums.as_mut_ptr().add(b_start);
        vst1q_s32(sum_ptr, s0);
        vst1q_s32(sum_ptr.add(4), s1);
        vst1q_s32(sum_ptr.add(8), s2);
        vst1q_s32(sum_ptr.add(12), s3);
        vst1q_s32(sum_ptr.add(16), s4);
        vst1q_s32(sum_ptr.add(20), s5);
        vst1q_s32(sum_ptr.add(24), s6);
        vst1q_s32(sum_ptr.add(28), s7);
        vst1q_s32(sum_ptr.add(32), s8);
        vst1q_s32(sum_ptr.add(36), s9);
        vst1q_s32(sum_ptr.add(40), s10);
        vst1q_s32(sum_ptr.add(44), s11);
        vst1q_s32(sum_ptr.add(48), s12);
        vst1q_s32(sum_ptr.add(52), s13);
        vst1q_s32(sum_ptr.add(56), s14);
        vst1q_s32(sum_ptr.add(60), s15);

        b_start += 64;
    }

    while b_start < ctx.batch_size {
        let mut sum = 0;
        for i in 0..pos_cnt {
            sum += x_t[pos_indices[i] as usize * padded_batch + b_start] as i32;
        }
        for i in 0..neg_cnt {
            sum -= x_t[neg_indices[i] as usize * padded_batch + b_start] as i32;
        }
        row_sums[b_start] = sum;
        b_start += 1;
    }

    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += (row_sums[b] as f32) * scale;
    }
}
