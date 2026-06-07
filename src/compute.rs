use crate::types::vec101_context;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Core computation for vec101.
/// 
/// # Safety
/// If on x86_64, this function requires the CPU to support AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec101_compute(ctx: &vec101_context) {
    let n = ctx.num_blocks;
    let w_stream = ctx.w_stream;
    let x_stream = ctx.x_stream;
    let i_stream = ctx.i_stream;
    let s_stream = ctx.s_stream;
    let out_buffer = ctx.out_buffer;

    for i in 0..n {
        if i + 2 < n {
            _mm_prefetch(w_stream.add(i + 2) as *const i8, _MM_HINT_T0);
            _mm_prefetch(x_stream.add((i + 2) * 256) as *const i8, _MM_HINT_T0);
        }

        let mut block_sum_pos: i32 = 0;
        let mut block_sum_neg: i32 = 0;
        let w_block = &(*w_stream.add(i));

        let ones_u8 = _mm256_set1_epi8(1);
        let ones_i16 = _mm256_set1_epi16(1);
        let mut acc_pos = _mm256_setzero_si256();
        let mut acc_neg = _mm256_setzero_si256();

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let x_ptr = x_stream.add(i * 256 + sub * 32);
            let x_val = _mm256_loadu_si256(x_ptr as *const __m256i);

            let mask_pos = expand_bits_to_mask(w_pos_32);
            let mask_neg = expand_bits_to_mask(w_neg_32);

            let x_pos = _mm256_and_si256(x_val, mask_pos);
            let x_neg = _mm256_and_si256(x_val, mask_neg);

            let sum16_pos = _mm256_maddubs_epi16(ones_u8, x_pos);
            let sum32_pos = _mm256_madd_epi16(sum16_pos, ones_i16);
            acc_pos = _mm256_add_epi32(acc_pos, sum32_pos);

            let sum16_neg = _mm256_maddubs_epi16(ones_u8, x_neg);
            let sum32_neg = _mm256_madd_epi16(sum16_neg, ones_i16);
            acc_neg = _mm256_add_epi32(acc_neg, sum32_neg);
        }

        let mut sum_arr_pos = [0i32; 8];
        _mm256_storeu_si256(sum_arr_pos.as_mut_ptr() as *mut __m256i, acc_pos);
        for val in sum_arr_pos.iter() {
            block_sum_pos += val;
        }

        let mut sum_arr_neg = [0i32; 8];
        _mm256_storeu_si256(sum_arr_neg.as_mut_ptr() as *mut __m256i, acc_neg);
        for val in sum_arr_neg.iter() {
            block_sum_neg += val;
        }

        let block_sum = block_sum_pos - block_sum_neg;
        let target_idx = *i_stream.add(i);
        let scale = *s_stream.add(i);
        let out_ptr = out_buffer.add(target_idx as usize);
        *out_ptr += (block_sum as f32) * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn expand_bits_to_mask(w_32: u32) -> __m256i {
    let mut mask_arr = [0u8; 32];
    for b in 0..32 {
        if (w_32 & (1 << b)) != 0 {
            mask_arr[b] = 0xFF;
        } else {
            mask_arr[b] = 0x00;
        }
    }
    _mm256_loadu_si256(mask_arr.as_ptr() as *const __m256i)
}

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec101_compute(ctx: &vec101_context) { unsafe {
    let n = ctx.num_blocks;
    let w_stream = ctx.w_stream;
    let x_stream = ctx.x_stream;
    let i_stream = ctx.i_stream;
    let s_stream = ctx.s_stream;
    let out_buffer = ctx.out_buffer;

    for i in 0..n {
        let mut block_sum_pos: i32 = 0;
        let mut block_sum_neg: i32 = 0;
        let w_block = &(*w_stream.add(i));

        let mut acc_pos = vdupq_n_s32(0);
        let mut acc_neg = vdupq_n_s32(0);

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let x_ptr = x_stream.add(i * 256 + sub * 32);

            let x_val_lo = vld1q_s8(x_ptr);
            let x_val_hi = vld1q_s8(x_ptr.add(16));

            let mask_pos_lo = expand_bits_to_mask_neon((w_pos_32 & 0xFFFF) as u16);
            let mask_pos_hi = expand_bits_to_mask_neon((w_pos_32 >> 16) as u16);
            let mask_neg_lo = expand_bits_to_mask_neon((w_neg_32 & 0xFFFF) as u16);
            let mask_neg_hi = expand_bits_to_mask_neon((w_neg_32 >> 16) as u16);

            let x_pos_lo = vandq_s8(x_val_lo, mask_pos_lo);
            let x_pos_hi = vandq_s8(x_val_hi, mask_pos_hi);
            let x_neg_lo = vandq_s8(x_val_lo, mask_neg_lo);
            let x_neg_hi = vandq_s8(x_val_hi, mask_neg_hi);

            acc_pos = vaddq_s32(acc_pos, vpaddlq_s16(vpaddlq_s8(x_pos_lo)));
            acc_pos = vaddq_s32(acc_pos, vpaddlq_s16(vpaddlq_s8(x_pos_hi)));
            
            acc_neg = vaddq_s32(acc_neg, vpaddlq_s16(vpaddlq_s8(x_neg_lo)));
            acc_neg = vaddq_s32(acc_neg, vpaddlq_s16(vpaddlq_s8(x_neg_hi)));
        }

        block_sum_pos += vaddvq_s32(acc_pos);
        block_sum_neg += vaddvq_s32(acc_neg);

        let block_sum = block_sum_pos - block_sum_neg;
        let target_idx = *i_stream.add(i);
        let scale = *s_stream.add(i);
        let out_ptr = out_buffer.add(target_idx as usize);
        *out_ptr += (block_sum as f32) * scale;
    }
}}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn expand_bits_to_mask_neon(w_16: u16) -> int8x16_t { unsafe {
    let mut mask_arr = [0i8; 16];
    for b in 0..16 {
        if (w_16 & (1 << b)) != 0 {
            mask_arr[b] = -1; // 0xFF
        } else {
            mask_arr[b] = 0x00;
        }
    }
    vld1q_s8(mask_arr.as_ptr())
}}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn vec101_compute(ctx: &vec101_context) { unsafe {
    let n = ctx.num_blocks;
    let w_stream = ctx.w_stream;
    let x_stream = ctx.x_stream;
    let i_stream = ctx.i_stream;
    let s_stream = ctx.s_stream;
    let out_buffer = ctx.out_buffer;

    for i in 0..n {
        let mut block_sum_pos: i32 = 0;
        let mut block_sum_neg: i32 = 0;
        let w_block = &(*w_stream.add(i));

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let x_ptr = x_stream.add(i * 256 + sub * 32);

            for b in 0..32 {
                let x_val = *x_ptr.add(b) as i32;
                if (w_pos_32 & (1 << b)) != 0 {
                    block_sum_pos += x_val;
                }
                if (w_neg_32 & (1 << b)) != 0 {
                    block_sum_neg += x_val;
                }
            }
        }

        let block_sum = block_sum_pos - block_sum_neg;
        let target_idx = *i_stream.add(i);
        let scale = *s_stream.add(i);
        let out_ptr = out_buffer.add(target_idx as usize);
        *out_ptr += (block_sum as f32) * scale;
    }
}}
