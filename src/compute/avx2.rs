use crate::types::{vec101_context, f16_to_f32};
extern crate alloc;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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

#[cfg(target_arch = "x86_64")]
pub unsafe fn process_row_avx2_gemv(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut final_sum = 0.0f32;

    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*ctx.w_stream.add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = f16_to_f32(w_super.scales[sub_blk]);
            let w_block = &w_super.blocks[sub_blk];
            
            let mut acc_pos = _mm256_setzero_si256();
            let mut acc_neg = _mm256_setzero_si256();

            for sub in 0..8 {
                let u64_idx = sub / 2;
                let shift_amt = (sub % 2) * 32;
                let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
                let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

                let x_ptr = ctx.x_stream.add(col * 2048 + sub_blk * 256 + sub * 32);
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
            let mut block_sum_pos = 0i32;
            for val in sum_arr_pos.iter() {
                block_sum_pos += val;
            }

            let mut sum_arr_neg = [0i32; 8];
            _mm256_storeu_si256(sum_arr_neg.as_mut_ptr() as *mut __m256i, acc_neg);
            let mut block_sum_neg = 0i32;
            for val in sum_arr_neg.iter() {
                block_sum_neg += val;
            }
            
            final_sum += ((block_sum_pos - block_sum_neg) as f32) * micro_scale;
        }
    }

    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += final_sum * scale;
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn process_row_avx2_gemm(row: usize, ctx: &vec101_context, x_t: &[i8], padded_batch: usize, row_sums: &mut [i32]) {
    // A simplified GEMM fallback to satisfy compilation and the new SuperBlock type
    let scale = *ctx.s_stream.add(row);
    let mut row_sums_f32 = alloc::vec![0.0f32; ctx.batch_size];

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*ctx.w_stream.add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = f16_to_f32(w_super.scales[sub_blk]);
            let w_block = &w_super.blocks[sub_blk];
            
            row_sums.fill(0);
            
            for sub in 0..4 {
                let mut pos_bits = w_block.w_pos_bits[sub];
                while pos_bits != 0 {
                    let tz = pos_bits.trailing_zeros();
                    pos_bits &= pos_bits - 1;
                    let f = col * 2048 + sub_blk * 256 + sub * 64 + tz as usize;
                    for b in 0..ctx.batch_size {
                        row_sums[b] += x_t[f * padded_batch + b] as i32;
                    }
                }
                
                let mut neg_bits = w_block.w_neg_bits[sub];
                while neg_bits != 0 {
                    let tz = neg_bits.trailing_zeros();
                    neg_bits &= neg_bits - 1;
                    let f = col * 2048 + sub_blk * 256 + sub * 64 + tz as usize;
                    for b in 0..ctx.batch_size {
                        row_sums[b] -= x_t[f * padded_batch + b] as i32;
                    }
                }
            }
            
            for b in 0..ctx.batch_size {
                row_sums_f32[b] += (row_sums[b] as f32) * micro_scale;
            }
        }
    }

    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += row_sums_f32[b] * scale;
    }
}
