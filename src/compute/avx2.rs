use crate::types::vec101_context;

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
    let mut block_sum_pos = 0i32;
    let mut block_sum_neg = 0i32;

    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);
    let mut acc_pos = _mm256_setzero_si256();
    let mut acc_neg = _mm256_setzero_si256();

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_block = &(*ctx.w_stream.add(block_idx));

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let x_ptr = ctx.x_stream.add(col * 256 + sub * 32);
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

    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += ((block_sum_pos - block_sum_neg) as f32) * scale;
}
#[cfg(target_arch = "x86_64")]
pub unsafe fn process_row_avx2_gemm(row: usize, ctx: &vec101_context, x_t: &[i8], padded_batch: usize, row_sums: &mut [i32]) {
    let scale = *ctx.s_stream.add(row);
    row_sums.fill(0);
    
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

    for i in 0..pos_cnt {
        let f = pos_indices[i] as usize;
        for b in 0..ctx.batch_size {
            row_sums[b] += x_t[f * padded_batch + b] as i32;
        }
    }
    for i in 0..neg_cnt {
        let f = neg_indices[i] as usize;
        for b in 0..ctx.batch_size {
            row_sums[b] -= x_t[f * padded_batch + b] as i32;
        }
    }
    
    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += (row_sums[b] as f32) * scale;
    }
}
