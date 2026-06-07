use crate::types::vec101_context;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn expand_bits_to_mask_neon(w_16: u16, bit_mask: uint8x16_t) -> int8x16_t {
    let lo = vdup_n_u8(w_16 as u8);
    let hi = vdup_n_u8((w_16 >> 8) as u8);
    let combined = vcombine_u8(lo, hi);
    // vtstq_u8 returns 0xFF where (combined & bit_mask) != 0
    vreinterpretq_s8_u8(vtstq_u8(combined, bit_mask))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn process_row_neon_gemv(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut acc = vdupq_n_s32(0);
    
    let bit_mask_arr: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bit_mask = vld1q_u8(bit_mask_arr.as_ptr());

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_block = &(*ctx.w_stream.add(block_idx));

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let mask_pos_lo = expand_bits_to_mask_neon((w_pos_32 & 0xFFFF) as u16, bit_mask);
            let mask_pos_hi = expand_bits_to_mask_neon((w_pos_32 >> 16) as u16, bit_mask);
            let mask_neg_lo = expand_bits_to_mask_neon((w_neg_32 & 0xFFFF) as u16, bit_mask);
            let mask_neg_hi = expand_bits_to_mask_neon((w_neg_32 >> 16) as u16, bit_mask);

            // Construct actual i8 weights: w = pos_mask - neg_mask
            // If pos_mask is -1 (0xFF), neg_mask is 0, w = -1 - 0 = -1 (Wait!)
            // We want w=1 when pos. So we need w = neg_mask - pos_mask.
            // If pos=1 (pos_mask=-1, neg_mask=0), w = 0 - (-1) = 1.
            // If neg=1 (pos_mask=0, neg_mask=-1), w = -1 - 0 = -1.
            let w_vec_lo = vsubq_s8(mask_neg_lo, mask_pos_lo);
            let w_vec_hi = vsubq_s8(mask_neg_hi, mask_pos_hi);

            let x_ptr = ctx.x_stream.add(col * 256 + sub * 32);
            let x_val_lo = vld1q_s8(x_ptr);
            let x_val_hi = vld1q_s8(x_ptr.add(16));

            core::arch::asm!(
                "sdot {acc:v}.4s, {x:v}.16b, {w:v}.16b",
                acc = inout(vreg) acc,
                x = in(vreg) x_val_lo,
                w = in(vreg) w_vec_lo,
            );
            
            core::arch::asm!(
                "sdot {acc:v}.4s, {x:v}.16b, {w:v}.16b",
                acc = inout(vreg) acc,
                x = in(vreg) x_val_hi,
                w = in(vreg) w_vec_hi,
            );
        }
    }

    let sum = vaddvq_s32(acc);
    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += (sum as f32) * scale;
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn process_row_neon_gemm(row: usize, ctx: &vec101_context, row_sums: &mut [i32]) {
    let scale = *ctx.s_stream.add(row);
    let in_features = ctx.blocks_per_row * 256;
    
    // Decode the 1.58-bit weights for this row into an i8 buffer ONCE
    // 4096 elements = 4KB, fits perfectly in L1 cache.
    let mut w_i8: alloc::vec::Vec<i8> = alloc::vec::Vec::with_capacity(in_features);
    w_i8.set_len(in_features);
    
    let bit_mask_arr: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bit_mask = vld1q_u8(bit_mask_arr.as_ptr());

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_block = &(*ctx.w_stream.add(block_idx));

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let mask_pos_lo = expand_bits_to_mask_neon((w_pos_32 & 0xFFFF) as u16, bit_mask);
            let mask_pos_hi = expand_bits_to_mask_neon((w_pos_32 >> 16) as u16, bit_mask);
            let mask_neg_lo = expand_bits_to_mask_neon((w_neg_32 & 0xFFFF) as u16, bit_mask);
            let mask_neg_hi = expand_bits_to_mask_neon((w_neg_32 >> 16) as u16, bit_mask);

            let w_vec_lo = vsubq_s8(mask_neg_lo, mask_pos_lo);
            let w_vec_hi = vsubq_s8(mask_neg_hi, mask_pos_hi);
            
            let feature_offset = col * 256 + sub * 32;
            vst1q_s8(w_i8.as_mut_ptr().add(feature_offset), w_vec_lo);
            vst1q_s8(w_i8.as_mut_ptr().add(feature_offset + 16), w_vec_hi);
        }
    }

    // Now blast SDOT over all batches
    for b in 0..ctx.batch_size {
        let mut acc = vdupq_n_s32(0);
        let x_batch_ptr = ctx.x_stream.add(b * in_features);
        let w_ptr = w_i8.as_ptr();

        for chunk in 0..(in_features / 16) {
            let offset = chunk * 16;
            let x_val = vld1q_s8(x_batch_ptr.add(offset));
            let w_val = vld1q_s8(w_ptr.add(offset));

            core::arch::asm!(
                "sdot {acc:v}.4s, {x:v}.16b, {w:v}.16b",
                acc = inout(vreg) acc,
                x = in(vreg) x_val,
                w = in(vreg) w_val,
            );
        }

        let sum = vaddvq_s32(acc);
        row_sums[b] = sum;
    }

    // Write out the scaled results
    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += (row_sums[b] as f32) * scale;
    }
}
