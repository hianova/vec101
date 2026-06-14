use crate::types::{vec101_context, f16_to_f32};
extern crate alloc;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn expand_bits_to_mask_neon(w_16: u16, bit_mask: uint8x16_t) -> int8x16_t {
    let lo = vdup_n_u8(w_16 as u8);
    let hi = vdup_n_u8((w_16 >> 8) as u8);
    let combined = vcombine_u8(lo, hi);
    vreinterpretq_s8_u8(vtstq_u8(combined, bit_mask))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn process_row_neon_gemv(row: usize, ctx: &vec101_context) {
    match ctx.quant_type {
        crate::types::QuantType::Bit1_58 => process_row_neon_gemv_bit1_58(row, ctx),
        crate::types::QuantType::Q4_0 => process_row_neon_gemv_q4_0(row, ctx),
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemv_bit1_58(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut final_sum = 0.0f32;
    
    let bit_mask_arr: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bit_mask = vld1q_u8(bit_mask_arr.as_ptr());

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*(ctx.w_stream as *const crate::types::Vec101SuperBlock).add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = f16_to_f32(w_super.scales[sub_blk]);
            let w_block = &w_super.blocks[sub_blk];
            let mut acc = vdupq_n_s32(0);
            
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

                let x_ptr = ctx.x_stream.add(col * 2048 + sub_blk * 256 + sub * 32);
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
            let sum = vaddvq_s32(acc);
            final_sum += (sum as f32) * micro_scale;
        }
    }

    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += final_sum * scale;
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemv_q4_0(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut final_sum = 0.0f32;
    
    let q4_blocks_per_row = ctx.blocks_per_row * 8;
    
    let mask = vdupq_n_u8(0x0F);
    let eight = vdupq_n_u8(8);
    
    for col in 0..q4_blocks_per_row {
        let block_idx = row * q4_blocks_per_row + col;
        let w_block = &(*(ctx.w_stream as *const crate::types::BlockQ4_0).add(block_idx));
        
        let q_vec = vld1q_u8(w_block.qs.as_ptr());
        
        let q0_u8 = vandq_u8(q_vec, mask);
        let q0_s8 = vreinterpretq_s8_u8(vsubq_u8(q0_u8, eight));
        
        let q1_u8 = vshrq_n_u8::<4>(q_vec);
        let q1_s8 = vreinterpretq_s8_u8(vsubq_u8(q1_u8, eight));
        
        let x_ptr = ctx.x_stream.add(col * 32);
        let x_vecs = vld2q_s8(x_ptr);
        
        let mut acc = vdupq_n_s32(0);
        core::arch::asm!(
            "sdot {acc:v}.4s, {x0:v}.16b, {w0:v}.16b",
            "sdot {acc:v}.4s, {x1:v}.16b, {w1:v}.16b",
            acc = inout(vreg) acc,
            x0 = in(vreg) x_vecs.0,
            w0 = in(vreg) q0_s8,
            x1 = in(vreg) x_vecs.1,
            w1 = in(vreg) q1_s8,
        );
        
        let block_sum = vaddvq_s32(acc);
        final_sum += (block_sum as f32) * f16_to_f32(w_block.d);
    }
    
    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += final_sum * scale;
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn process_row_neon_gemm(row: usize, ctx: &vec101_context, row_sums: &mut [f32]) {
    match ctx.quant_type {
        crate::types::QuantType::Bit1_58 => process_row_neon_gemm_bit1_58(row, ctx, row_sums),
        crate::types::QuantType::Q4_0 => process_row_neon_gemm_q4_0(row, ctx, row_sums),
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemm_bit1_58(row: usize, ctx: &vec101_context, row_sums: &mut [f32]) {
    let scale = *ctx.s_stream.add(row);
    let in_features = ctx.blocks_per_row * 2048;
    
    let bit_mask_arr: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bit_mask = vld1q_u8(bit_mask_arr.as_ptr());

    for b in 0..ctx.batch_size {
        row_sums[b] = 0.0;
    }

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*(ctx.w_stream as *const crate::types::Vec101SuperBlock).add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = f16_to_f32(w_super.scales[sub_blk]);
            let w_block = &w_super.blocks[sub_blk];

            let mut w_micro = [0i8; 256];
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
                
                let offset = sub * 32;
                vst1q_s8(w_micro.as_mut_ptr().add(offset), w_vec_lo);
                vst1q_s8(w_micro.as_mut_ptr().add(offset + 16), w_vec_hi);
            }

            for b in 0..ctx.batch_size {
                let x_batch_ptr = ctx.x_stream.add(b * in_features);
                let mut acc = vdupq_n_s32(0);

                for chunk in 0..16 { // 16 chunks of 16 bytes = 256 bytes per micro block
                    let offset = col * 2048 + sub_blk * 256 + chunk * 16;
                    let x_val = vld1q_s8(x_batch_ptr.add(offset));
                    let w_val = vld1q_s8(w_micro.as_ptr().add(chunk * 16));

                    core::arch::asm!(
                        "sdot {acc:v}.4s, {x:v}.16b, {w:v}.16b",
                        acc = inout(vreg) acc,
                        x = in(vreg) x_val,
                        w = in(vreg) w_val,
                    );
                }

                let sum = vaddvq_s32(acc);
                row_sums[b] += (sum as f32) * micro_scale;
            }
        }
    }

    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += row_sums[b] * scale;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemm_q4_0(row: usize, ctx: &vec101_context, row_sums: &mut [f32]) {
    let scale = *ctx.s_stream.add(row);
    
    for b in 0..ctx.batch_size {
        row_sums[b] = 0.0;
    }
    
    let q4_blocks_per_row = ctx.blocks_per_row * 8;
    let in_features = q4_blocks_per_row * 32;
    
    let mask = vdupq_n_u8(0x0F);
    let eight = vdupq_n_u8(8);
    
    for b in 0..ctx.batch_size {
        let x_batch_ptr = ctx.x_stream.add(b * in_features);
        
        for col in 0..q4_blocks_per_row {
            let block_idx = row * q4_blocks_per_row + col;
            let w_block = &(*(ctx.w_stream as *const crate::types::BlockQ4_0).add(block_idx));
            
            let q_vec = vld1q_u8(w_block.qs.as_ptr());
            
            let q0_u8 = vandq_u8(q_vec, mask);
            let q0_s8 = vreinterpretq_s8_u8(vsubq_u8(q0_u8, eight));
            
            let q1_u8 = vshrq_n_u8::<4>(q_vec);
            let q1_s8 = vreinterpretq_s8_u8(vsubq_u8(q1_u8, eight));
            
            let x_ptr = x_batch_ptr.add(col * 32);
            let x_vecs = vld2q_s8(x_ptr);
            
            let mut acc = vdupq_n_s32(0);
            core::arch::asm!(
                "sdot {acc:v}.4s, {x0:v}.16b, {w0:v}.16b",
                "sdot {acc:v}.4s, {x1:v}.16b, {w1:v}.16b",
                acc = inout(vreg) acc,
                x0 = in(vreg) x_vecs.0,
                w0 = in(vreg) q0_s8,
                x1 = in(vreg) x_vecs.1,
                w1 = in(vreg) q1_s8,
            );
            
            let block_sum = vaddvq_s32(acc);
            row_sums[b] += (block_sum as f32) * f16_to_f32(w_block.d);
        }
        
        *ctx.out_buffer.add(b * ctx.num_rows + row) += row_sums[b] * scale;
    }
}
