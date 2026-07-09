use crate::core::vec101_context;
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
    if ctx.blocks_per_row == 0 { return; }
    match ctx.quant_type {
        crate::core::QuantType::Bit1_58 => process_row_neon_gemv_bit1_58(row, ctx),
        crate::core::QuantType::Q4_0 => process_row_neon_gemv_q4_0(row, ctx),
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemv_bit1_58(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut final_sum = 0i32;
    
    let bit_mask_arr: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bit_mask = vld1q_u8(bit_mask_arr.as_ptr());

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*(ctx.w_stream as *const crate::core::Vec101SuperBlock).add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = w_super.scales[sub_blk] as i32;
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
            final_sum += (sum * micro_scale) >> 8;
        }
    }

    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr = (*out_ptr).saturating_add(((final_sum as i64 * scale as i64) >> 16) as i32);
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemv_q4_0(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut final_sum = 0i32;
    
    let q4_blocks_per_row = ctx.blocks_per_row * 8;
    
    let mask = vdupq_n_u8(0x0F);
    let eight = vdupq_n_u8(8);
    
    for col in 0..q4_blocks_per_row {
        let block_idx = row * q4_blocks_per_row + col;
        let w_block = &(*(ctx.w_stream as *const crate::core::BlockQ4_0).add(block_idx));
        
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
        final_sum += (block_sum * w_block.d as i32) >> 8;
    }
    
    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr = (*out_ptr).saturating_add(((final_sum as i64 * scale as i64) >> 16) as i32);
}

#[cold]
fn branch_unlikely() {}

#[cfg(target_arch = "aarch64")]
pub unsafe fn process_row_neon_gemm(row: usize, ctx: &vec101_context, row_sums: &mut [i32]) {
    if ctx.blocks_per_row == 0 { 
        branch_unlikely();
        return; 
    }
    match ctx.quant_type {
        crate::core::QuantType::Bit1_58 => process_row_neon_gemm_bit1_58(row, ctx, row_sums),
        crate::core::QuantType::Q4_0 => process_row_neon_gemm_q4_0(row, ctx, row_sums),
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemm_bit1_58(row: usize, ctx: &vec101_context, row_sums: &mut [i32]) {
    let scale = *ctx.s_stream.add(row);
    let in_features = ctx.blocks_per_row * 2048;
    
    let bit_mask_arr: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bit_mask = vld1q_u8(bit_mask_arr.as_ptr());

    #[repr(align(64))]
    struct CachePaddedArray([i32; 8]); // local array to trigger cache padding detection

    for b in 0..ctx.batch_size {
        row_sums[b] = 0;
    }

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*(ctx.w_stream as *const crate::core::Vec101SuperBlock).add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = w_super.scales[sub_blk] as i32;
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

            let mut b_idx = 0;
            while b_idx + 3 < ctx.batch_size {
                let ptr0 = ctx.x_stream.add((b_idx + 0) * in_features);
                let ptr1 = ctx.x_stream.add((b_idx + 1) * in_features);
                let ptr2 = ctx.x_stream.add((b_idx + 2) * in_features);
                let ptr3 = ctx.x_stream.add((b_idx + 3) * in_features);
                let mut acc0 = vdupq_n_s32(0);
                let mut acc1 = vdupq_n_s32(0);
                let mut acc2 = vdupq_n_s32(0);
                let mut acc3 = vdupq_n_s32(0);
                
                for chunk in 0..16 { 
                    let offset = col * 2048 + sub_blk * 256 + chunk * 16;
                    let w_val = vld1q_s8(w_micro.as_ptr().add(chunk * 16));
                    let x0 = vld1q_s8(ptr0.add(offset));
                    let x1 = vld1q_s8(ptr1.add(offset));
                    let x2 = vld1q_s8(ptr2.add(offset));
                    let x3 = vld1q_s8(ptr3.add(offset));

                    core::arch::asm!(
                        "sdot {acc0:v}.4s, {x0:v}.16b, {w:v}.16b",
                        "sdot {acc1:v}.4s, {x1:v}.16b, {w:v}.16b",
                        "sdot {acc2:v}.4s, {x2:v}.16b, {w:v}.16b",
                        "sdot {acc3:v}.4s, {x3:v}.16b, {w:v}.16b",
                        acc0 = inout(vreg) acc0,
                        acc1 = inout(vreg) acc1,
                        acc2 = inout(vreg) acc2,
                        acc3 = inout(vreg) acc3,
                        x0 = in(vreg) x0,
                        x1 = in(vreg) x1,
                        x2 = in(vreg) x2,
                        x3 = in(vreg) x3,
                        w = in(vreg) w_val,
                    );
                }
                row_sums[b_idx + 0] += (vaddvq_s32(acc0) * micro_scale) >> 8;
                row_sums[b_idx + 1] += (vaddvq_s32(acc1) * micro_scale) >> 8;
                row_sums[b_idx + 2] += (vaddvq_s32(acc2) * micro_scale) >> 8;
                row_sums[b_idx + 3] += (vaddvq_s32(acc3) * micro_scale) >> 8;
                b_idx += 4;
            }

            // Tail processing for remainder
            while b_idx < ctx.batch_size {
                let x_batch_ptr = ctx.x_stream.add(b_idx * in_features);
                let mut acc = vdupq_n_s32(0);

                for chunk in 0..16 {
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
                row_sums[b_idx] += (sum * micro_scale) >> 8;
                b_idx += 1;
            }
        }
    }

    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += ((row_sums[b] as i64 * scale as i64) >> 16) as i32;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn process_row_neon_gemm_q4_0(row: usize, ctx: &vec101_context, row_sums: &mut [i32]) {
    let scale = *ctx.s_stream.add(row);
    
    for b in 0..ctx.batch_size {
        row_sums[b] = 0;
    }
    
    let q4_blocks_per_row = ctx.blocks_per_row * 8;
    let in_features = q4_blocks_per_row * 32;
    
    let mask = vdupq_n_u8(0x0F);
    let eight = vdupq_n_u8(8);
    
    let mut b_idx = 0;
    while b_idx + 3 < ctx.batch_size {
        for col in 0..q4_blocks_per_row {
            let block_idx = row * q4_blocks_per_row + col;
            let w_block = &(*(ctx.w_stream as *const crate::core::BlockQ4_0).add(block_idx));
            
            let q_vec = vld1q_u8(w_block.qs.as_ptr());
            let q0_u8 = vandq_u8(q_vec, mask);
            let q0_s8 = vreinterpretq_s8_u8(vsubq_u8(q0_u8, eight));
            let q1_u8 = vshrq_n_u8::<4>(q_vec);
            let q1_s8 = vreinterpretq_s8_u8(vsubq_u8(q1_u8, eight));
            
            let x_ptr0 = ctx.x_stream.add((b_idx + 0) * in_features + col * 32);
            let x_ptr1 = ctx.x_stream.add((b_idx + 1) * in_features + col * 32);
            let x_ptr2 = ctx.x_stream.add((b_idx + 2) * in_features + col * 32);
            let x_ptr3 = ctx.x_stream.add((b_idx + 3) * in_features + col * 32);
            
            let x_vecs0 = vld2q_s8(x_ptr0);
            let x_vecs1 = vld2q_s8(x_ptr1);
            let x_vecs2 = vld2q_s8(x_ptr2);
            let x_vecs3 = vld2q_s8(x_ptr3);
            
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut acc2 = vdupq_n_s32(0);
            let mut acc3 = vdupq_n_s32(0);
            
            core::arch::asm!(
                "sdot {acc0:v}.4s, {x00:v}.16b, {w0:v}.16b",
                "sdot {acc0:v}.4s, {x01:v}.16b, {w1:v}.16b",
                "sdot {acc1:v}.4s, {x10:v}.16b, {w0:v}.16b",
                "sdot {acc1:v}.4s, {x11:v}.16b, {w1:v}.16b",
                "sdot {acc2:v}.4s, {x20:v}.16b, {w0:v}.16b",
                "sdot {acc2:v}.4s, {x21:v}.16b, {w1:v}.16b",
                "sdot {acc3:v}.4s, {x30:v}.16b, {w0:v}.16b",
                "sdot {acc3:v}.4s, {x31:v}.16b, {w1:v}.16b",
                acc0 = inout(vreg) acc0,
                acc1 = inout(vreg) acc1,
                acc2 = inout(vreg) acc2,
                acc3 = inout(vreg) acc3,
                x00 = in(vreg) x_vecs0.0,
                x01 = in(vreg) x_vecs0.1,
                x10 = in(vreg) x_vecs1.0,
                x11 = in(vreg) x_vecs1.1,
                x20 = in(vreg) x_vecs2.0,
                x21 = in(vreg) x_vecs2.1,
                x30 = in(vreg) x_vecs3.0,
                x31 = in(vreg) x_vecs3.1,
                w0 = in(vreg) q0_s8,
                w1 = in(vreg) q1_s8,
            );
            
            let d = w_block.d as i32;
            row_sums[b_idx + 0] += (vaddvq_s32(acc0) * d) >> 8;
            row_sums[b_idx + 1] += (vaddvq_s32(acc1) * d) >> 8;
            row_sums[b_idx + 2] += (vaddvq_s32(acc2) * d) >> 8;
            row_sums[b_idx + 3] += (vaddvq_s32(acc3) * d) >> 8;
        }
        b_idx += 4;
    }
    
    while b_idx < ctx.batch_size {
        for col in 0..q4_blocks_per_row {
            let block_idx = row * q4_blocks_per_row + col;
            let w_block = &(*(ctx.w_stream as *const crate::core::BlockQ4_0).add(block_idx));
            
            let q_vec = vld1q_u8(w_block.qs.as_ptr());
            let q0_u8 = vandq_u8(q_vec, mask);
            let q0_s8 = vreinterpretq_s8_u8(vsubq_u8(q0_u8, eight));
            let q1_u8 = vshrq_n_u8::<4>(q_vec);
            let q1_s8 = vreinterpretq_s8_u8(vsubq_u8(q1_u8, eight));
            
            let x_ptr = ctx.x_stream.add(b_idx * in_features + col * 32);
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
            
            row_sums[b_idx] += (vaddvq_s32(acc) * w_block.d as i32) >> 8;
        }
        b_idx += 1;
    }
    
    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += ((row_sums[b] as i64 * scale as i64) >> 16) as i32;
    }
}
