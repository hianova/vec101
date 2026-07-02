use crate::core::vec101_context;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn process_row_scalar_gemv(row: usize, ctx: &vec101_context) {
    if ctx.blocks_per_row == 0 { return; }
    match ctx.quant_type {
        crate::core::QuantType::Bit1_58 => process_row_scalar_gemv_bit1_58(row, ctx),
        crate::core::QuantType::Q4_0 => process_row_scalar_gemv_q4_0(row, ctx),
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn process_row_scalar_gemv_bit1_58(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut final_sum = 0i32;

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*(ctx.w_stream as *const crate::core::Vec101SuperBlock).add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = w_super.scales[sub_blk] as i32;
            let w_block = &w_super.blocks[sub_blk];
            let mut micro_sum = 0i32;

            for sub in 0..8 {
                let u64_idx = sub / 2;
                let shift_amt = (sub % 2) * 32;
                let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
                let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

                let x_ptr = ctx.x_stream.add(col * 2048 + sub_blk * 256 + sub * 32);

                for k in 0..32 {
                    let x_val = *x_ptr.add(k) as i32;
                    if (w_pos_32 & (1 << k)) != 0 {
                        micro_sum += x_val;
                    } else if (w_neg_32 & (1 << k)) != 0 {
                        micro_sum -= x_val;
                    }
                }
            }
            final_sum += (micro_sum * micro_scale) >> 8;
        }
    }

    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += ((final_sum as i64 * scale as i64) >> 16) as i32;
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn process_row_scalar_gemv_q4_0(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut final_sum = 0i32;
    
    let q4_blocks_per_row = ctx.blocks_per_row * 8;
    
    for col in 0..q4_blocks_per_row {
        let block_idx = row * q4_blocks_per_row + col;
        let w_block = &(*(ctx.w_stream as *const crate::core::BlockQ4_0).add(block_idx));
        
        let mut block_sum = 0;
        let mut x_idx = col * 32;
        
        for i in 0..16 {
            let q = w_block.qs[i];
            let q0 = (q & 0x0F) as i32 - 8;
            let q1 = (q >> 4) as i32 - 8;
            
            block_sum += q0 * (ctx.x_stream.add(x_idx) as *const i8).read() as i32;
            block_sum += q1 * (ctx.x_stream.add(x_idx + 1) as *const i8).read() as i32;
            x_idx += 2;
        }
        
        final_sum += (block_sum * w_block.d as i32) >> 8;
    }
    
    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += ((final_sum as i64 * scale as i64) >> 16) as i32;
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn process_row_scalar_gemm(row: usize, ctx: &vec101_context, x_t: &[i8], padded_batch: usize, row_sums: &mut [i32]) {
    if ctx.blocks_per_row == 0 { return; }
    match ctx.quant_type {
        crate::core::QuantType::Bit1_58 => process_row_scalar_gemm_bit1_58(row, ctx, x_t, padded_batch, row_sums),
        crate::core::QuantType::Q4_0 => process_row_scalar_gemm_q4_0(row, ctx, x_t, padded_batch, row_sums),
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn process_row_scalar_gemm_bit1_58(row: usize, ctx: &vec101_context, x_t: &[i8], padded_batch: usize, row_sums: &mut [i32]) {
    let scale = *ctx.s_stream.add(row);
    let mut row_sums_int = alloc::vec![0i32; ctx.batch_size];

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_super = &(*(ctx.w_stream as *const crate::core::Vec101SuperBlock).add(block_idx));

        for sub_blk in 0..8 {
            let micro_scale = w_super.scales[sub_blk] as i32;
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
                row_sums_int[b] += (row_sums[b] * micro_scale) >> 8;
            }
        }
    }

    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += (row_sums_int[b] * scale) >> 16;
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn process_row_scalar_gemm_q4_0(row: usize, ctx: &vec101_context, x_t: &[i8], padded_batch: usize, row_sums: &mut [i32]) {
    let scale = *ctx.s_stream.add(row);
    let mut row_sums_int = alloc::vec![0i32; ctx.batch_size];
    
    let q4_blocks_per_row = ctx.blocks_per_row * 8;
    
    for col in 0..q4_blocks_per_row {
        let block_idx = row * q4_blocks_per_row + col;
        let w_block = &(*(ctx.w_stream as *const crate::core::BlockQ4_0).add(block_idx));
        
        row_sums.fill(0);
        let mut x_idx = col * 32;
        
        for i in 0..16 {
            let q = w_block.qs[i];
            let q0 = (q & 0x0F) as i32 - 8;
            let q1 = (q >> 4) as i32 - 8;
            
            for b in 0..ctx.batch_size {
                row_sums[b] += q0 * (x_t[x_idx * padded_batch + b] as i32);
                row_sums[b] += q1 * (x_t[(x_idx + 1) * padded_batch + b] as i32);
            }
            x_idx += 2;
        }
        
        let micro_scale = w_block.d as i32;
        for b in 0..ctx.batch_size {
            row_sums_int[b] += (row_sums[b] * micro_scale) >> 8;
        }
    }
    
    for b in 0..ctx.batch_size {
        *ctx.out_buffer.add(b * ctx.num_rows + row) += (row_sums_int[b] * scale) >> 16;
    }
}
