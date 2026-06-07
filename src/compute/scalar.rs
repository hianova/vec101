use crate::types::vec101_context;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn process_row_scalar_gemv(row: usize, ctx: &vec101_context) {
    let scale = *ctx.s_stream.add(row);
    let mut block_sum = 0i32;

    for col in 0..ctx.blocks_per_row {
        let block_idx = row * ctx.blocks_per_row + col;
        let w_block = &(*ctx.w_stream.add(block_idx));

        for sub in 0..8 {
            let u64_idx = sub / 2;
            let shift_amt = (sub % 2) * 32;
            let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
            let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

            let x_ptr = ctx.x_stream.add(col * 256 + sub * 32);

            for k in 0..32 {
                let x_val = *x_ptr.add(k) as i32;
                if (w_pos_32 & (1 << k)) != 0 {
                    block_sum += x_val;
                } else if (w_neg_32 & (1 << k)) != 0 {
                    block_sum -= x_val;
                }
            }
        }
    }

    let out_ptr = ctx.out_buffer.add(row);
    *out_ptr += (block_sum as f32) * scale;
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn process_row_scalar_gemm(row: usize, ctx: &vec101_context, x_t: &[i8], padded_batch: usize, row_sums: &mut [i32]) {
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
