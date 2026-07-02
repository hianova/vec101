use vec101::core::{Vec101SuperBlock, BlockQ4_0};

pub struct XorShift32(pub u32);
impl XorShift32 {
    pub fn new(seed: u32) -> Self { Self(seed | 1) }
    pub fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    pub fn next_u64(&mut self) -> u64 {
        ((self.next() as u64) << 32) | (self.next() as u64)
    }
    pub fn next_i8(&mut self) -> i8 {
        self.next() as i8
    }
    pub fn next_i32(&mut self) -> i32 {
        self.next() as i32
    }
}

pub fn naive_fp32_compute(
    batch_size: usize,
    num_rows: usize,
    blocks_per_row: usize,
    w_stream: &[Vec101SuperBlock],
    x_stream: &[i8],
    s_stream: &[i32],
    out_buffer: &mut [i32],
) {
    let in_features = blocks_per_row * 2048;
    for b in 0..batch_size {
        for r in 0..num_rows {
            let mut row_sum = 0i32;
            for c in 0..blocks_per_row {
                let w_super = &w_stream[r * blocks_per_row + c];
                for sub_blk in 0..8 {
                    let micro_scale = w_super.scales[sub_blk] as i32;
                    let w_block = &w_super.blocks[sub_blk];
                    let mut micro_sum = 0i32;
                    for sub in 0..8 {
                        let u64_idx = sub / 2;
                        let shift_amt = (sub % 2) * 32;
                        let w_pos_32 = (w_block.w_pos_bits[u64_idx] >> shift_amt) as u32;
                        let w_neg_32 = (w_block.w_neg_bits[u64_idx] >> shift_amt) as u32;

                        for bit in 0..32 {
                            let x_val = x_stream[b * in_features + c * 2048 + sub_blk * 256 + sub * 32 + bit] as i32;
                            let is_pos = (w_pos_32 & (1 << bit)) != 0;
                            let is_neg = (w_neg_32 & (1 << bit)) != 0;
                            let weight = (if is_pos { 1 } else { 0 }) - (if is_neg { 1 } else { 0 });
                            micro_sum += weight * x_val;
                        }
                    }
                    row_sum += (micro_sum * micro_scale) >> 8;
                }
            }
            let scale = s_stream[r];
            out_buffer[b * num_rows + r] += ((row_sum as i64 * scale as i64) >> 16) as i32;
        }
    }
}

pub fn naive_q4_0_compute(
    batch_size: usize,
    num_rows: usize,
    blocks_per_row: usize,
    w_stream: &[BlockQ4_0],
    x_stream: &[i8],
    s_stream: &[i32],
    out_buffer: &mut [i32],
) {
    let q4_blocks_per_row = blocks_per_row * 8;
    let in_features = q4_blocks_per_row * 32;
    for b in 0..batch_size {
        for r in 0..num_rows {
            let mut row_sum = 0i32;
            for col in 0..q4_blocks_per_row {
                let w_block = &w_stream[r * q4_blocks_per_row + col];
                let d = w_block.d as i32;
                let mut block_sum = 0i32;
                for i in 0..16 {
                    let q = w_block.qs[i];
                    let q0 = (q & 0x0F) as i32 - 8;
                    let q1 = (q >> 4) as i32 - 8;
                    
                    let x0 = x_stream[b * in_features + col * 32 + i * 2] as i32;
                    let x1 = x_stream[b * in_features + col * 32 + i * 2 + 1] as i32;
                    
                    block_sum += q0 * x0 + q1 * x1;
                }
                row_sum += (block_sum * d) >> 8;
            }
            let scale = s_stream[r];
            out_buffer[b * num_rows + r] += ((row_sum as i64 * scale as i64) >> 16) as i32;
        }
    }
}
