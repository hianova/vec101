use no_std_tool::compress::{zigzag_encode_i32, zigzag_decode_u32, leb128_encode_u32, leb128_decode_u32};

pub struct VecTimeSeriesEncoder {
    pub last_value: i32,
}

impl VecTimeSeriesEncoder {
    pub fn new(start_value: i32) -> Self {
        Self { last_value: start_value }
    }

    /// Processes a block of exactly 8 elements using auto-vectorizable loops.
    /// Returns the number of bytes written to `out_buf`.
    /// `out_buf` should have at least 40 bytes of capacity (8 * 5 max LEB128 bytes).
    #[inline(never)]
    pub fn encode_block_8(&mut self, input: &[i32; 8], out_buf: &mut [u8]) -> usize {
        let mut shifted = [0i32; 8];
        let mut deltas = [0i32; 8];
        let mut zigzags = [0u32; 8];

        // Prepare shifted array for vectorized subtraction
        shifted[0] = self.last_value;
        shifted[1..8].copy_from_slice(&input[..7]);
        
        // Auto-vectorizable vector subtraction
        for i in 0..8 {
            deltas[i] = input[i].wrapping_sub(shifted[i]);
        }
        self.last_value = input[7];

        // Auto-vectorizable zigzag encoding (branchless)
        for i in 0..8 {
            zigzags[i] = zigzag_encode_i32(deltas[i]);
        }

        // Scalar packing (LEB128 produces variable length output)
        let mut offset = 0;
        for &zz in &zigzags {
            offset += leb128_encode_u32(zz, &mut out_buf[offset..]);
        }
        
        offset
    }
}

pub struct VecTimeSeriesDecoder {
    pub last_value: i32,
}

impl VecTimeSeriesDecoder {
    pub fn new(start_value: i32) -> Self {
        Self { last_value: start_value }
    }

    /// Decodes exactly 8 elements.
    /// Returns the number of bytes read from `in_buf`.
    pub fn decode_block_8(&mut self, in_buf: &[u8], out_buf: &mut [i32; 8]) -> usize {
        let mut zigzags = [0u32; 8];
        let mut offset = 0;
        
        // Scalar unpacking (LEB128 is variable length)
        for zz in &mut zigzags {
            if let Some((val, bytes_read)) = leb128_decode_u32(&in_buf[offset..]) {
                *zz = val;
                offset += bytes_read;
            } else {
                return 0; // Error or incomplete buffer
            }
        }

        let mut deltas = [0i32; 8];
        // Auto-vectorizable zigzag decoding
        for i in 0..8 {
            deltas[i] = zigzag_decode_u32(zigzags[i]);
        }

        // Prefix sum (cumulative sum) for decoding is harder to vectorize automatically
        // but we can compute it scalarly or rely on standard fast-math unrolling
        let mut current = self.last_value;
        for i in 0..8 {
            current = current.wrapping_add(deltas[i]);
            out_buf[i] = current;
        }
        self.last_value = current;

        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_time_series_encoder() {
        let mut encoder = VecTimeSeriesEncoder::new(1000);
        let mut decoder = VecTimeSeriesDecoder::new(1000);

        let input = [1005, 1010, 1011, 1011, 1008, 1000, 995, 990];
        let mut compressed = [0u8; 40];
        
        let size = encoder.encode_block_8(&input, &mut compressed);
        assert!(size <= 16); // Small deltas should compress well

        let mut output = [0i32; 8];
        let read_size = decoder.decode_block_8(&compressed[..size], &mut output);
        
        assert_eq!(size, read_size);
        assert_eq!(input, output);
    }
}
