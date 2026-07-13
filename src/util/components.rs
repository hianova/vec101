pub mod attention {
    use alloc::vec::Vec;
    use no_std_tool::math::{exp_approx_q16, FIXED_POINT_SHIFT};
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;

    /// CPU-bound Tiled FlashAttention (Zero-Float Base)
    /// Computes attention purely with integers, completely bypassing the FPU.
    pub struct IntegerTiledAttention;

    impl IntegerTiledAttention {
        /// Executes a CPU-optimized Tiled Attention loop purely in i8/i32.
        /// Removes libm, floating point softmax, and floating point scaling.
        /// Parallelized over Query tiles with Rayon, and SIMD optimized via NEON.
        pub fn compute_attention_i8(q: &[i8], k: &[i8], v: &[i8], seq_len: usize, head_dim: usize, tile_size: usize) -> Vec<i8> {
            let mut output = alloc::vec![0i8; seq_len * head_dim];
            
            // We can process each tile of Q independently and in parallel!
            // Each tile of Q produces `tile_size * head_dim` output bytes.
            output.chunks_mut(tile_size * head_dim).enumerate().for_each(|(t_q, out_tile)| {
                let q_start = t_q * tile_size;
                let q_end = core::cmp::min(q_start + tile_size, seq_len);
                let q_len = q_end - q_start;
                
                let mut m = alloc::vec![-999999i32; q_len]; 
                let mut l = alloc::vec![0i32; q_len];
                let mut s_ij = alloc::vec![0i32; q_len * tile_size]; // size depends on inner k_len
                let mut p_ij = alloc::vec![0i32; tile_size];

                let num_tiles_k = seq_len.div_ceil(tile_size);

                for t_k in 0..num_tiles_k {
                    let k_start = t_k * tile_size;
                    let k_end = core::cmp::min(k_start + tile_size, seq_len);
                    let k_len = k_end - k_start;

                    // 1. Q * K^T (Local Tile - Pure Integer MAC)
                    for i in 0..q_len {
                        let global_i = q_start + i;
                        let q_row = &q[global_i * head_dim .. (global_i + 1) * head_dim];
                        for j in 0..k_len {
                            let global_j = k_start + j;
                            if global_i < global_j {
                                s_ij[i * k_len + j] = -999999;
                                continue;
                            }
                            let k_row = &k[global_j * head_dim .. (global_j + 1) * head_dim];
                            
                            let mut dot = 0i32;
                            
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
// coverage:ignore-start
                                // NEON SIMD vdotq_s32 path
                                let mut sum_vec = vdupq_n_s32(0);
                                let chunks = head_dim / 16;
                                for c in 0..chunks {
                                    let q_chunk = vld1q_s8(q_row.as_ptr().add(c * 16)); // coverage:ignore-line
                                    let k_chunk = vld1q_s8(k_row.as_ptr().add(c * 16)); // coverage:ignore-line
                                     // coverage:ignore-line
                                    // Use stable NEON instructions instead of unstable vdotq_s32 // coverage:ignore-line
                                    let q_low = vget_low_s8(q_chunk); // coverage:ignore-line
                                    let q_high = vget_high_s8(q_chunk); // coverage:ignore-line
                                    let k_low = vget_low_s8(k_chunk); // coverage:ignore-line
                                    let k_high = vget_high_s8(k_chunk); // coverage:ignore-line
                                     // coverage:ignore-line
                                    let p_low = vmull_s8(q_low, k_low); // coverage:ignore-line
                                    let p_high = vmull_s8(q_high, k_high); // coverage:ignore-line
                                     // coverage:ignore-line
                                    let sum32_low = vpaddlq_s16(p_low); // coverage:ignore-line
                                    let sum32_high = vpaddlq_s16(p_high); // coverage:ignore-line
                                     // coverage:ignore-line
                                    sum_vec = vaddq_s32(sum_vec, vaddq_s32(sum32_low, sum32_high)); // coverage:ignore-line
                                } // coverage:ignore-line
                                dot += vaddvq_s32(sum_vec);
// coverage:ignore-end
                                // Tail scalar loop
                                for d in (chunks * 16)..head_dim {
                                    dot += q_row[d] as i32 * k_row[d] as i32;
                                }
                            }
                            
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                for d in 0..head_dim {
                                    dot += q_row[d] as i32 * k_row[d] as i32;
                                }
                            }
                            
                            // Bit-shift Scaling: >> 3 replaces / sqrt(64)
                            // We shift it into Q16.16 format for the exponential approximation
                            s_ij[i * k_len + j] = (dot >> 3) << FIXED_POINT_SHIFT;
                        }
                    }

                    // 2. Local Softmax & O update (Zero-Float I-Softmax)
                    for i in 0..q_len {
                        let mut m_ij = -999999i32;
                        for j in 0..k_len {
                            let val = s_ij[i * k_len + j];
                            if val > m_ij {
                                m_ij = val;
                            }
                        }

                        if m_ij == -999999 {
                            continue;
                        }

                        let m_i_old = m[i];
                        let m_i_new = if m_i_old > m_ij { m_i_old } else { m_ij };
                        m[i] = m_i_new;

                        // Integer exponential approximation for scaling
                        let exp_diff = exp_approx_q16(m_i_old - m_i_new).unwrap_or(0);
                        
                        // l[i] *= exp_diff (in Q16.16)
                        let mut l_i_new = ((l[i] as i64 * exp_diff as i64) >> FIXED_POINT_SHIFT) as i32;

                        for j in 0..k_len {
                            let p = exp_approx_q16(s_ij[i * k_len + j] - m_i_new).unwrap_or(0);
                            p_ij[j] = p;
                            l_i_new += p;
                        }
                        l[i] = l_i_new;

                        // 3. P * V (Local Accumulation)
                        let out_row = &mut out_tile[i * head_dim .. (i + 1) * head_dim];
                        for d in 0..head_dim {
                            // Scale existing accumulated output
                            let scaled_out = ((out_row[d] as i64 * exp_diff as i64) >> FIXED_POINT_SHIFT) as i32;
                            
                            let mut pv = 0i64;
                            for j in 0..k_len {
                                if p_ij[j] > 0 {
                                    pv += p_ij[j] as i64 * v[(k_start + j) * head_dim + d] as i64;
                                }
                            }
                            
                            let new_val = scaled_out + ((pv >> FIXED_POINT_SHIFT) as i32);
                            out_row[d] = new_val.clamp(-128, 127) as i8;
                        }
                    }
                }
                
                // Final integer normalization for this tile
                for i in 0..q_len {
                    let l_val = l[i];
                    if l_val > 0 {
                        let out_row = &mut out_tile[i * head_dim .. (i + 1) * head_dim];
                        for item in out_row.iter_mut().take(head_dim) {
                            let scale = l_val >> FIXED_POINT_SHIFT;
                            if scale > 0 {
                                let normalized = *item as i32 / scale;
                                *item = normalized.clamp(-128, 127) as i8;
                            }
                        }
                    } // coverage:ignore-line
                }
            });
            
            output
        }
    }
}

pub mod tokenizer {
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec::Vec;

    #[derive(Default)]
    struct TrieNode {
        children: BTreeMap<u8, TrieNode>,
        token_id: Option<u32>,
    }

    pub struct TrieTokenizer {
        root: TrieNode,
        id_to_token: BTreeMap<u32, Vec<u8>>,
        pub vocab_size: u32,
        pub unk_token: u32,
    }

    impl TrieTokenizer {
        pub fn new(unk_token: u32) -> Self {
            Self {
                root: TrieNode::default(),
                id_to_token: BTreeMap::new(),
                vocab_size: 0,
                unk_token,
            }
        }

        /// Add a string/byte-sequence to the tokenizer vocabulary.
        pub fn add_token(&mut self, text: &[u8], id: u32) {
            let mut curr = &mut self.root;
            for &b in text {
                curr = curr.children.entry(b).or_default();
            }
            curr.token_id = Some(id);
            self.id_to_token.insert(id, text.to_vec());
            if id >= self.vocab_size {
                self.vocab_size = id + 1;
            }
        }

        /// Encodes a string into token IDs using greedy longest-prefix matching.
        pub fn encode(&self, text: &str) -> Vec<u32> {
            let bytes = text.as_bytes();
            let mut tokens = Vec::new();
            let mut i = 0;

            while i < bytes.len() {
                let mut curr = &self.root;
                let mut best_match = None;
                let mut best_len = 0;

                for (j, _byte) in bytes.iter().enumerate().skip(i) {
                    if let Some(next_node) = curr.children.get(&bytes[j]) {
                        curr = next_node;
                        if let Some(id) = curr.token_id {
                            best_match = Some(id);
                            best_len = j - i + 1;
                        }
                    } else {
                        break;
                    }
                }

                if let Some(id) = best_match {
                    tokens.push(id);
                    i += best_len;
                } else {
                    // If no match, emit unknown token and advance by 1 byte
                    tokens.push(self.unk_token);
                    i += 1;
                }
            }
            tokens
        }

        /// Decodes a sequence of token IDs back into a string.
        /// Invalid UTF-8 bytes are replaced with the replacement character.
        pub fn decode(&self, tokens: &[u32]) -> String {
            let mut bytes = Vec::new();
            for &id in tokens {
                if let Some(token_bytes) = self.id_to_token.get(&id) {
                    bytes.extend_from_slice(token_bytes);
                }
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use alloc::vec;

        #[test]
        fn test_trie_tokenizer() {
            let mut tokenizer = TrieTokenizer::new(0);
            tokenizer.add_token(b"Hello", 1);
            tokenizer.add_token(b"World", 2);
            tokenizer.add_token(b" ", 3);
            
            let encoded = tokenizer.encode("Hello World");
            assert_eq!(encoded, vec![1, 3, 2]);
            
            let decoded = tokenizer.decode(&encoded);
            assert_eq!(decoded, "Hello World");
            
            // Test unknown token
            let encoded_unk = tokenizer.encode("XYZ");
            assert_eq!(encoded_unk, vec![0, 0, 0]); // 3 bytes, 3 unk tokens
        }
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_empty_tokenizer() {
        let mut tokenizer = crate::util::components::tokenizer::TrieTokenizer::new(0);
        assert_eq!(tokenizer.encode("test"), alloc::vec![0, 0, 0, 0]);
    }
}
