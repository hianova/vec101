pub mod memory_tracker {
    use core::sync::atomic::{AtomicU32, Ordering};

    // Global counters for memory leaks and thread drop checks using AtomicU32
    static RESOURCE_COUNT: AtomicU32 = AtomicU32::new(0);
    static THREAD_ACTIVE_COUNT: AtomicU32 = AtomicU32::new(0);

    /// A scoped resource tracker to ensure everything is dropped correctly.
    /// Since we operate in a `no_alloc` environment, this helps track lifetimes
    /// of important structures across threads.
    pub struct ScopedResource;

    impl ScopedResource {
        #[inline]
        pub fn new() -> Self {
            RESOURCE_COUNT.fetch_add(1, Ordering::SeqCst);
            THREAD_ACTIVE_COUNT.fetch_add(1, Ordering::SeqCst);
            Self
        }
    }

    impl Default for ScopedResource {
        #[inline]
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for ScopedResource {
        #[inline]
        fn drop(&mut self) {
            RESOURCE_COUNT.fetch_sub(1, Ordering::SeqCst);
            THREAD_ACTIVE_COUNT.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Checks if there are any memory leaks (resources not dropped).
    /// Returns true if there are no leaks.
    #[inline]
    pub fn check_memory_leaks() -> bool {
        RESOURCE_COUNT.load(Ordering::SeqCst) == 0
    }

    /// Checks if all threads/operations have correctly dropped.
    /// Returns true if all are dropped.
    #[inline]
    pub fn check_thread_drops() -> bool {
        THREAD_ACTIVE_COUNT.load(Ordering::SeqCst) == 0
    }
}

pub mod attention {
    use alloc::vec::Vec;

    /// CPU-bound Tiled FlashAttention (FP32 Base)
    /// To prevent L1 Cache misses, Q, K, V are tiled into small cache-aligned blocks.
    pub struct CpuTiledAttention;

    impl CpuTiledAttention {
        /// Executes a CPU-optimized Tiled Attention loop in FP32.
        /// This bypasses integer approximations for Softmax.
        pub fn compute_attention_f32(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, head_dim: usize, tile_size: usize) -> Vec<f32> {
            let mut output = alloc::vec![0.0f32; seq_len * head_dim];
            let mut m = alloc::vec![f32::NEG_INFINITY; seq_len];
            let mut l = alloc::vec![0.0f32; seq_len];
            let scale = 1.0 / libm::sqrtf(head_dim as f32);

            let num_tiles = seq_len.div_ceil(tile_size);
            
            let mut s_ij = alloc::vec![0.0f32; tile_size * tile_size];
            let mut p_ij = alloc::vec![0.0f32; tile_size];

            for t_q in 0..num_tiles {
                let q_start = t_q * tile_size;
                let q_end = core::cmp::min(q_start + tile_size, seq_len);
                let q_len = q_end - q_start;

                for t_k in 0..num_tiles {
                    let k_start = t_k * tile_size;
                    let k_end = core::cmp::min(k_start + tile_size, seq_len);
                    let k_len = k_end - k_start;

                    // 1. Q * K^T (Local Tile)
                    for i in 0..q_len {
                        let global_i = q_start + i;
                        let q_row = &q[global_i * head_dim .. (global_i + 1) * head_dim];
                        for j in 0..k_len {
                            let global_j = k_start + j;
                            // Causal mask: query cannot attend to future keys
                            if global_i < global_j {
                                s_ij[i * k_len + j] = f32::NEG_INFINITY;
                                continue;
                            }
                            let k_row = &k[global_j * head_dim .. (global_j + 1) * head_dim];
                            let mut dot = 0.0;
                            for d in 0..head_dim {
                                dot += q_row[d] * k_row[d];
                            }
                            s_ij[i * k_len + j] = dot * scale;
                        }
                    }

                    // 2. Local Softmax & O update
                    for i in 0..q_len {
                        let global_i = q_start + i;
                        
                        let mut m_ij = f32::NEG_INFINITY;
                        for j in 0..k_len {
                            let val = s_ij[i * k_len + j];
                            if val > m_ij {
                                m_ij = val;
                            }
                        }

                        if m_ij == f32::NEG_INFINITY {
                            continue;
                        }

                        let m_i_old = m[global_i];
                        let m_i_new = if m_i_old > m_ij { m_i_old } else { m_ij };
                        m[global_i] = m_i_new;

                        let exp_diff = libm::expf(m_i_old - m_i_new);
                        let mut l_i_new = l[global_i] * exp_diff;

                        for j in 0..k_len {
                            let p = libm::expf(s_ij[i * k_len + j] - m_i_new);
                            p_ij[j] = p;
                            l_i_new += p;
                        }
                        l[global_i] = l_i_new;

                        // 3. P * V (Local Accumulation)
                        let out_row = &mut output[global_i * head_dim .. (global_i + 1) * head_dim];
                        for d in 0..head_dim {
                            out_row[d] *= exp_diff;
                            let mut pv = 0.0;
                            for j in 0..k_len {
                                if p_ij[j] > 0.0 {
                                    pv += p_ij[j] * v[(k_start + j) * head_dim + d];
                                }
                            }
                            out_row[d] += pv;
                        }
                    }
                }
            }

            // Final normalization
            for i in 0..seq_len {
                let l_inv = if l[i] > 0.0 { 1.0 / l[i] } else { 0.0 };
                let out_row = &mut output[i * head_dim .. (i + 1) * head_dim];
                for d in 0..head_dim {
                    out_row[d] *= l_inv;
                }
            }
            
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
