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
