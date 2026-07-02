use alloc::vec::Vec;
use crate::util::math_int::*;

pub fn rmsnorm(x: &mut [i32], weight: &[i32], eps: i32) {
    let hidden_dim = weight.len();
    for chunk in x.chunks_mut(hidden_dim) {
        let mut sum_sq = 0i64;
        for &v in chunk.iter() {
            sum_sq += (v as i64 * v as i64) >> 16;
        }
        let mean_sq = (sum_sq / hidden_dim as i64) as i32;
        let inv_rms = rsqrt_approx_i32((mean_sq + eps) as u32);

        for i in 0..hidden_dim {
            let val = (chunk[i] as i64 * inv_rms as i64) >> 16;
            chunk[i] = ((val * weight[i] as i64) >> 16) as i32;
        }
    }
}

pub fn silu(x: i32) -> i32 {
    let exp_neg_x = exp_approx_q16(-x);
    let sigmoid = ((1i64 << 16) << 16) / ((1 << 16) + exp_neg_x) as i64;
    ((x as i64 * sigmoid) >> 16) as i32
}

pub fn swiglu(x: &mut [i32], v: &[i32]) {
    for i in 0..x.len() {
        x[i] = ((silu(x[i]) as i64 * v[i] as i64) >> 16) as i32;
    }
}

pub fn gelu(x: i32) -> i32 {
    let c1 = 2930;
    let c2 = 52290;
    let x_sq = (x as i64 * x as i64) >> 16;
    let x_cube = (x_sq * x as i64) >> 16;
    let inner = (c2 as i64 * (x as i64 + ((c1 as i64 * x_cube) >> 16))) >> 16;
    
    let exp2 = exp_approx_q16((inner * 2) as i32);
    let tanh = ((exp2 - (1 << 16)) as i64 * (1 << 16)) / (exp2 + (1 << 16)) as i64;
    
    (((x as i64 * ((1 << 16) + tanh)) >> 1) >> 16) as i32
}

pub fn geglu(x: &mut [i32], v: &[i32]) {
    for i in 0..x.len() {
        x[i] = ((gelu(x[i]) as i64 * v[i] as i64) >> 16) as i32;
    }
}

pub fn rope(q: &mut [i32], k: &mut [i32], _start_pos: usize, _hidden_dim: usize, _head_dim: usize, _base: i32) {
    // Dummy rope to avoid complex integer sin/cos implementation for now
    for i in 0..q.len() {
        q[i] = q[i];
        k[i] = k[i];
    }
}

pub fn attention(
    q: &[i32],
    k_cache: &[Vec<i32>],
    v_cache: &[Vec<i32>],
    cache_len: usize,
    num_heads: usize,
    head_dim: usize,
    out: &mut [i32],
) {
    let hidden_dim = num_heads * head_dim;
    let num_tokens = q.len() / hidden_dim;
    let scale = rsqrt_approx_i32(head_dim as u32);

    for t in 0..num_tokens {
        let q_token = &q[t * hidden_dim..(t + 1) * hidden_dim];
        let out_token = &mut out[t * hidden_dim..(t + 1) * hidden_dim];
        let seq_len = cache_len + t + 1;

        for h in 0..num_heads {
            let q_head = &q_token[h * head_dim..(h + 1) * head_dim];
            
            let mut scores = alloc::vec![0i32; seq_len];
            for seq_idx in 0..seq_len {
                let k_head = &k_cache[seq_idx][h * head_dim..(h + 1) * head_dim];
                let mut dot = 0i64;
                for i in 0..head_dim {
                    dot += (q_head[i] as i64 * k_head[i] as i64) >> 16;
                }
                scores[seq_idx] = ((dot * scale as i64) >> 16) as i32;
            }

            let mut max_score = i32::MIN;
            for &s in &scores {
                if s > max_score { max_score = s; }
            }

            let mut sum_exp = 0i64;
            for s in &mut scores {
                *s = exp_approx_q16(*s - max_score);
                sum_exp += *s as i64;
            }

            for s in &mut scores {
                *s = ((*s as i64 * (1 << 16)) / sum_exp) as i32;
            }

            let out_head = &mut out_token[h * head_dim..(h + 1) * head_dim];
            out_head.fill(0);
            
            for seq_idx in 0..seq_len {
                let v_head = &v_cache[seq_idx][h * head_dim..(h + 1) * head_dim];
                let score = scores[seq_idx];
                for i in 0..head_dim {
                    out_head[i] += ((score as i64 * v_head[i] as i64) >> 16) as i32;
                }
            }
        }
    }
}

pub fn rmsnorm_int8(q: &[i8], weight_i8: &[i8], weight_scale: i32, eps: i32, out_q: &mut [i8], out_scales: &mut [i32]) {
    let hidden_dim = weight_i8.len();
    for (t, (in_chunk, out_chunk)) in q.chunks(hidden_dim).zip(out_q.chunks_mut(hidden_dim)).enumerate() {
        let mut sum_sq = 0i32;
        for &v in in_chunk {
            sum_sq += (v as i32) * (v as i32);
        }
        
        let mean_sq = sum_sq / hidden_dim as i32;
        let inv_rms = rsqrt_approx_i32((mean_sq + eps) as u32);
        let combined_scale = ((inv_rms as i64 * weight_scale as i64) >> 16) as i32;

        let mut max_abs_int = 0i32;
        for i in 0..hidden_dim {
            let prod = (in_chunk[i] as i32) * (weight_i8[i] as i32);
            let abs = prod.abs();
            if abs > max_abs_int { max_abs_int = abs; }
        }

        let max_out = (max_abs_int as i64 * combined_scale as i64) >> 16;
        let out_scale = if max_out == 0 { 1 } else { (max_out / 127) as i32 };
        
        let multiplier = if out_scale == 0 { 0 } else { ((combined_scale as i64) << 16) / out_scale as i64 };

        for i in 0..hidden_dim {
            let prod = (in_chunk[i] as i32) * (weight_i8[i] as i32);
            let mut quantized = ((prod as i64 * multiplier) >> 31) as i32;
            quantized = quantized.clamp(-128, 127);
            out_chunk[i] = quantized as i8;
        }

        out_scales[t] = out_scale;
    }
}

pub fn swiglu_int8(q: &[i8], in_scales: &[i32], v_weight_i8: &[i8], v_weight_scale: i32, out_q: &mut [i8], out_scales: &mut [i32]) {
    let hidden_dim = v_weight_i8.len();
    for (t, (in_chunk, out_chunk)) in q.chunks(hidden_dim).zip(out_q.chunks_mut(hidden_dim)).enumerate() {
        let s = in_scales[t];
        
        let mut max_abs_int = 0i32;
        for i in 0..hidden_dim {
            let x_val = in_chunk[i];
            let x_fixed = x_val as i32 * s;
            let l_val = silu(x_fixed) >> 8; 
            let v_val = v_weight_i8[i] as i32;
            let prod = l_val * v_val;
            let abs = prod.abs();
            if abs > max_abs_int { max_abs_int = abs; }
        }
        
        let combined_scale = v_weight_scale;
        let max_out = (max_abs_int as i64 * combined_scale as i64) >> 16;
        let out_scale = if max_out == 0 { 1 } else { (max_out / 127) as i32 };
        let multiplier = if out_scale == 0 { 0 } else { ((combined_scale as i64) << 16) / out_scale as i64 };

        for i in 0..hidden_dim {
            let x_val = in_chunk[i];
            let x_fixed = x_val as i32 * s;
            let l_val = silu(x_fixed) >> 8;
            let v_val = v_weight_i8[i] as i32;
            let prod = l_val * v_val;
            
            let mut quantized = ((prod as i64 * multiplier) >> 31) as i32;
            quantized = quantized.clamp(-128, 127);
            out_chunk[i] = quantized as i8;
        }
        
        out_scales[t] = out_scale;
    }
}

use crate::core::vec101_block;

pub fn quantize_to_ternary(x_i8: &[i8], blocks: &mut [vec101_block]) -> i32 {
    let mut sum_abs = 0i64;
    for &v in x_i8 {
        sum_abs += (v as i64).abs();
    }
    let scale = if x_i8.is_empty() { 0 } else { (sum_abs / x_i8.len() as i64) as i32 };

    for (i, chunk) in x_i8.chunks_exact(256).enumerate() {
        let mut pos = [0u64; 4];
        let mut neg = [0u64; 4];
        for k in 0..4 {
            let mut p_bits = 0u64;
            let mut n_bits = 0u64;
            let sub_chunk = &chunk[k * 64 .. (k + 1) * 64];
            for (j, &v) in sub_chunk.iter().enumerate() {
                if v > 0 {
                    p_bits |= 1 << j;
                } else if v < 0 {
                    n_bits |= 1 << j;
                }
            }
            pos[k] = p_bits;
            neg[k] = n_bits;
        }
        blocks[i] = vec101_block {
            w_pos_bits: pos,
            w_neg_bits: neg,
        };
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_rmsnorm() {
        let mut x = vec![1 << 16, 2 << 16, 3 << 16, 4 << 16];
        let w = vec![1 << 16; 4];
        rmsnorm(&mut x, &w, 1);
        assert!(x[0] != 0);
    }

    #[test]
    fn test_silu() {
        assert_eq!(silu(0), 0);
        assert!(silu(1 << 16) > 0);
        assert!(silu(-(1 << 16)) < 0);
    }

    #[test]
    fn test_swiglu() {
        let mut x = vec![1 << 16, -(1 << 16)];
        let v = vec![1 << 16, 1 << 16];
        swiglu(&mut x, &v);
        assert!(x[0] > 0);
    }

    #[test]
    fn test_gelu() {
        assert_eq!(gelu(0), 0);
        assert!(gelu(1 << 16) > 0);
        assert!(gelu(-(1 << 16)) < 0);
    }

    #[test]
    fn test_geglu() {
        let mut x = vec![1 << 16, -(1 << 16)];
        let v = vec![1 << 16, 1 << 16];
        geglu(&mut x, &v);
        assert!(x[0] > 0);
    }

    #[test]
    fn test_rope() {
        let mut q = vec![1; 4];
        let mut k = vec![2; 4];
        rope(&mut q, &mut k, 0, 4, 2, 10000);
        assert_eq!(q[0], 1);
        assert_eq!(k[0], 2);
    }

    #[test]
    fn test_attention() {
        let q = vec![1 << 16; 4];
        let k_cache = vec![vec![1 << 16; 4]];
        let v_cache = vec![vec![1 << 16; 4]];
        let mut out = vec![0; 4];
        attention(&q, &k_cache, &v_cache, 0, 2, 2, &mut out);
        assert!(out[0] > 0);
    }

    #[test]
    fn test_rmsnorm_int8() {
        let q = vec![1, 2, 3, 4];
        let w = vec![1, 1, 1, 1];
        let mut out_q = vec![0; 4];
        let mut out_scales = vec![0; 1];
        rmsnorm_int8(&q, &w, 100, 1, &mut out_q, &mut out_scales);
        assert!(out_scales[0] >= 0);
    }

    #[test]
    fn test_swiglu_int8() {
        let q = vec![1, -1];
        let in_scales = vec![100];
        let v = vec![1, 1];
        let mut out_q = vec![0; 2];
        let mut out_scales = vec![0; 1];
        swiglu_int8(&q, &in_scales, &v, 100, &mut out_q, &mut out_scales);
        assert!(out_scales[0] >= 0);
    }

    #[test]
    fn test_quantize_to_ternary() {
        let x_i8 = vec![1; 256];
        let mut blocks = vec![crate::core::vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 1];
        let scale = quantize_to_ternary(&x_i8, &mut blocks);
        assert_eq!(scale, 1);
        
        let empty_i8: Vec<i8> = vec![];
        let mut empty_blocks = vec![];
        let scale2 = quantize_to_ternary(&empty_i8, &mut empty_blocks);
        assert_eq!(scale2, 0);

        // Test negatives and zeros
        let mut mixed = vec![0; 256];
        mixed[0] = 1;
        mixed[1] = -1;
        let mut blocks2 = vec![crate::core::vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 1];
        quantize_to_ternary(&mixed, &mut blocks2);
        assert_eq!(blocks2[0].w_pos_bits[0] & 1, 1);
        assert_eq!(blocks2[0].w_neg_bits[0] & 2, 2);
    }
}
