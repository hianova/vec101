use alloc::vec::Vec;

/// Root Mean Square Normalization (RMSNorm).
/// Operates on a flat array of `batch_size * hidden_dim`.
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    let hidden_dim = weight.len();
    for chunk in x.chunks_mut(hidden_dim) {
        let mut sum_sq = 0.0;
        for &v in chunk.iter() {
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / libm::sqrtf(mean_sq + eps);

        for i in 0..hidden_dim {
            chunk[i] = chunk[i] * inv_rms * weight[i];
        }
    }
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + libm::expf(-x));
    x * sigmoid
}

/// Applies SwiGLU activation over a slice.
pub fn swiglu(x: &mut [f32], v: &[f32]) {
    for i in 0..x.len() {
        x[i] = silu(x[i]) * v[i];
    }
}

/// Rotary Position Embedding (RoPE).
/// Batched: `q` and `k` contain `K` tokens. `start_pos` is the position of the first token.
pub fn rope(q: &mut [f32], k: &mut [f32], start_pos: usize, hidden_dim: usize, head_dim: usize, base: f32) {
    let num_tokens = q.len() / hidden_dim;
    let half = head_dim / 2;

    for t in 0..num_tokens {
        let pos = start_pos + t;
        let q_token = &mut q[t * hidden_dim..(t + 1) * hidden_dim];
        let k_token = &mut k[t * hidden_dim..(t + 1) * hidden_dim];

        for i in 0..half {
            let freq = 1.0 / libm::powf(base, (2 * i) as f32 / head_dim as f32);
            let val = (pos as f32) * freq;
            let fcos = libm::cosf(val);
            let fsin = libm::sinf(val);

            for head in 0..(hidden_dim / head_dim) {
                let offset = head * head_dim;
                
                let q0 = q_token[offset + i];
                let q1 = q_token[offset + i + half];
                q_token[offset + i] = q0 * fcos - q1 * fsin;
                q_token[offset + i + half] = q0 * fsin + q1 * fcos;

                let k0 = k_token[offset + i];
                let k1 = k_token[offset + i + half];
                k_token[offset + i] = k0 * fcos - k1 * fsin;
                k_token[offset + i + half] = k0 * fsin + k1 * fcos;
            }
        }
    }
}

/// Standard Scaled Dot-Product Attention (Batched for K tokens).
/// `q` shape: [K, hidden_dim]
/// `out` shape: [K, hidden_dim]
pub fn attention(
    q: &[f32],
    k_cache: &[Vec<f32>],
    v_cache: &[Vec<f32>],
    cache_len: usize,
    num_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let hidden_dim = num_heads * head_dim;
    let num_tokens = q.len() / hidden_dim;
    let scale = 1.0 / libm::sqrtf(head_dim as f32);

    for t in 0..num_tokens {
        let q_token = &q[t * hidden_dim..(t + 1) * hidden_dim];
        let out_token = &mut out[t * hidden_dim..(t + 1) * hidden_dim];
        let seq_len = cache_len + t + 1;

        for h in 0..num_heads {
            let q_head = &q_token[h * head_dim..(h + 1) * head_dim];
            
            let mut scores = alloc::vec![0.0; seq_len];
            for seq_idx in 0..seq_len {
                let k_head = &k_cache[seq_idx][h * head_dim..(h + 1) * head_dim];
                let mut dot = 0.0;
                for i in 0..head_dim {
                    dot += q_head[i] * k_head[i];
                }
                scores[seq_idx] = dot * scale;
            }

            let mut max_score = f32::NEG_INFINITY;
            for &s in &scores {
                if s > max_score { max_score = s; }
            }

            let mut sum_exp = 0.0;
            for s in &mut scores {
                *s = libm::expf(*s - max_score);
                sum_exp += *s;
            }

            for s in &mut scores {
                *s /= sum_exp;
            }

            let out_head = &mut out_token[h * head_dim..(h + 1) * head_dim];
            out_head.fill(0.0);
            
            for seq_idx in 0..seq_len {
                let v_head = &v_cache[seq_idx][h * head_dim..(h + 1) * head_dim];
                let score = scores[seq_idx];
                for i in 0..head_dim {
                    out_head[i] += score * v_head[i];
                }
            }
        }
    }
}

/// Fused INT8 RMSNorm.
/// Computes RMSNorm directly from `i8` array to `i8` array using fixed-point math, bypassing `f32` allocations.
/// Requires pre-quantized `weight_i8` and its `weight_scale`.
/// Returns the new dynamic scale factor `s'` per token.
pub fn rmsnorm_int8(q: &[i8], weight_i8: &[i8], weight_scale: f32, eps: f32, out_q: &mut [i8], out_scales: &mut [f32]) {
    let hidden_dim = weight_i8.len();
    for (t, (in_chunk, out_chunk)) in q.chunks(hidden_dim).zip(out_q.chunks_mut(hidden_dim)).enumerate() {
        // Pass 1: Compute sum of squares in pure integer
        let mut sum_sq = 0i32;
        for &v in in_chunk {
            sum_sq += (v as i32) * (v as i32);
        }
        
        // Float scalar ops (O(1) per token, entirely negligible)
        let mean_sq = (sum_sq as f32) / (hidden_dim as f32);
        let inv_rms = 1.0 / libm::sqrtf(mean_sq + eps);
        let combined_scale = inv_rms * weight_scale;

        // Pass 2: Find max_abs of (x_i8 * weight_i8)
        let mut max_abs_int = 0i32;
        for i in 0..hidden_dim {
            let prod = (in_chunk[i] as i32) * (weight_i8[i] as i32);
            let abs = prod.abs();
            if abs > max_abs_int { max_abs_int = abs; }
        }

        let max_out_float = (max_abs_int as f32) * combined_scale;
        let out_scale = if max_out_float == 0.0 { 1.0 } else { max_out_float / 127.0 };
        
        let multiplier_f32 = if out_scale == 0.0 { 0.0 } else { combined_scale / out_scale };
        let q_shift = 15;
        let mult_int = libm::roundf(multiplier_f32 * ((1 << q_shift) as f32)) as i32;

        // Pass 3: Scale to output using fixed-point integer math
        for i in 0..hidden_dim {
            let prod = (in_chunk[i] as i32) * (weight_i8[i] as i32);
            let mut quantized = ((prod as i64 * mult_int as i64) >> q_shift) as i32;
            if quantized > 127 { quantized = 127; }
            if quantized < -128 { quantized = -128; }
            out_chunk[i] = quantized as i8;
        }

        out_scales[t] = out_scale;
    }
}

/// Fused INT8 SwiGLU.
/// Computes SiLU and multiplies by V directly into an INT8 output buffer using a Dynamic LUT.
pub fn swiglu_int8(q: &[i8], in_scales: &[f32], v_weight_i8: &[i8], v_weight_scale: f32, out_q: &mut [i8], out_scales: &mut [f32]) {
    let hidden_dim = v_weight_i8.len();
    for (t, (in_chunk, out_chunk)) in q.chunks(hidden_dim).zip(out_q.chunks_mut(hidden_dim)).enumerate() {
        let s = in_scales[t];
        
        // 1. Build Dynamic LUT for SiLU
        let mut lut_f32 = [0.0f32; 256];
        let mut max_lut_abs = 0.0f32;
        for x_int in -128..=127i32 {
            let x_float = (x_int as f32) * s;
            let val = silu(x_float);
            lut_f32[(x_int + 128) as usize] = val;
            let abs = libm::fabsf(val);
            if abs > max_lut_abs { max_lut_abs = abs; }
        }
        
        let lut_scale = if max_lut_abs == 0.0 { 1.0 } else { max_lut_abs / 127.0 };
        let inv_lut_scale = if lut_scale == 0.0 { 0.0 } else { 1.0 / lut_scale };
        
        let mut lut_i8 = [0i8; 256];
        for i in 0..256 {
            let mut q_val = libm::roundf(lut_f32[i] * inv_lut_scale) as i32;
            if q_val > 127 { q_val = 127; }
            if q_val < -128 { q_val = -128; }
            lut_i8[i] = q_val as i8;
        }
        
        let combined_scale = lut_scale * v_weight_scale;
        
        // 2. Find max absolute product in the chunk
        let mut max_abs_int = 0i32;
        for i in 0..hidden_dim {
            let x_val = in_chunk[i];
            let l_val = lut_i8[(x_val as i32 + 128) as usize] as i32;
            let v_val = v_weight_i8[i] as i32;
            let prod = l_val * v_val;
            let abs = prod.abs();
            if abs > max_abs_int { max_abs_int = abs; }
        }
        
        let max_out_float = (max_abs_int as f32) * combined_scale;
        let out_scale = if max_out_float == 0.0 { 1.0 } else { max_out_float / 127.0 };
        
        let multiplier_f32 = if out_scale == 0.0 { 0.0 } else { combined_scale / out_scale };
        let q_shift = 15;
        let mult_int = libm::roundf(multiplier_f32 * ((1 << q_shift) as f32)) as i32;
        
        // 3. Process the chunk entirely in integer
        for i in 0..hidden_dim {
            let x_val = in_chunk[i];
            let l_val = lut_i8[(x_val as i32 + 128) as usize] as i32;
            let v_val = v_weight_i8[i] as i32;
            let prod = l_val * v_val;
            
            let mut quantized = ((prod as i64 * mult_int as i64) >> q_shift) as i32;
            if quantized > 127 { quantized = 127; }
            if quantized < -128 { quantized = -128; }
            out_chunk[i] = quantized as i8;
        }
        
        out_scales[t] = out_scale;
    }
}
