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
        #[cfg(not(miri))]
        let inv_rms = 1.0 / libm::sqrtf(mean_sq + eps);
        #[cfg(miri)]
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

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

#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    // GELU approximation
    let c1 = 0.044715;
    let c2 = 0.7978845608; // sqrt(2/pi)
    let inner = c2 * (x + c1 * x * x * x);
    0.5 * x * (1.0 + libm::tanhf(inner))
}

/// Applies GeGLU activation over a slice (for Gemma models).
pub fn geglu(x: &mut [f32], v: &[f32]) {
    for i in 0..x.len() {
        x[i] = gelu(x[i]) * v[i];
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
    #[cfg(not(miri))]
    let scale = 1.0 / libm::sqrtf(head_dim as f32);
    #[cfg(miri)]
    let scale = 1.0 / (head_dim as f32).sqrt();

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
        // Pass 1: Compute sum of squares
        let mut sum_sq = 0i32;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let mut sum_sq_acc = vdupq_n_s32(0);
            let mut chunks = in_chunk.chunks_exact(16);
            for chunk in chunks.by_ref() {
                let x = vld1q_s8(chunk.as_ptr());
                core::arch::asm!(
                    "sdot {acc:v}.4s, {x:v}.16b, {x:v}.16b",
                    acc = inout(vreg) sum_sq_acc,
                    x = in(vreg) x,
                );
            }
            sum_sq += vaddvq_s32(sum_sq_acc);
            for &v in chunks.remainder() {
                sum_sq += (v as i32) * (v as i32);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for &v in in_chunk {
                sum_sq += (v as i32) * (v as i32);
            }
        }
        
        // Float scalar ops (O(1) per token, entirely negligible)
        let mean_sq = (sum_sq as f32) / (hidden_dim as f32);
        #[cfg(not(miri))]
        let inv_rms = 1.0 / libm::sqrtf(mean_sq + eps);
        #[cfg(miri)]
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let combined_scale = inv_rms * weight_scale;

        // Pass 2: Find max_abs of (x_i8 * weight_i8)
        let mut max_abs_int = 0i32;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let mut max_vec = vdupq_n_s16(0);
            let mut x_chunks = in_chunk.chunks_exact(16);
            let mut w_chunks = weight_i8.chunks_exact(16);
            for (x_ch, w_ch) in x_chunks.by_ref().zip(w_chunks.by_ref()) {
                let x = vld1q_s8(x_ch.as_ptr());
                let w = vld1q_s8(w_ch.as_ptr());
                let p_lo = vmull_s8(vget_low_s8(x), vget_low_s8(w));
                let p_hi = vmull_s8(vget_high_s8(x), vget_high_s8(w));
                max_vec = vmaxq_s16(max_vec, vmaxq_s16(vabsq_s16(p_lo), vabsq_s16(p_hi)));
            }
            max_abs_int = vmaxvq_s16(max_vec) as i32;
            for (&x, &w) in x_chunks.remainder().iter().zip(w_chunks.remainder()) {
                let abs = ((x as i32) * (w as i32)).abs();
                if abs > max_abs_int { max_abs_int = abs; }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..hidden_dim {
                let prod = (in_chunk[i] as i32) * (weight_i8[i] as i32);
                let abs = prod.abs();
                if abs > max_abs_int { max_abs_int = abs; }
            }
        }

        let max_out_float = (max_abs_int as f32) * combined_scale;
        let out_scale = if max_out_float == 0.0 { 1.0 } else { max_out_float / 127.0 };
        
        let multiplier_f32 = if out_scale == 0.0 { 0.0 } else { combined_scale / out_scale };
        let q_shift = 15;
        let mult_int = libm::roundf(multiplier_f32 * ((1 << q_shift) as f32)) as i32;

        // Pass 3: Scale to output using fixed-point integer math
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let mut x_chunks = in_chunk.chunks_exact(16);
            let mut w_chunks = weight_i8.chunks_exact(16);
            let mut out_chunks = out_chunk.chunks_exact_mut(16);
            for ((x_ch, w_ch), out_ch) in x_chunks.by_ref().zip(w_chunks.by_ref()).zip(out_chunks.by_ref()) {
                let x = vld1q_s8(x_ch.as_ptr());
                let w = vld1q_s8(w_ch.as_ptr());
                
                let p_lo = vmull_s8(vget_low_s8(x), vget_low_s8(w));
                let p_hi = vmull_s8(vget_high_s8(x), vget_high_s8(w));
                
                let p_ll = vmovl_s16(vget_low_s16(p_lo));
                let p_lh = vmovl_s16(vget_high_s16(p_lo));
                let p_hl = vmovl_s16(vget_low_s16(p_hi));
                let p_hh = vmovl_s16(vget_high_s16(p_hi));
                
                let r_ll = vmulq_n_s32(p_ll, mult_int);
                let r_lh = vmulq_n_s32(p_lh, mult_int);
                let r_hl = vmulq_n_s32(p_hl, mult_int);
                let r_hh = vmulq_n_s32(p_hh, mult_int);
                
                let s_ll = vshrq_n_s32::<15>(r_ll);
                let s_lh = vshrq_n_s32::<15>(r_lh);
                let s_hl = vshrq_n_s32::<15>(r_hl);
                let s_hh = vshrq_n_s32::<15>(r_hh);
                
                let n_l = vcombine_s16(vqmovn_s32(s_ll), vqmovn_s32(s_lh));
                let n_h = vcombine_s16(vqmovn_s32(s_hl), vqmovn_s32(s_hh));
                
                let final_q = vcombine_s8(vqmovn_s16(n_l), vqmovn_s16(n_h));
                vst1q_s8(out_ch.as_mut_ptr(), final_q);
            }
            
            // remainder (if hidden_dim % 16 != 0)
            for ((&x, &w), out) in x_chunks.remainder().iter().zip(w_chunks.remainder()).zip(out_chunks.into_remainder()) {
                let prod = (x as i32) * (w as i32);
                let mut quantized = ((prod as i64 * mult_int as i64) >> q_shift) as i32;
                quantized = quantized.clamp(-128, 127);
                *out = quantized as i8;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..hidden_dim {
                let prod = (in_chunk[i] as i32) * (weight_i8[i] as i32);
                let mut quantized = ((prod as i64 * mult_int as i64) >> q_shift) as i32;
                if quantized > 127 { quantized = 127; }
                if quantized < -128 { quantized = -128; }
                out_chunk[i] = quantized as i8;
            }
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
            q_val = q_val.clamp(-128, 127);
            lut_i8[i] = q_val as i8;
        }
        
        let combined_scale = lut_scale * v_weight_scale;
        
        // 2. Find max absolute product in the chunk
        let mut max_abs_int = 0i32;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let t0 = vld1q_u8_x4(lut_i8[0..64].as_ptr() as *const u8);
            let t1 = vld1q_u8_x4(lut_i8[64..128].as_ptr() as *const u8);
            let t2 = vld1q_u8_x4(lut_i8[128..192].as_ptr() as *const u8);
            let t3 = vld1q_u8_x4(lut_i8[192..256].as_ptr() as *const u8);
            
            let mut max_vec = vdupq_n_s16(0);
            let mut x_chunks = in_chunk.chunks_exact(16);
            let mut v_chunks = v_weight_i8.chunks_exact(16);
            for (x_ch, v_ch) in x_chunks.by_ref().zip(v_chunks.by_ref()) {
                let x = vld1q_s8(x_ch.as_ptr());
                let x_u8 = vaddq_u8(vreinterpretq_u8_s8(x), vdupq_n_u8(128));
                let r0 = vqtbl4q_u8(t0, x_u8);
                let r1 = vqtbl4q_u8(t1, vsubq_u8(x_u8, vdupq_n_u8(64)));
                let r2 = vqtbl4q_u8(t2, vsubq_u8(x_u8, vdupq_n_u8(128)));
                let r3 = vqtbl4q_u8(t3, vsubq_u8(x_u8, vdupq_n_u8(192)));
                
                let res = vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3));
                let l_val = vreinterpretq_s8_u8(res);
                let v = vld1q_s8(v_ch.as_ptr());
                
                let p_lo = vmull_s8(vget_low_s8(l_val), vget_low_s8(v));
                let p_hi = vmull_s8(vget_high_s8(l_val), vget_high_s8(v));
                max_vec = vmaxq_s16(max_vec, vmaxq_s16(vabsq_s16(p_lo), vabsq_s16(p_hi)));
            }
            max_abs_int = vmaxvq_s16(max_vec) as i32;
            for (&x, &v) in x_chunks.remainder().iter().zip(v_chunks.remainder()) {
                let l_val = lut_i8[(x as i32 + 128) as usize] as i32;
                let prod = l_val * (v as i32);
                let abs = prod.abs();
                if abs > max_abs_int { max_abs_int = abs; }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..hidden_dim {
                let x_val = in_chunk[i];
                let l_val = lut_i8[(x_val as i32 + 128) as usize] as i32;
                let v_val = v_weight_i8[i] as i32;
                let prod = l_val * v_val;
                let abs = prod.abs();
                if abs > max_abs_int { max_abs_int = abs; }
            }
        }
        
        let max_out_float = (max_abs_int as f32) * combined_scale;
        let out_scale = if max_out_float == 0.0 { 1.0 } else { max_out_float / 127.0 };
        
        let multiplier_f32 = if out_scale == 0.0 { 0.0 } else { combined_scale / out_scale };
        let q_shift = 15;
        let mult_int = libm::roundf(multiplier_f32 * ((1 << q_shift) as f32)) as i32;
        
        // 3. Process the chunk entirely in integer
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let t0 = vld1q_u8_x4(lut_i8[0..64].as_ptr() as *const u8);
            let t1 = vld1q_u8_x4(lut_i8[64..128].as_ptr() as *const u8);
            let t2 = vld1q_u8_x4(lut_i8[128..192].as_ptr() as *const u8);
            let t3 = vld1q_u8_x4(lut_i8[192..256].as_ptr() as *const u8);

            let mut x_chunks = in_chunk.chunks_exact(16);
            let mut v_chunks = v_weight_i8.chunks_exact(16);
            let mut out_chunks = out_chunk.chunks_exact_mut(16);
            for ((x_ch, v_ch), out_ch) in x_chunks.by_ref().zip(v_chunks.by_ref()).zip(out_chunks.by_ref()) {
                let x = vld1q_s8(x_ch.as_ptr());
                let x_u8 = vaddq_u8(vreinterpretq_u8_s8(x), vdupq_n_u8(128));
                let r0 = vqtbl4q_u8(t0, x_u8);
                let r1 = vqtbl4q_u8(t1, vsubq_u8(x_u8, vdupq_n_u8(64)));
                let r2 = vqtbl4q_u8(t2, vsubq_u8(x_u8, vdupq_n_u8(128)));
                let r3 = vqtbl4q_u8(t3, vsubq_u8(x_u8, vdupq_n_u8(192)));
                
                let res = vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3));
                let l_val = vreinterpretq_s8_u8(res);
                let v = vld1q_s8(v_ch.as_ptr());
                
                let p_lo = vmull_s8(vget_low_s8(l_val), vget_low_s8(v));
                let p_hi = vmull_s8(vget_high_s8(l_val), vget_high_s8(v));
                
                let p_ll = vmovl_s16(vget_low_s16(p_lo));
                let p_lh = vmovl_s16(vget_high_s16(p_lo));
                let p_hl = vmovl_s16(vget_low_s16(p_hi));
                let p_hh = vmovl_s16(vget_high_s16(p_hi));
                
                let r_ll = vmulq_n_s32(p_ll, mult_int);
                let r_lh = vmulq_n_s32(p_lh, mult_int);
                let r_hl = vmulq_n_s32(p_hl, mult_int);
                let r_hh = vmulq_n_s32(p_hh, mult_int);
                
                let s_ll = vshrq_n_s32::<15>(r_ll);
                let s_lh = vshrq_n_s32::<15>(r_lh);
                let s_hl = vshrq_n_s32::<15>(r_hl);
                let s_hh = vshrq_n_s32::<15>(r_hh);
                
                let n_l = vcombine_s16(vqmovn_s32(s_ll), vqmovn_s32(s_lh));
                let n_h = vcombine_s16(vqmovn_s32(s_hl), vqmovn_s32(s_hh));
                
                let final_q = vcombine_s8(vqmovn_s16(n_l), vqmovn_s16(n_h));
                vst1q_s8(out_ch.as_mut_ptr(), final_q);
            }
            
            for ((&x, &v), out) in x_chunks.remainder().iter().zip(v_chunks.remainder()).zip(out_chunks.into_remainder()) {
                let l_val = lut_i8[(x as i32 + 128) as usize] as i32;
                let prod = l_val * (v as i32);
                let mut quantized = ((prod as i64 * mult_int as i64) >> q_shift) as i32;
                quantized = quantized.clamp(-128, 127);
                *out = quantized as i8;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
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
        }
        
        out_scales[t] = out_scale;
    }
}

use crate::types::vec101_block;

/// Compresses an INT8 activation stream into 1.58-bit ternary masks for GPU bitwise operations.
pub fn quantize_to_ternary(x_i8: &[i8], blocks: &mut [vec101_block]) -> f32 {
    let mut sum_abs = 0i64;
    for &v in x_i8 {
        sum_abs += (v as i64).abs();
    }
    let scale = if x_i8.is_empty() { 0.0 } else { (sum_abs as f32) / (x_i8.len() as f32) };

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
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        rmsnorm(&mut x, &weight, eps);
        // mean_sq = 7.5, rms = 2.7386
        assert!((x[0] - 0.365148).abs() < 1e-4);
        assert!((x[3] - 1.46059).abs() < 1e-4);
    }

    #[test]
    fn test_rmsnorm_int8() {
        let q = vec![10i8, 20i8, 30i8, 40i8];
        let weight_i8 = vec![127i8; 4];
        let mut out_q = vec![0i8; 4];
        let mut out_scales = vec![0.0f32; 1];
        rmsnorm_int8(&q, &weight_i8, 1.0/127.0, 1e-5, &mut out_q, &mut out_scales);
        assert!(out_scales[0] > 0.0);
        // The output should be quantized. Element 3 is the largest, should map near 127.
        assert!(out_q[3] >= 126);
    }

    #[test]
    fn test_swiglu_int8() {
        let q = vec![50i8, -50i8, 0i8, 100i8];
        let in_scales = vec![0.1f32];
        let v_weight_i8 = vec![127i8; 4];
        let mut out_q = vec![0i8; 4];
        let mut out_scales = vec![0.0f32; 1];
        swiglu_int8(&q, &in_scales, &v_weight_i8, 1.0/127.0, &mut out_q, &mut out_scales);
        assert!(out_scales[0] > 0.0);
    }

    #[test]
    fn test_quantize_to_ternary() {
        let mut x = vec![0i8; 256];
        x[0] = 50;
        x[1] = -50;
        x[2] = 0;
        let mut blocks = vec![crate::types::vec101_block { w_pos_bits: [0;4], w_neg_bits: [0;4] }];
        quantize_to_ternary(&x, &mut blocks);
        assert_eq!(blocks[0].w_pos_bits[0] & 1, 1);
        assert_eq!(blocks[0].w_neg_bits[0] & 2, 2);
        assert_eq!(blocks[0].w_pos_bits[0] & 4, 0);
    }

    #[test]
    fn test_rope() {
        let mut q = vec![1.0, 0.0, 1.0, 0.0];
        let mut k = vec![1.0, 0.0, 1.0, 0.0];
        rope(&mut q, &mut k, 0, 4, 2, 10000.0);
        assert_eq!(q[0], 1.0); // At pos 0, rot is 0
    }
}
