#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Calculates INT8 cosine similarity between two vectors.
///
/// Hardware Acceleration:
/// - AARCH64: Uses NEON intrinsics (vmull_s8, vaddq_s32, vpaddlq_s16)
/// - Other: Uses scalar fallback
pub fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut dot = 0i32;
    let mut norm_a = 0i32;
    let mut norm_b = 0i32;

    let len = a.len().min(b.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut sum_dot = vdupq_n_s32(0);
        let mut sum_norm_a = vdupq_n_s32(0);
        let mut sum_norm_b = vdupq_n_s32(0);

        let chunks = len / 16;
        for c in 0..chunks {
            let va = vld1q_s8(a.as_ptr().add(c * 16));
            let vb = vld1q_s8(b.as_ptr().add(c * 16));

            let va_l = vget_low_s8(va);
            let va_h = vget_high_s8(va);
            let vb_l = vget_low_s8(vb);
            let vb_h = vget_high_s8(vb);

            // dot
            let p_dot_l = vmull_s8(va_l, vb_l);
            let p_dot_h = vmull_s8(va_h, vb_h);
            sum_dot = vaddq_s32(
                sum_dot,
                vaddq_s32(vpaddlq_s16(p_dot_l), vpaddlq_s16(p_dot_h)),
            );

            // norm_a
            let p_na_l = vmull_s8(va_l, va_l);
            let p_na_h = vmull_s8(va_h, va_h);
            sum_norm_a = vaddq_s32(
                sum_norm_a,
                vaddq_s32(vpaddlq_s16(p_na_l), vpaddlq_s16(p_na_h)),
            );

            // norm_b
            let p_nb_l = vmull_s8(vb_l, vb_l);
            let p_nb_h = vmull_s8(vb_h, vb_h);
            sum_norm_b = vaddq_s32(
                sum_norm_b,
                vaddq_s32(vpaddlq_s16(p_nb_l), vpaddlq_s16(p_nb_h)),
            );
        }

        dot += vaddvq_s32(sum_dot);
        norm_a += vaddvq_s32(sum_norm_a);
        norm_b += vaddvq_s32(sum_norm_b);

        for i in (chunks * 16)..len {
            let ai = a[i] as i32;
            let bi = b[i] as i32;
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..len {
        let ai = a[i] as i32;
        let bi = b[i] as i32;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if norm_a == 0 || norm_b == 0 {
        return 0;
    }
    let sign = if dot < 0 { -1 } else { 1 };
    let score = (dot as i64 * dot as i64 * sign) / (norm_a as i64 * norm_b as i64).max(1);
    score as i32
}
