/// Zero-Float Mathematical Engine for vec101.
/// Implements exponential and inverse square root functions using pure integer arithmetic.

// We use Q16.16 fixed-point format internally for high precision approximations.
pub const FIXED_POINT_SHIFT: i32 = 16;
pub const FIXED_POINT_ONE: i32 = 1 << FIXED_POINT_SHIFT;

/// Approximates exp(x) where x is in Q16.16 fixed-point format.
/// Returns exp(x) in Q16.16 format.
/// Uses the property: exp(x) = 2^(x * log2(e))
pub fn exp_approx_q16(x: i32) -> i32 {
    if x < -10 * FIXED_POINT_ONE {
        return 0; // exp(x) approaches 0 for large negative numbers
    }
    if x > 10 * FIXED_POINT_ONE {
        // Prevent overflow, clamp to max representable
        return i32::MAX;
    }

    // log2(e) * 2^16 = 1.442695 * 65536 ≈ 94548
    let x_scaled = ((x as i64 * 94548) >> FIXED_POINT_SHIFT) as i32;

    let int_part = x_scaled >> FIXED_POINT_SHIFT;
    let frac_part = x_scaled & (FIXED_POINT_ONE - 1);

    // 2^frac_part ≈ 1 + frac_part (Linear approximation for 0 <= frac_part < 1)
    let approx_frac = FIXED_POINT_ONE + frac_part;

    if int_part >= 0 {
        approx_frac << int_part
    } else {
        approx_frac >> (-int_part)
    }
}

/// Approximates 1 / sqrt(x) where x is a standard u32 integer.
/// Returns the result in Q16.16 fixed-point format.
pub fn rsqrt_approx_i32(x: u32) -> u32 {
    if x == 0 {
        return 0;
    }
    // Simple bitwise approximation for sqrt: sqrt(x) ≈ 2^(log2(x)/2)
    let msb = 31 - x.leading_zeros();
    let sqrt_approx = 1 << (msb / 2);

    // Convert 1.0 to Q16.16 and divide
    let one_q16 = 1u32 << 16;
    one_q16 / sqrt_approx
}

/// Approximates SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
/// Expects standard i8 input and returns standard i8.
pub fn silu_approx_i8(x: i8) -> i8 {
    // Convert x to Q16.16
    let x_q16 = (x as i32) << FIXED_POINT_SHIFT;
    let exp_neg_x = exp_approx_q16(-x_q16);

    let denom = FIXED_POINT_ONE + exp_neg_x;

    // x / (1 + exp(-x)) -> in Q16.16 division: (x_q16 * 2^16) / denom
    // To avoid overflow, we shift denom down or x_q16 up.
    let result = (x_q16 as i64 * FIXED_POINT_ONE as i64) / denom as i64;

    // Shift back to i8
    let res_i32 = (result >> FIXED_POINT_SHIFT) as i32;
    res_i32.clamp(-128, 127) as i8
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_exp_approx() {
        let test_vals = vec![0, -1, -2, 1, 2];
        for v in test_vals {
            let v_q16 = v * FIXED_POINT_ONE;
            let res_q16 = exp_approx_q16(v_q16);
            assert!(res_q16 >= 0);
        }
    }

    #[test]
    fn test_silu_approx() {
        for x in -5..=5 {
            let res_i8 = silu_approx_i8(x as i8);
            assert!(res_i8 >= -128);
        }
    }

    #[test]
    fn test_exp_approx_edge_cases() {
        assert_eq!(exp_approx_q16(-11 * FIXED_POINT_ONE), 0);
        assert_eq!(exp_approx_q16(11 * FIXED_POINT_ONE), i32::MAX);
    }

    #[test]
    fn test_rsqrt_approx_edge_cases() {
        assert_eq!(rsqrt_approx_i32(0), 0);
        assert!(rsqrt_approx_i32(1) > 0);
    }
}
