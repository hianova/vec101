#![allow(unsafe_op_in_unsafe_fn)]
use core::arch::aarch64::*;
pub unsafe fn test() {
    let t0 = uint8x16x4_t(vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0));
    let x = vdupq_n_u8(0);
    let _ = vqtbl4q_u8(t0, x);
}
fn main() {}
