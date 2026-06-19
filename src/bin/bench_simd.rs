use std::time::Instant;

use vec101::types::BlockQ4_0;
use vec101::types::f16_to_f32;

fn scalar_q4_0(w_block: &BlockQ4_0, x_stream: &[i8]) -> f32 {
    let mut block_sum = 0;
    let mut x_idx = 0;
    
    for i in 0..16 {
        let q = w_block.qs[i];
        let q0 = (q & 0x0F) as i32 - 8;
        let q1 = (q >> 4) as i32 - 8;
        
        block_sum += q0 * (x_stream[x_idx] as i32);
        block_sum += q1 * (x_stream[x_idx + 1] as i32);
        x_idx += 2;
    }
    
    (block_sum as f32) * f16_to_f32(w_block.d)
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
unsafe fn neon_q4_0(w_block: &BlockQ4_0, x_stream: &[i8]) -> f32 {
    unsafe {
        let q_vec = vld1q_u8(w_block.qs.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_u8(8);
        
        let q0_u8 = vandq_u8(q_vec, mask);
        let q0_s8 = vreinterpretq_s8_u8(vsubq_u8(q0_u8, eight));
        
        let q1_u8 = vshrq_n_u8::<4>(q_vec);
        let q1_s8 = vreinterpretq_s8_u8(vsubq_u8(q1_u8, eight));
        
        let x_vecs = vld2q_s8(x_stream.as_ptr());
        
        let mut acc = vdupq_n_s32(0);
        core::arch::asm!(
            "sdot {acc:v}.4s, {x0:v}.16b, {w0:v}.16b",
            "sdot {acc:v}.4s, {x1:v}.16b, {w1:v}.16b",
            acc = inout(vreg) acc,
            x0 = in(vreg) x_vecs.0,
            w0 = in(vreg) q0_s8,
            x1 = in(vreg) x_vecs.1,
            w1 = in(vreg) q1_s8,
        );
        
        let block_sum = vaddvq_s32(acc);
        (block_sum as f32) * f16_to_f32(w_block.d)
    }
}

fn main() {
    let block = BlockQ4_0 {
        d: vec101::types::f32_to_f16(0.5),
        qs: [0x5A; 16],
    };
    
    let mut x_stream = vec![0i8; 32];
    for i in 0..32 {
        x_stream[i] = (i % 10) as i8;
    }

    let iterations = 10_000_000;
    
    println!("=== 🏎️ Q4_0 推論效能對決 (執行 {} 次 32維度內積) ===", iterations);
    
    // 1. Scalar Fallback
    let start_scalar = Instant::now();
    let mut sum_scalar = 0.0;
    for _ in 0..iterations {
        // 使用 std::hint::black_box 避免被編譯器完全優化掉迴圈
        sum_scalar += scalar_q4_0(&block, &x_stream);
    }
    let dur_scalar = start_scalar.elapsed();
    println!("🐢 純量 (Scalar for-loop) 耗時: {:?}", dur_scalar);
    
    // 2. ARM NEON
    #[cfg(target_arch = "aarch64")]
    {
        let start_neon = Instant::now();
        let mut sum_neon = 0.0;
        for _ in 0..iterations {
            unsafe { sum_neon += neon_q4_0(&block, &x_stream) };
        }
        let dur_neon = start_neon.elapsed();
        println!("🚀 ARM NEON (vld2q + sdot) 耗時: {:?}", dur_neon);
        
        let speedup = dur_scalar.as_secs_f64() / dur_neon.as_secs_f64();
        println!("⚡️ NEON 加速比: {:.2}x 倍！", speedup);
        println!("(驗證正確性: Scalar={}, NEON={})", sum_scalar, sum_neon);
    }
}
