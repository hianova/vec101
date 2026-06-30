use crate::types::vec101_context;
extern crate alloc;
use alloc::vec;

use crate::sync::{Arc, AtomicUsize, Ordering, spin_loop};
#[cfg(any(feature = "std", loom))]
use crate::sync::spawn_thread;

pub mod avx2;
pub mod neon;
pub mod scalar;

#[cfg(feature = "gpu-metal")]
pub mod metal_backend;

// ==========================================
// Main Dispatcher
// ==========================================

/// Main compute dispatcher.
/// # Safety
/// Caller must ensure that the provided context contains valid, aligned memory pointers.
pub unsafe fn vec101_compute(ctx: &vec101_context) {
    if ctx.batch_size == 0 || ctx.num_rows == 0 {
        return;
    }

    #[cfg(feature = "gpu-metal")]
    {
        if ctx.quant_type == crate::types::QuantType::Bit1_58 {
            let num_micro = ctx.blocks_per_row * 8;
            let mut x_blocks = alloc::vec![crate::types::vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; num_micro * ctx.batch_size];
            let x_slice = unsafe { core::slice::from_raw_parts(ctx.x_stream, ctx.batch_size * num_micro * 256) };
            let x_scale = crate::ops::quantize_to_ternary(x_slice, &mut x_blocks);
            metal_backend::metal_compute(ctx, &x_blocks, x_scale);
            return;
        } else if ctx.quant_type == crate::types::QuantType::Q4_0 {
            metal_backend::metal_compute_q4_0(ctx);
            return;
        }
    }

    let is_gemm = ctx.batch_size > 1;

    let in_features = match ctx.quant_type {
        crate::types::QuantType::Bit1_58 => ctx.blocks_per_row * 2048,
        crate::types::QuantType::Q4_0 => ctx.blocks_per_row * 2048,
    };
    let padded_batch = (ctx.batch_size + 63) & !63; // Pad to 64 for unrolled registers
    
    #[cfg(not(target_arch = "aarch64"))]
    let x_t = if is_gemm {
        let mut t = vec![0i8; in_features * padded_batch];
        let x_slice = unsafe { core::slice::from_raw_parts(ctx.x_stream, ctx.batch_size * in_features) };
        for b in 0..ctx.batch_size {
            for f in 0..in_features {
                t[f * padded_batch + b] = x_slice[b * in_features + f];
            }
        }
        t
    } else {
        vec![]
    };
    
    #[cfg(not(target_arch = "aarch64"))]
    let x_t_arc = Arc::new(x_t);

    #[cfg(target_arch = "aarch64")]
    let mut row_sums_f32 = vec![0.0f32; padded_batch];
    #[cfg(not(target_arch = "aarch64"))]
    let mut row_sums = vec![0i32; padded_batch];
    let num_threads = if ctx.num_threads == 0 { 1 } else { ctx.num_threads };
    let row_counter = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicUsize::new(0));
    let ctx_ptr = ctx as *const vec101_context as usize;

    #[cfg(any(feature = "std", loom))]
    let use_threads = num_threads > 1;
    #[cfg(not(any(feature = "std", loom)))]
    let use_threads = false;

    if use_threads {
        #[cfg(any(feature = "std", loom))]
        for _ in 0..(num_threads - 1) {
            let rc = Arc::clone(&row_counter);
            let comp = Arc::clone(&completed);
            #[cfg(not(target_arch = "aarch64"))]
            let x_t_clone = Arc::clone(&x_t_arc);
            
            spawn_thread(move || {
                let thread_ctx = unsafe { &*(ctx_ptr as *const vec101_context) };
                
                #[cfg(target_arch = "aarch64")]
                let mut row_sums_f32 = vec![0.0f32; padded_batch];
                #[cfg(not(target_arch = "aarch64"))]
                let mut row_sums = vec![0i32; padded_batch];
                
                loop {
                    let row = rc.fetch_add(1, Ordering::Relaxed);
                    if row >= thread_ctx.num_rows {
                        break;
                    }
                    if is_gemm {
                        #[cfg(target_arch = "x86_64")]
                        unsafe { avx2::process_row_avx2_gemm(row, thread_ctx, &x_t_clone, padded_batch, &mut row_sums) };
                        #[cfg(target_arch = "aarch64")]
                        unsafe { neon::process_row_neon_gemm(row, thread_ctx, &mut row_sums_f32) };
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        unsafe { scalar::process_row_scalar_gemm(row, thread_ctx, &x_t_clone, padded_batch, &mut row_sums) };
                    } else {
                        #[cfg(target_arch = "x86_64")]
                        unsafe { avx2::process_row_avx2_gemv(row, thread_ctx) };
                        #[cfg(target_arch = "aarch64")]
                        unsafe { neon::process_row_neon_gemv(row, thread_ctx) };
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        unsafe { scalar::process_row_scalar_gemv(row, thread_ctx) };
                    }
                }
                comp.fetch_add(1, Ordering::Release);
            });
        }
    }

    // Main thread execution
    loop {
        let row = row_counter.fetch_add(1, Ordering::Relaxed);
        if row >= ctx.num_rows {
            break;
        }
        if is_gemm {
            #[cfg(target_arch = "x86_64")]
            avx2::process_row_avx2_gemm(row, ctx, &x_t_arc, padded_batch, &mut row_sums);
            #[cfg(target_arch = "aarch64")]
            neon::process_row_neon_gemm(row, ctx, &mut row_sums_f32);
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            scalar::process_row_scalar_gemm(row, ctx, &x_t_arc, padded_batch, &mut row_sums);
        } else {
            #[cfg(target_arch = "x86_64")]
            avx2::process_row_avx2_gemv(row, ctx);
            #[cfg(target_arch = "aarch64")]
            neon::process_row_neon_gemv(row, ctx);
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            scalar::process_row_scalar_gemv(row, ctx);
        }
    }

    if use_threads {
        while completed.load(Ordering::Acquire) < num_threads - 1 {
            spin_loop();
        }
    }
}
