use crate::types::vec101_context;
extern crate alloc;
use alloc::vec;

#[cfg(not(loom))]
use alloc::sync::Arc;
#[cfg(loom)]
use loom::sync::Arc;

#[cfg(not(loom))]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(loom))]
use core::hint::spin_loop;
#[cfg(loom)]
use loom::hint::spin_loop;

#[cfg(all(feature = "std", not(loom)))]
use std::thread::spawn as spawn_thread;
#[cfg(loom)]
use loom::thread::spawn as spawn_thread;

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
        } else {
            panic!("Metal backend for Q4_0 is not yet implemented in this dual-engine layout.");
        }
    }

    let num_threads = if ctx.num_threads == 0 { 1 } else { ctx.num_threads };
    let is_gemm = ctx.batch_size > 1;

    let in_features = ctx.blocks_per_row * 2048; // 8 * 256
    let padded_batch = (ctx.batch_size + 63) & !63; // Pad to 64 for unrolled registers
    

    
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
    let x_t_arc = Arc::new(x_t);

    if num_threads == 1 {
        #[cfg(target_arch = "aarch64")]
        let mut row_sums_f32 = vec![0.0f32; padded_batch];
        #[cfg(not(target_arch = "aarch64"))]
        let mut row_sums = vec![0i32; padded_batch];
        
        for row in 0..ctx.num_rows {
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
        return;
    }

    let chunk_size = ctx.num_rows.div_ceil(num_threads);
    let remaining = Arc::new(AtomicUsize::new(num_threads));

    for t in 0..num_threads {
        let start_row = t * chunk_size;
        let mut end_row = start_row + chunk_size;
        if end_row > ctx.num_rows {
            end_row = ctx.num_rows;
        }

        let ctx_addr = ctx as *const vec101_context as usize;
        let rem = remaining.clone();
        let x_t_clone = x_t_arc.clone();

        #[cfg(any(feature = "std", loom))]
        {
            spawn_thread(move || {
                let ctx_ref = unsafe { &*(ctx_addr as *const vec101_context) };
                if start_row < end_row {
                    #[cfg(target_arch = "aarch64")]
                    let mut row_sums_f32 = vec![0.0f32; padded_batch];
                    #[cfg(not(target_arch = "aarch64"))]
                    let mut row_sums = vec![0i32; padded_batch];
                    
                    for row in start_row..end_row {
                        if is_gemm {
                            #[cfg(target_arch = "x86_64")]
                            avx2::process_row_avx2_gemm(row, ctx_ref, &x_t_clone, padded_batch, &mut row_sums);
                            #[cfg(target_arch = "aarch64")]
                            neon::process_row_neon_gemm(row, ctx_ref, &mut row_sums_f32);
                            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                            scalar::process_row_scalar_gemm(row, ctx_ref, &x_t_clone, padded_batch, &mut row_sums);
                        } else {
                            #[cfg(target_arch = "x86_64")]
                            avx2::process_row_avx2_gemv(row, ctx_ref);
                            #[cfg(target_arch = "aarch64")]
                            neon::process_row_neon_gemv(row, ctx_ref);
                            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                            scalar::process_row_scalar_gemv(row, ctx_ref);
                        }
                    }
                }
                rem.fetch_sub(1, Ordering::SeqCst);
            });
        }
        #[cfg(all(not(feature = "std"), not(loom)))]
        {
            let ctx_ref = unsafe { &*(ctx_addr as *const vec101_context) };
            if start_row < end_row {
                #[cfg(target_arch = "aarch64")]
                let mut row_sums_f32 = vec![0.0f32; padded_batch];
                #[cfg(not(target_arch = "aarch64"))]
                let mut row_sums = vec![0i32; padded_batch];
                
                for row in start_row..end_row {
                    if is_gemm {
                        #[cfg(target_arch = "x86_64")]
                        process_row_avx2_gemm(row, ctx_ref, &x_t_clone, padded_batch, &mut row_sums);
                        #[cfg(target_arch = "aarch64")]
                        process_row_neon_gemm(row, ctx_ref, &mut row_sums_f32);
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        process_row_scalar_gemm(row, ctx_ref, &x_t_clone, padded_batch, &mut row_sums);
                    } else {
                        #[cfg(target_arch = "x86_64")]
                        process_row_avx2_gemv(row, ctx_ref);
                        #[cfg(target_arch = "aarch64")]
                        process_row_neon_gemv(row, ctx_ref);
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        process_row_scalar_gemv(row, ctx_ref);
                    }
                }
            }
            rem.fetch_sub(1, Ordering::SeqCst);
        }
    }

    while remaining.load(Ordering::SeqCst) > 0 {
        spin_loop();
    }
}
