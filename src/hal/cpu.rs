use crate::types::vec101_context;
use crate::hal::Vec101Backend;
extern crate alloc;
use alloc::vec;

use crate::sync::{Arc, AtomicUsize, Ordering, spin_loop};
#[cfg(any(feature = "std", loom))]
use crate::sync::spawn_thread;

use crate::compute::{avx2, neon, scalar};

pub struct CpuBackend {
    pub num_threads: usize,
}

impl CpuBackend {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl Vec101Backend for CpuBackend {
    fn compute(&self, ctx: &vec101_context) {
        let is_gemm = ctx.batch_size > 1;
        let num_threads = if self.num_threads == 0 { 1 } else { self.num_threads };

        // Zero allocation fast path for single-threaded GEMV
        if !is_gemm && num_threads <= 1 {
            for row in 0..ctx.num_rows {
                #[cfg(target_arch = "x86_64")]
                unsafe { crate::compute::avx2::process_row_avx2_gemv(row, ctx); }
                #[cfg(target_arch = "aarch64")]
                unsafe { crate::compute::neon::process_row_neon_gemv(row, ctx); }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                unsafe { crate::compute::scalar::process_row_scalar_gemv(row, ctx); }
            }
            return;
        }

        let in_features = match ctx.quant_type {
            crate::types::QuantType::Bit1_58 => ctx.blocks_per_row * 2048,
            crate::types::QuantType::Q4_0 => ctx.blocks_per_row * 256,
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
        let num_threads = if self.num_threads == 0 { 1 } else { self.num_threads };
        let ctx_ptr = ctx as *const vec101_context as usize;

        #[cfg(all(feature = "std", not(loom)))]
        let use_threads = num_threads > 1;
        #[cfg(any(not(feature = "std"), loom))]
        let use_threads = false;

        if use_threads {
            #[cfg(all(feature = "std", not(loom)))]
            {
                let row_counter = AtomicUsize::new(0);
                rayon::scope(|s| {
                    for _ in 0..(num_threads - 1) {
                        #[cfg(not(target_arch = "aarch64"))]
                        let x_t_ref = &x_t;
                        let rc_ref = &row_counter;
                        
                        s.spawn(move |_| {
                            let thread_ctx = unsafe { &*(ctx_ptr as *const vec101_context) };
                            #[cfg(target_arch = "aarch64")]
                            let mut t_row_sums_f32 = vec![0.0f32; padded_batch];
                            #[cfg(not(target_arch = "aarch64"))]
                            let mut t_row_sums = vec![0i32; padded_batch];
                            
                            loop {
                                let row = rc_ref.fetch_add(1, Ordering::Relaxed);
                                if row >= thread_ctx.num_rows { break; }
                                if is_gemm {
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe { crate::compute::avx2::process_row_avx2_gemm(row, thread_ctx, x_t_ref, padded_batch, &mut t_row_sums) };
                                    #[cfg(target_arch = "aarch64")]
                                    unsafe { crate::compute::neon::process_row_neon_gemm(row, thread_ctx, &mut t_row_sums_f32) };
                                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                                    unsafe { crate::compute::scalar::process_row_scalar_gemm(row, thread_ctx, x_t_ref, padded_batch, &mut t_row_sums) };
                                } else {
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe { crate::compute::avx2::process_row_avx2_gemv(row, thread_ctx) };
                                    #[cfg(target_arch = "aarch64")]
                                    unsafe { crate::compute::neon::process_row_neon_gemv(row, thread_ctx) };
                                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                                    unsafe { crate::compute::scalar::process_row_scalar_gemv(row, thread_ctx) };
                                }
                            }
                        });
                    }

                    // Main thread work inside scope
                    loop {
                        let row = row_counter.fetch_add(1, Ordering::Relaxed);
                        if row >= ctx.num_rows { break; }
                        if is_gemm {
                            #[cfg(target_arch = "x86_64")]
                            unsafe { crate::compute::avx2::process_row_avx2_gemm(row, ctx, &x_t, padded_batch, &mut row_sums); }
                            #[cfg(target_arch = "aarch64")]
                            unsafe { crate::compute::neon::process_row_neon_gemm(row, ctx, &mut row_sums_f32); }
                            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                            unsafe { crate::compute::scalar::process_row_scalar_gemm(row, ctx, &x_t, padded_batch, &mut row_sums); }
                        } else {
                            #[cfg(target_arch = "x86_64")]
                            unsafe { crate::compute::avx2::process_row_avx2_gemv(row, ctx); }
                            #[cfg(target_arch = "aarch64")]
                            unsafe { crate::compute::neon::process_row_neon_gemv(row, ctx); }
                            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                            unsafe { crate::compute::scalar::process_row_scalar_gemv(row, ctx); }
                        }
                    }
                });
            }
        } else {
            // Main thread execution (fallback when use_threads = false)
            for row in 0..ctx.num_rows {
                if is_gemm {
                    #[cfg(target_arch = "x86_64")]
                    unsafe { crate::compute::avx2::process_row_avx2_gemm(row, ctx, &x_t, padded_batch, &mut row_sums); }
                    #[cfg(target_arch = "aarch64")]
                    unsafe { crate::compute::neon::process_row_neon_gemm(row, ctx, &mut row_sums_f32); }
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    unsafe { crate::compute::scalar::process_row_scalar_gemm(row, ctx, &x_t, padded_batch, &mut row_sums); }
                } else {
                    #[cfg(target_arch = "x86_64")]
                    unsafe { crate::compute::avx2::process_row_avx2_gemv(row, ctx); }
                    #[cfg(target_arch = "aarch64")]
                    unsafe { crate::compute::neon::process_row_neon_gemv(row, ctx); }
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    unsafe { crate::compute::scalar::process_row_scalar_gemv(row, ctx); }
                }
            }
        }
    }
}
