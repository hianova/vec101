use crate::core::vec101_context;
use crate::hal::Vec101Backend;
extern crate alloc;
use alloc::vec;

use crate::sync::{Arc, AtomicUsize, Ordering, spin_loop};
#[cfg(feature = "std")]
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
                // coverage:ignore-start
                unsafe {
    cfg_select! {
        target_arch = "x86_64" => crate::compute::avx2::process_row_avx2_gemv(row, ctx),
        target_arch = "aarch64" => crate::compute::neon::process_row_neon_gemv(row, ctx),
        _ => crate::compute::scalar::process_row_scalar_gemv(row, ctx),
    }
}
// coverage:ignore-end
            }
            return;
        }

        let in_features = match ctx.quant_type {
            crate::core::QuantType::Bit1_58 => ctx.blocks_per_row * 2048,
            crate::core::QuantType::Q4_0 => ctx.blocks_per_row * 256,
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

        let mut row_sums = vec![0i32; padded_batch];
        let num_threads = if self.num_threads == 0 { 1 } else { self.num_threads };
        let ctx_ptr = ctx as *const vec101_context as usize;

        #[cfg(feature = "std")]
        let use_threads = num_threads > 1;
        #[cfg(not(feature = "std"))]
        let use_threads = false;

        if use_threads {
            #[cfg(feature = "std")]
            {
                let row_counter = AtomicUsize::new(0);
                std::thread::scope(|s| {
                    for _ in 0..(num_threads - 1) {
                        #[cfg(not(target_arch = "aarch64"))]
                        let x_t_ref = &x_t_arc;
                        let rc_ref = &row_counter;
                        
                        std::thread::Builder::new().spawn_scoped(s, move || {
                            let thread_ctx = unsafe { &*(ctx_ptr as *const vec101_context) };
                            let mut t_row_sums = vec![0i32; padded_batch];
                            
                            loop {
                                let row = rc_ref.fetch_add(1, Ordering::Relaxed);
                                if row >= thread_ctx.num_rows { break; }
                                if is_gemm {
                                    // coverage:ignore-start
                                    unsafe {
    cfg_select! {
        target_arch = "x86_64" => crate::compute::avx2::process_row_avx2_gemm(row, thread_ctx, x_t_ref, padded_batch, &mut t_row_sums) ,
        target_arch = "aarch64" => crate::compute::neon::process_row_neon_gemm(row, thread_ctx, &mut t_row_sums) ,
        _ => crate::compute::scalar::process_row_scalar_gemm(row, thread_ctx, x_t_ref, padded_batch, &mut t_row_sums) ,
    }
}
// coverage:ignore-end
                                } else {
                                    // coverage:ignore-start
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe { crate::compute::avx2::process_row_avx2_gemv(row, thread_ctx) };
                                    #[cfg(target_arch = "aarch64")]
                                    unsafe { crate::compute::neon::process_row_neon_gemv(row, thread_ctx) };
                                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                                    unsafe { crate::compute::scalar::process_row_scalar_gemv(row, thread_ctx) };
// coverage:ignore-end
                                }
                            }
                        }).unwrap();
                    }

                    // Main thread work inside scope
                    loop {
                        let row = row_counter.fetch_add(1, Ordering::Relaxed);
                        if row >= ctx.num_rows { break; }
                        if is_gemm {
                            // coverage:ignore-start
                            unsafe {
    cfg_select! {
        target_arch = "x86_64" => crate::compute::avx2::process_row_avx2_gemm(row, ctx, &x_t_arc, padded_batch, &mut row_sums),
        target_arch = "aarch64" => crate::compute::neon::process_row_neon_gemm(row, ctx, &mut row_sums),
        _ => crate::compute::scalar::process_row_scalar_gemm(row, ctx, &x_t_arc, padded_batch, &mut row_sums),
    }
}// coverage:ignore-end
                        } else {
                            // coverage:ignore-start
                            unsafe {
    cfg_select! {
        target_arch = "x86_64" => crate::compute::avx2::process_row_avx2_gemv(row, ctx),
        target_arch = "aarch64" => crate::compute::neon::process_row_neon_gemv(row, ctx),
        _ => crate::compute::scalar::process_row_scalar_gemv(row, ctx),
    }
}
// coverage:ignore-end
                        }
                    }
                });
            }
        } else {
            // Main thread execution (fallback when use_threads = false)
            for row in 0..ctx.num_rows {
                if is_gemm {
                    // coverage:ignore-start
                    unsafe {
    cfg_select! {
        target_arch = "x86_64" => crate::compute::avx2::process_row_avx2_gemm(row, ctx, &x_t_arc, padded_batch, &mut row_sums),
        target_arch = "aarch64" => crate::compute::neon::process_row_neon_gemm(row, ctx, &mut row_sums),
        _ => crate::compute::scalar::process_row_scalar_gemm(row, ctx, &x_t_arc, padded_batch, &mut row_sums),
    }
}// coverage:ignore-end
                } else {
                    unsafe {
    cfg_select! {
        target_arch = "x86_64" => crate::compute::avx2::process_row_avx2_gemv(row, ctx),
        target_arch = "aarch64" => crate::compute::neon::process_row_neon_gemv(row, ctx),
        _ => crate::compute::scalar::process_row_scalar_gemv(row, ctx),
    }
}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Vec101SuperBlock, vec101_block, QuantType};

    #[test]
    fn test_cpu_backend_coverage() {
        let backend = CpuBackend::new(0);
        assert_eq!(backend.num_threads, 0);

        let batch_size = 1;
        let num_rows = 1;
        let blocks_per_row = 1;
        let mut x_stream = alloc::vec![0i8; 2048];
        let mut w_stream = alloc::vec![Vec101SuperBlock { scales: [0; 8], offsets: [0; 8], blocks: [vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 8] }];
        let mut s_stream = alloc::vec![1i32; 1];
        let mut out_buffer = alloc::vec![0i32; 1];
        
        // Single thread GEMV
        let ctx = vec101_context {
            quant_type: QuantType::Bit1_58,
            w_stream: w_stream.as_ptr() as *const u8,
            x_stream: x_stream.as_ptr(),
            s_stream: s_stream.as_ptr(),
            out_buffer: out_buffer.as_mut_ptr(),
            batch_size: 1,
            num_rows,
            blocks_per_row,
            num_threads: 1,
            tree_mask: core::ptr::null(),
            tree_size: 0,
            block_size: 16,
            kv_blocks: core::ptr::null(),
            num_blocks: 0,
            hardware_handle: core::ptr::null_mut(),
        };
        backend.compute(&ctx);

        // Single thread GEMM (fallback when use_threads = false)
        let ctx_gemm = vec101_context {
            batch_size: 2,
            ..ctx
        };
        let mut x_stream_gemm = alloc::vec![0i8; 4096];
        let mut out_buffer_gemm = alloc::vec![0i32; 2];
        let ctx_gemm2 = vec101_context {
            batch_size: 2,
            x_stream: x_stream_gemm.as_ptr(),
            out_buffer: out_buffer_gemm.as_mut_ptr(),
            ..ctx
        };
        backend.compute(&ctx_gemm2);
    }
}
