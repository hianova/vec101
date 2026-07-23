use crate::core::vec101_context;
use crate::hal::Vec101Backend;
extern crate alloc;
#[cfg(feature = "std")]
use crate::sync::spawn_thread;
use crate::sync::{Arc, AtomicUsize, Ordering, spin_loop};
use alloc::vec;
#[repr(C, align(64))]
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
        let num_threads = if self.num_threads == 0 {
            1
        } else {
            self.num_threads
        };
        if !is_gemm && num_threads <= 1 {
            let in_features = ctx.blocks_per_row * 2048;
            let mut x_mask = vec![0u64; in_features / 64];
            let x_slice = unsafe { core::slice::from_raw_parts(ctx.x_stream, in_features) };
            for f in 0..in_features {
                if x_slice[f] != 0 {
                    x_mask[f / 64] |= 1 << (f % 64);
                }
            }
            let x_mask_arc = Arc::new(x_mask);
            for row in 0..ctx.num_rows {
                no_std_tool::vec101_compute::process_row_gemv_safe(row, ctx, &x_mask_arc);
            }
            return;
        }
        let in_features = ctx.blocks_per_row * 2048;
        let padded_batch = (ctx.batch_size + 63) & !63;
        let mut x_mask = vec![0u64; in_features / 64];
        let x_t = if is_gemm {
            let mut t = vec![0i8; in_features * padded_batch];
            let x_slice =
                unsafe { core::slice::from_raw_parts(ctx.x_stream, ctx.batch_size * in_features) };
            for b in 0..ctx.batch_size {
                for f in 0..in_features {
                    let val = x_slice[b * in_features + f];
                    t[f * padded_batch + b] = val;
                    if val != 0 {
                        x_mask[f / 64] |= 1 << (f % 64);
                    }
                }
            }
            t
        } else {
            let x_slice = unsafe { core::slice::from_raw_parts(ctx.x_stream, in_features) };
            for f in 0..in_features {
                if x_slice[f] != 0 {
                    x_mask[f / 64] |= 1 << (f % 64);
                }
            }
            vec![]
        };
        let x_t_arc = Arc::new(x_t);
        let x_mask_arc = Arc::new(x_mask);
        let mut row_sums = vec![0i32; padded_batch];
        let num_threads = if self.num_threads == 0 {
            1
        } else {
            self.num_threads
        };
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
                        let x_t_ref = &x_t_arc;
                        let x_mask_ref = &x_mask_arc;
                        let rc_ref = &row_counter;
                        std :: thread :: Builder :: new () . spawn_scoped (s , move | | { let thread_ctx = unsafe { & * (ctx_ptr as * const vec101_context) } ; let mut t_row_sums = vec ! [0i32 ; padded_batch] ; loop { let row = rc_ref . fetch_add (1 , Ordering :: Relaxed) ; if row >= thread_ctx . num_rows { break ; } if is_gemm { no_std_tool::vec101_compute::process_row_gemm_safe(row, thread_ctx, x_t_ref, x_mask_ref, padded_batch, &mut t_row_sums); } else { no_std_tool::vec101_compute::process_row_gemv_safe(row, thread_ctx, x_mask_ref); } } }) . unwrap () ;
                    }
                    loop {
                        let row = row_counter.fetch_add(1, Ordering::Relaxed);
                        if row >= ctx.num_rows {
                            break;
                        }
                        if is_gemm {
                            no_std_tool::vec101_compute::process_row_gemm_safe(row, ctx, &x_t_arc, &x_mask_arc, padded_batch, &mut row_sums);
                        } else {
                            no_std_tool::vec101_compute::process_row_gemv_safe(row, ctx, &x_mask_arc);
                        }
                    }
                });
            }
        } else {
            for row in 0..ctx.num_rows {
                if is_gemm {
                    no_std_tool::vec101_compute::process_row_gemm_safe(row, ctx, &x_t_arc, &x_mask_arc, padded_batch, &mut row_sums);
                } else {
                    no_std_tool::vec101_compute::process_row_gemv_safe(row, ctx, &x_mask_arc);
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{QuantType, Vec101SuperBlock, vec101_block};
    #[test]
    fn test_cpu_backend_coverage() {
        let backend = CpuBackend::new(0);
        assert_eq!(backend.num_threads, 0);
        let batch_size = 1;
        let num_rows = 1;
        let blocks_per_row = 1;
        let mut x_stream = alloc :: vec ! [0i8 ; 2048];
        let mut w_stream = alloc::vec![Vec101SuperBlock {
            scales: [0; 8],
            offsets: [0; 8],
            _padding: [0; 32],
            blocks: [vec101_block {
                w_pos_bits: [0; 4],
                w_neg_bits: [0; 4]
            }; 8]
        }];
        let mut s_stream = alloc :: vec ! [1i32 ; 1];
        let mut out_buffer = alloc :: vec ! [0i32 ; 1];
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
            enable_liquid: false,
            dt: 0.0,
            liquid_state: core::ptr::null_mut(),
            liquid_tau: core::ptr::null(),
            liquid_out_buffer: core::ptr::null_mut(),
        };
        backend.compute(&ctx);
        let ctx_gemm = vec101_context {
            batch_size: 2,
            ..ctx
        };
        let mut x_stream_gemm = alloc :: vec ! [0i8 ; 4096];
        let mut out_buffer_gemm = alloc :: vec ! [0i32 ; 2];
        let ctx_gemm2 = vec101_context {
            batch_size: 2,
            x_stream: x_stream_gemm.as_ptr(),
            out_buffer: out_buffer_gemm.as_mut_ptr(),
            ..ctx
        };
        backend.compute(&ctx_gemm2);
    }
}
