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
            let x_mask_len = in_features / 64;
            let use_scratch = !ctx.scratch_buffer.is_null() && ctx.scratch_size >= x_mask_len * 8;
            let mut fallback_mask = vec![0u64; if use_scratch { 0 } else { x_mask_len }];
            let x_mask_slice = if use_scratch {
                unsafe {
                    let m = core::slice::from_raw_parts_mut(ctx.scratch_buffer as *mut u64, in_features / 64);
                    m.fill(0);
                    m
                }
            } else {
                fallback_mask.as_mut_slice()
            };

            let x_slice = unsafe { core::slice::from_raw_parts(ctx.x_stream, in_features) };
            for f in 0..in_features {
                if x_slice[f] != 0 {
                    x_mask_slice[f / 64] |= 1 << (f % 64);
                }
            }
            let x_mask_ref = &*x_mask_slice;
            for row in 0..ctx.num_rows {
                no_std_tool::vec101_compute::process_row_gemv_safe(row, ctx, x_mask_ref);
            }
            return;
        }
        let in_features = ctx.blocks_per_row * 2048;
        let padded_batch = (ctx.batch_size + 63) & !63;

        // Calculate required scratch sizes
        let x_mask_len = in_features / 64;
        let x_t_len = if is_gemm { in_features * padded_batch } else { 0 };
        let row_sums_len = if is_gemm { padded_batch * num_threads } else { 0 };

        let total_scratch_needed = (x_mask_len * 8) + x_t_len + (row_sums_len * 4);
        let use_scratch = !ctx.scratch_buffer.is_null() && ctx.scratch_size >= total_scratch_needed;

        // Either borrow from scratch_buffer or allocate on heap
        let mut fallback_x_mask = vec![0u64; if use_scratch { 0 } else { x_mask_len }];
        let mut fallback_x_t = vec![0i8; if use_scratch { 0 } else { x_t_len }];
        let mut fallback_row_sums = vec![0i32; if use_scratch { 0 } else { row_sums_len }];

        let (x_mask_slice, x_t_slice, row_sums_slice) = if use_scratch {
            unsafe {
                let x_mask_ptr = ctx.scratch_buffer as *mut u64;
                let x_t_ptr = ctx.scratch_buffer.add(x_mask_len * 8) as *mut i8;
                let row_sums_ptr = ctx.scratch_buffer.add((x_mask_len * 8) + x_t_len) as *mut i32;
                let m = core::slice::from_raw_parts_mut(x_mask_ptr, x_mask_len);
                let t = core::slice::from_raw_parts_mut(x_t_ptr, x_t_len);
                let r = core::slice::from_raw_parts_mut(row_sums_ptr, row_sums_len);
                m.fill(0);
                t.fill(0);
                r.fill(0);
                (m, t, r)
            }
        } else {
            (fallback_x_mask.as_mut_slice(), fallback_x_t.as_mut_slice(), fallback_row_sums.as_mut_slice())
        };

        if is_gemm {
            let x_slice = unsafe { core::slice::from_raw_parts(ctx.x_stream, ctx.batch_size * in_features) };
            for b in 0..ctx.batch_size {
                for f in 0..in_features {
                    let val = x_slice[b * in_features + f];
                    x_t_slice[f * padded_batch + b] = val;
                    if val != 0 {
                        x_mask_slice[f / 64] |= 1 << (f % 64);
                    }
                }
            }
        } else {
            let x_slice = unsafe { core::slice::from_raw_parts(ctx.x_stream, in_features) };
            for f in 0..in_features {
                if x_slice[f] != 0 {
                    x_mask_slice[f / 64] |= 1 << (f % 64);
                }
            }
        }
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
        
        let x_mask_ref = &*x_mask_slice;
        let x_t_ref = &*x_t_slice;
        
        // We must wrap the row_sums_slice in a struct that implements Send to share it between threads.
        // It's safe because each thread gets a disjoint slice of the buffer.
        struct SyncPtr(*mut i32);
        unsafe impl Send for SyncPtr {}
        unsafe impl Sync for SyncPtr {}
        let row_sums_sync_ptr = SyncPtr(row_sums_slice.as_mut_ptr());

        if use_threads {
            #[cfg(feature = "std")]
            {
                let row_counter = AtomicUsize::new(0);
                std::thread::scope(|s| {
                    for thread_idx in 0..(num_threads - 1) {
                        let rc_ref = &row_counter;
                        let sync_ptr = &row_sums_sync_ptr;
                        std::thread::Builder::new().spawn_scoped(s, move || { 
                            let thread_ctx = unsafe { &*(ctx_ptr as *const vec101_context) }; 
                            let thread_row_sums_ptr = unsafe { sync_ptr.0.add(thread_idx * padded_batch) };
                            let thread_row_sums = unsafe { core::slice::from_raw_parts_mut(thread_row_sums_ptr, padded_batch) };
                            
                            loop { 
                                let row = rc_ref.fetch_add(1, Ordering::Relaxed); 
                                if row >= thread_ctx.num_rows { break; } 
                                if is_gemm { 
                                    no_std_tool::vec101_compute::process_row_gemm_safe(row, thread_ctx, x_t_ref, x_mask_ref, padded_batch, thread_row_sums); 
                                } else { 
                                    no_std_tool::vec101_compute::process_row_gemv_safe(row, thread_ctx, x_mask_ref); 
                                } 
                            } 
                        }).unwrap();
                    }
                    
                    let main_thread_idx = num_threads - 1;
                    let main_row_sums_ptr = unsafe { row_sums_sync_ptr.0.add(main_thread_idx * padded_batch) };
                    let main_row_sums = unsafe { core::slice::from_raw_parts_mut(main_row_sums_ptr, padded_batch) };
                    
                    loop {
                        let row = row_counter.fetch_add(1, Ordering::Relaxed);
                        if row >= ctx.num_rows {
                            break;
                        }
                        if is_gemm {
                            no_std_tool::vec101_compute::process_row_gemm_safe(row, ctx, x_t_ref, x_mask_ref, padded_batch, main_row_sums);
                        } else {
                            no_std_tool::vec101_compute::process_row_gemv_safe(row, ctx, x_mask_ref);
                        }
                    }
                });
            }
        } else {
            let main_row_sums = unsafe { core::slice::from_raw_parts_mut(row_sums_sync_ptr.0, padded_batch) };
            for row in 0..ctx.num_rows {
                if is_gemm {
                    no_std_tool::vec101_compute::process_row_gemm_safe(row, ctx, x_t_ref, x_mask_ref, padded_batch, main_row_sums);
                } else {
                    no_std_tool::vec101_compute::process_row_gemv_safe(row, ctx, x_mask_ref);
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
            scratch_buffer: core::ptr::null_mut(),
            scratch_size: 0,
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
