use crate::types::vec101_context;
extern crate alloc;
use alloc::vec;

use crate::sync::{Arc, AtomicUsize, Ordering, spin_loop};
#[cfg(any(feature = "std", loom))]
use crate::sync::spawn_thread;

pub mod avx2;
pub mod neon;
pub mod scalar;

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

    use crate::hal::Vec101Backend;

    #[cfg(feature = "cuda")]
    {
        if !ctx.hardware_handle.is_null() {
            let device = unsafe { &*(ctx.hardware_handle as *const std::sync::Arc<cudarc::driver::CudaDevice>) };
            let backend = crate::hal::cuda::CudaBackend::new(device.clone());
            backend.compute(ctx);
            return;
        }
    }

    #[cfg(feature = "gpu-metal")]
    {
        // Try Metal Backend if handle provided or just natively (for now, MetalBackend::new() does its own setup)
        let backend = crate::hal::metal::MetalBackend::new();
        backend.compute(ctx);
        return;
    }

    // Fallback to CPU Backend
    let backend = crate::hal::cpu::CpuBackend::new(ctx.num_threads);
    backend.compute(ctx);
}
