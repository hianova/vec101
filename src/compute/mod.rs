use crate::core::vec101_context;
extern crate alloc;
#[cfg(feature = "std")]
pub mod batch;
#[doc = " Main compute dispatcher."]
#[doc = " # Safety"]
#[doc = " Caller must ensure that the provided context contains valid, aligned memory pointers."]
pub unsafe fn vec101_compute(ctx: &vec101_context) {
    if ctx.batch_size == 0 || ctx.num_rows == 0 {
        return;
    }
    use crate::hal::Vec101Backend;
    #[cfg(feature = "cuda")]
    {
        if !ctx.hardware_handle.is_null() {
            let device = unsafe {
                &*(ctx.hardware_handle as *const std::sync::Arc<cudarc::driver::CudaDevice>)
            };
            let backend = crate::gpu::cuda::CudaBackend::new(device.clone());
            backend.compute(ctx);
            return;
        }
    }
    #[cfg(feature = "gpu-metal")]
    {
        let backend = crate::gpu::metal::MetalBackend::new();
        backend.compute(ctx);
        return;
    }
    let backend = crate::hal::cpu::CpuBackend::new(ctx.num_threads);
    backend.compute(ctx);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vec101_compute_early_exit() {
        let ctx = vec101_context {
            batch_size: 0,
            num_rows: 0,
            blocks_per_row: 0,
            num_threads: 0,
            quant_type: crate::core::QuantType::Bit1_58,
            w_stream: core::ptr::null(),
            x_stream: core::ptr::null(),
            s_stream: core::ptr::null(),
            out_buffer: core::ptr::null_mut(),
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
        unsafe {
            vec101_compute(&ctx);
        }
    }
}
