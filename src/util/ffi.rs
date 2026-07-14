use crate::compute::vec101_compute;
use crate::core::vec101_context;

/// C-ABI compatible interface to execute the vec101 core engine.
/// This allows GPU runtimes (like Metal or CUDA) to reuse the compute loop,
/// or external languages to call the highly optimized CPU paths.
/// # Safety
/// Performs raw pointer dereferences. The caller MUST provide a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vec101_compute_c(ctx: *const vec101_context) {
    if !ctx.is_null() {
        vec101_compute(&*ctx);
    }
}
