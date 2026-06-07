use crate::types::vec101_context;
use crate::compute::vec101_compute;

/// C-ABI compatible interface to execute the vec101 core engine.
/// This allows GPU runtimes (like Metal or CUDA) to reuse the compute loop,
/// or external languages to call the highly optimized CPU paths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vec101_compute_c(ctx: *const vec101_context) {
    if !ctx.is_null() {
        vec101_compute(&*ctx);
    }
}
