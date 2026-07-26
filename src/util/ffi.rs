use crate::compute::vec101_compute;
use crate::core::vec101_context;
#[doc = " C-ABI compatible interface to execute the vec101 core engine."]
#[doc = " This allows GPU runtimes (like Metal or CUDA) to reuse the compute loop,"]
#[doc = " or external languages to call the highly optimized CPU paths."]
#[doc = " # Safety"]
#[doc = " Performs raw pointer dereferences. The caller MUST provide a valid pointer."]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vec101_compute_c(context: *const vec101_context) {
    if !context.is_null() {
        unsafe {
            vec101_compute(&*context);
        }
    }
}
