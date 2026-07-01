use crate::types::vec101_context;
use crate::hal::Vec101Backend;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }
}

#[cfg(feature = "cuda")]
impl Vec101Backend for CudaBackend {
    fn compute(&self, _ctx: &vec101_context) {
        // TODO: Implement actual cuBLAS or custom PTX invocation using cudarc.
        // The device is held safely. We just need to enqueue kernels.
        unimplemented!("CudaBackend::compute is not yet implemented.");
    }
}
