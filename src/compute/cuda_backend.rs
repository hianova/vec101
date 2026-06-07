#[cfg(feature = "gpu-cuda")]
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
#[cfg(feature = "gpu-cuda")]
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
#[cfg(feature = "gpu-cuda")]
use crate::types::{vec101_context, vec101_block};

#[cfg(feature = "gpu-cuda")]
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vec101_gemv(
        w_stream: &[vec101_block],
        x_stream: &[vec101_block],
        s_stream: &[f32],
        mut out_buffer: DisjointSlice<f32>,
        blocks_per_row: u32,
        x_scale: f32,
    ) {
        let idx = thread::index_1d().get() as usize;
        
        if idx >= s_stream.len() {
            return;
        }

        let w_offset = idx * blocks_per_row as usize;
        let mut row_sum = 0i32;

        for i in 0..(blocks_per_row as usize) {
            let w_blk = &w_stream[w_offset + i];
            let x_blk = &x_stream[i];

            for k in 0..4 {
                let pos_prod = (x_blk.w_pos_bits[k] & w_blk.w_pos_bits[k]) | (x_blk.w_neg_bits[k] & w_blk.w_neg_bits[k]);
                let neg_prod = (x_blk.w_pos_bits[k] & w_blk.w_neg_bits[k]) | (x_blk.w_neg_bits[k] & w_blk.w_pos_bits[k]);
                
                row_sum += pos_prod.count_ones() as i32 - neg_prod.count_ones() as i32;
            }
        }
        
        if let Some(out_elem) = out_buffer.get_mut(idx) {
            *out_elem = (row_sum as f32) * s_stream[idx] * x_scale;
        }
    }
}

#[cfg(feature = "gpu-cuda")]
pub unsafe fn cuda_compute(ctx: &vec101_context, x_blocks: &[vec101_block], x_scale: f32) {
    let cuda_ctx = CudaContext::new(0).expect("Failed to initialize CUDA context");
    let stream = cuda_ctx.default_stream();

    let num_rows = ctx.num_rows;
    let blocks_per_row = ctx.blocks_per_row;

    let w_slice = core::slice::from_raw_parts(ctx.w_stream, num_rows * blocks_per_row);
    let s_slice = core::slice::from_raw_parts(ctx.s_stream, num_rows);
    let out_slice = core::slice::from_raw_parts_mut(ctx.out_buffer, num_rows);

    // Transfer buffers to Device
    let w_buf = DeviceBuffer::from_host(&stream, w_slice).unwrap();
    let x_buf = DeviceBuffer::from_host(&stream, x_blocks).unwrap();
    let s_buf = DeviceBuffer::from_host(&stream, s_slice).unwrap();
    let mut out_buf = DeviceBuffer::<f32>::zeroed(&stream, num_rows).unwrap();

    let module = kernels::load(&cuda_ctx).unwrap();

    module
        .vec101_gemv(
            &stream,
            LaunchConfig::for_num_elems(num_rows as u32),
            &w_buf,
            &x_buf,
            &s_buf,
            &mut out_buf,
            blocks_per_row as u32,
            x_scale,
        )
        .unwrap();

    // Copy results back to host
    let result = out_buf.to_host_vec(&stream).unwrap();
    out_slice.copy_from_slice(&result);
}
