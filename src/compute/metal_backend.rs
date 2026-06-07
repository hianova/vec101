#[cfg(feature = "gpu-metal")]
use metal::*;
#[cfg(feature = "gpu-metal")]
use core::ffi::c_void;
use crate::types::{vec101_context, vec101_block};

#[cfg(feature = "gpu-metal")]
pub unsafe fn metal_compute(ctx: &vec101_context, x_blocks: &[vec101_block], x_scale: f32) {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    
    let source = include_str!("shader.metal");
    let compile_options = CompileOptions::new();
    let library = device.new_library_with_source(source, &compile_options).unwrap();
    let kernel = library.get_function("vec101_gemv", None).unwrap();
    let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel).unwrap();
    
    let w_len = (ctx.num_rows * ctx.blocks_per_row * core::mem::size_of::<vec101_block>()) as u64;
    let s_len = (ctx.num_rows * core::mem::size_of::<f32>()) as u64;
    let out_len = (ctx.num_rows * core::mem::size_of::<f32>()) as u64;
    let x_len = (x_blocks.len() * core::mem::size_of::<vec101_block>()) as u64;
    
    // Zero-copy wrapping for large weight buffers
    let w_buffer = device.new_buffer_with_bytes_no_copy(
        ctx.w_stream as *const c_void,
        w_len,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let s_buffer = device.new_buffer_with_bytes_no_copy(
        ctx.s_stream as *const c_void,
        s_len,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    let out_buffer = device.new_buffer_with_bytes_no_copy(
        ctx.out_buffer as *const c_void,
        out_len,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    
    // For x_blocks, since it's locally allocated on stack or dynamically, we can also use no_copy as long as we wait
    let x_buffer = device.new_buffer_with_bytes_no_copy(
        x_blocks.as_ptr() as *const c_void,
        x_len,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    
    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();
    
    compute_encoder.set_compute_pipeline_state(&pipeline_state);
    compute_encoder.set_buffer(0, Some(&w_buffer), 0);
    compute_encoder.set_buffer(1, Some(&x_buffer), 0);
    compute_encoder.set_buffer(2, Some(&s_buffer), 0);
    compute_encoder.set_buffer(3, Some(&out_buffer), 0);
    
    let blocks_per_row = ctx.blocks_per_row as u32;
    compute_encoder.set_bytes(4, core::mem::size_of::<u32>() as u64, &blocks_per_row as *const _ as *const c_void);
    compute_encoder.set_bytes(5, core::mem::size_of::<f32>() as u64, &x_scale as *const _ as *const c_void);
    
    let grid_size = MTLSize::new(ctx.num_rows as u64, 1, 1);
    
    // Calculate threadgroup size (M-series usually max 1024, but let's use a safe value like 256 or the pipeline's max)
    let max_threads = pipeline_state.max_total_threads_per_threadgroup();
    let tg_size = if (ctx.num_rows as u64) < max_threads { ctx.num_rows as u64 } else { max_threads };
    let threadgroup_size = MTLSize::new(tg_size, 1, 1);
    
    compute_encoder.dispatch_threads(grid_size, threadgroup_size);
    compute_encoder.end_encoding();
    
    command_buffer.commit();
    command_buffer.wait_until_completed(); // synchronous for now
}
