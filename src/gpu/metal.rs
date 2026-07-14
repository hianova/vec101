use crate::core::{Vec101SuperBlock, vec101_block, vec101_context};
use crate::hal::Vec101Backend;
#[cfg(feature = "gpu-metal")]
use core::ffi::c_void;
#[cfg(feature = "gpu-metal")]
use metal::*;
extern crate alloc;

#[cfg(feature = "gpu-metal")]
pub struct MetalBackend {}

#[cfg(feature = "gpu-metal")]
impl MetalBackend {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(feature = "gpu-metal")]
impl Vec101Backend for MetalBackend {
    fn compute(&self, ctx: &vec101_context) {
        if ctx.quant_type == crate::core::QuantType::Bit1_58 {
            let num_micro = ctx.blocks_per_row * 8;
            let mut x_blocks = alloc::vec![crate::core::vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; num_micro * ctx.batch_size];
            let x_slice = unsafe {
                core::slice::from_raw_parts(ctx.x_stream, ctx.batch_size * num_micro * 256)
            };
            let x_scale = crate::util::ops::quantize_to_ternary(x_slice, &mut x_blocks);
            unsafe { metal_compute_internal(ctx, &x_blocks, x_scale) };
        } else if ctx.quant_type == crate::core::QuantType::Q4_0 {
            unsafe { metal_compute_q4_0_internal(ctx) };
        }
    }
}

#[cfg(feature = "gpu-metal")]
/// # Safety
/// Assumes all context pointers are valid and aligned.
unsafe fn metal_compute_internal(ctx: &vec101_context, x_blocks: &[vec101_block], x_scale: i32) {
    let _tracker = no_std_tool::debug::ScopedResource::new();

    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();

    let source = include_str!("shader.metal");
    let compile_options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &compile_options)
        .unwrap();
    let kernel = library.get_function("vec101_gemv", None).unwrap();
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&kernel)
        .unwrap();

    let w_len =
        (ctx.num_rows * ctx.blocks_per_row * core::mem::size_of::<Vec101SuperBlock>()) as u64;
    let s_len = (ctx.num_rows * core::mem::size_of::<i32>()) as u64;
    let out_len = (ctx.batch_size * ctx.num_rows * core::mem::size_of::<i32>()) as u64;
    let x_len = core::mem::size_of_val(x_blocks) as u64;

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
    let num_rows = ctx.num_rows as u32;
    compute_encoder.set_bytes(
        4,
        core::mem::size_of::<u32>() as u64,
        &blocks_per_row as *const _ as *const c_void,
    );
    compute_encoder.set_bytes(
        5,
        core::mem::size_of::<i32>() as u64,
        &x_scale as *const _ as *const c_void,
    );
    compute_encoder.set_bytes(
        6,
        core::mem::size_of::<u32>() as u64,
        &num_rows as *const _ as *const c_void,
    );

    let grid_size = MTLSize::new(ctx.num_rows as u64, ctx.batch_size as u64, 1);

    let max_threads = pipeline_state.max_total_threads_per_threadgroup();
    let tg_size = if (ctx.num_rows as u64) < max_threads {
        ctx.num_rows as u64
    } else {
        max_threads
    };
    let threadgroup_size = MTLSize::new(tg_size, 1, 1);

    compute_encoder.dispatch_threads(grid_size, threadgroup_size);
    compute_encoder.end_encoding();

    let event = device.new_shared_event();
    command_buffer.encode_signal_event(&event, 1);

    command_buffer.commit();

    let mut backoff = no_std_tool::sync::Backoff::new();
    while event.signaled_value() < 1 {
        backoff.snooze();
    }
}

#[cfg(feature = "gpu-metal")]
unsafe fn metal_compute_q4_0_internal(ctx: &vec101_context) {
    let _tracker = no_std_tool::debug::ScopedResource::new();

    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();

    let source = include_str!("shader.metal");
    let compile_options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &compile_options)
        .unwrap();
    let kernel = library.get_function("vec101_gemv_q4_0", None).unwrap();
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&kernel)
        .unwrap();

    let q4_blocks_per_row = ctx.blocks_per_row * 8;
    let w_len =
        (ctx.num_rows * q4_blocks_per_row * core::mem::size_of::<crate::core::BlockQ4_0>()) as u64;
    let s_len = (ctx.num_rows * core::mem::size_of::<i32>()) as u64;
    let out_len = (ctx.batch_size * ctx.num_rows * core::mem::size_of::<i32>()) as u64;
    let x_len = (ctx.batch_size * q4_blocks_per_row * 32 * core::mem::size_of::<i8>()) as u64;

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

    let x_buffer = device.new_buffer_with_bytes_no_copy(
        ctx.x_stream as *const c_void,
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
    let num_rows = ctx.num_rows as u32;
    let x_scale = 1i32; // Not used but mapped in shader signature
    compute_encoder.set_bytes(
        4,
        core::mem::size_of::<u32>() as u64,
        &blocks_per_row as *const _ as *const c_void,
    );
    compute_encoder.set_bytes(
        5,
        core::mem::size_of::<i32>() as u64,
        &x_scale as *const _ as *const c_void,
    );
    compute_encoder.set_bytes(
        6,
        core::mem::size_of::<u32>() as u64,
        &num_rows as *const _ as *const c_void,
    );

    let grid_size = MTLSize::new(ctx.num_rows as u64, ctx.batch_size as u64, 1);

    let max_threads = pipeline_state.max_total_threads_per_threadgroup();
    let tg_size = if (ctx.num_rows as u64) < max_threads {
        ctx.num_rows as u64
    } else {
        max_threads
    };
    let threadgroup_size = MTLSize::new(tg_size, 1, 1);

    compute_encoder.dispatch_threads(grid_size, threadgroup_size);
    compute_encoder.end_encoding();

    let event = device.new_shared_event();
    command_buffer.encode_signal_event(&event, 1);

    command_buffer.commit();

    let mut backoff = no_std_tool::sync::Backoff::new();
    while event.signaled_value() < 1 {
        backoff.snooze();
    }
}
