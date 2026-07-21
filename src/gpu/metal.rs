use crate::core::{Vec101SuperBlock, vec101_block, vec101_context};
use crate::hal::Vec101Backend;
#[cfg(feature = "gpu-metal")]
use core::ffi::c_void;
#[cfg(feature = "gpu-metal")]
use metal::*;
extern crate alloc;
#[cfg(feature = "gpu-metal")]
#[repr(C, align(64))]
pub struct MetalBackend {}
#[cfg(feature = "gpu-metal")]
impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}
use std::sync::OnceLock;
struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}
static METAL_CTX: OnceLock<MetalContext> = OnceLock::new();
fn get_metal_context() -> &'static MetalContext {
    METAL_CTX.get_or_init(|| {
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
        MetalContext {
            device,
            command_queue,
            pipeline_state,
        }
    })
}
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
            let mut x_blocks = alloc :: vec ! [crate :: core :: vec101_block { w_pos_bits : [0 ; 4] , w_neg_bits : [0 ; 4] } ; num_micro * ctx . batch_size];
            let x_slice = unsafe {
                core::slice::from_raw_parts(ctx.x_stream, ctx.batch_size * num_micro * 256)
            };
            let x_scale = crate::util::ops::quantize_to_ternary(x_slice, &mut x_blocks);
            unsafe { metal_compute_internal(ctx, &x_blocks, x_scale) };
        }
    }
}
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::thread_local;
static W_BUFFER_CACHE: OnceLock<Mutex<HashMap<usize, metal::Buffer>>> = OnceLock::new();
thread_local! { static S_BUFFER : RefCell < Option < metal :: Buffer >> = const { RefCell :: new (None) } ; static OUT_BUFFER : RefCell < Option < metal :: Buffer >> = const { RefCell :: new (None) } ; static X_BUFFER : RefCell < Option < metal :: Buffer >> = const { RefCell :: new (None) } ; }
#[cfg(feature = "gpu-metal")]
#[doc = " # Safety"]
#[doc = " Assumes all context pointers are valid and aligned."]
unsafe fn metal_compute_internal(ctx: &vec101_context, x_blocks: &[vec101_block], x_scale: i32) {
    let _tracker = no_std_tool::debug::ScopedResource::new();
    let mctx = get_metal_context();
    let device = &mctx.device;
    let command_queue = &mctx.command_queue;
    let pipeline_state = &mctx.pipeline_state;
    let w_len =
        (ctx.num_rows * ctx.blocks_per_row * core::mem::size_of::<Vec101SuperBlock>()) as u64;
    let s_len = (ctx.num_rows * core::mem::size_of::<i32>()) as u64;
    let out_len = (ctx.batch_size * ctx.num_rows * core::mem::size_of::<i32>()) as u64;
    let x_len = core::mem::size_of_val(x_blocks) as u64;
    let w_ptr = ctx.w_stream as usize;
    let w_buffer = {
        let mut cache = W_BUFFER_CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        if let Some(buf) = cache.get(&w_ptr) {
            buf.clone()
        } else {
            let buf = device.new_buffer_with_data(
                ctx.w_stream as *const c_void,
                w_len,
                MTLResourceOptions::StorageModeShared,
            );
            cache.insert(w_ptr, buf.clone());
            buf
        }
    };
    let s_buffer = S_BUFFER.with(|b| {
        let mut b = b.borrow_mut();
        if b.is_none() || b.as_ref().unwrap().length() < s_len {
            *b = Some(device.new_buffer_with_data(
                ctx.s_stream as *const c_void,
                s_len,
                MTLResourceOptions::StorageModeShared,
            ));
        } else {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    ctx.s_stream as *const u8,
                    b.as_ref().unwrap().contents() as *mut u8,
                    s_len as usize,
                );
            }
        }
        b.as_ref().unwrap().clone()
    });
    let mut out_data = alloc :: vec ! [0u8 ; out_len as usize];
    let out_buffer = OUT_BUFFER.with(|b| {
        let mut b = b.borrow_mut();
        if b.is_none() || b.as_ref().unwrap().length() < out_len {
            *b = Some(device.new_buffer_with_data(
                out_data.as_ptr() as *const c_void,
                out_len,
                MTLResourceOptions::StorageModeShared,
            ));
        }
        b.as_ref().unwrap().clone()
    });
    let x_buffer = X_BUFFER.with(|b| {
        let mut b = b.borrow_mut();
        if b.is_none() || b.as_ref().unwrap().length() < x_len {
            *b = Some(device.new_buffer_with_data(
                x_blocks.as_ptr() as *const c_void,
                x_len,
                MTLResourceOptions::StorageModeShared,
            ));
        } else {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    x_blocks.as_ptr() as *const u8,
                    b.as_ref().unwrap().contents() as *mut u8,
                    x_len as usize,
                );
            }
        }
        b.as_ref().unwrap().clone()
    });
    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();
    compute_encoder.set_compute_pipeline_state(pipeline_state);
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
    let x_scale_f32 = x_scale as f32;
    compute_encoder.set_bytes(
        5,
        core::mem::size_of::<f32>() as u64,
        &x_scale_f32 as *const _ as *const c_void,
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
    command_buffer.commit();
    command_buffer.wait_until_completed();
    core::ptr::copy_nonoverlapping(
        out_buffer.contents() as *const u8,
        ctx.out_buffer as *mut u8,
        out_len as usize,
    );
}
#[cfg(feature = "gpu-metal")]
unsafe fn metal_compute_q4_0_internal(ctx: &vec101_context) {
    unimplemented!("Q4_0 not implemented for Metal");
}
