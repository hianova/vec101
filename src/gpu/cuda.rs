use crate::core::vec101_context;
use crate::hal::Vec101Backend;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
const CUDA_SOURCE: &str = r#"
#include <stdint.h>

struct vec101_block {
    uint64_t w_pos_bits[4];
    uint64_t w_neg_bits[4];
};

struct Vec101SuperBlock {
    int16_t scales[8];
    int16_t offsets[8];
    char _padding[32];
    vec101_block blocks[8];
};

extern "C" __global__ void vec101_gemv(
    const Vec101SuperBlock* w_stream,
    const vec101_block* x_stream,
    const float* s_stream,
    float* out_buffer,
    unsigned int blocks_per_row,
    float x_scale,
    unsigned int num_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int batch = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= num_rows) return;
    
    const Vec101SuperBlock* row_w_stream = w_stream + (row * blocks_per_row);
    const vec101_block* batch_x_stream = x_stream + (batch * blocks_per_row * 8);
    
    float row_sum = 0.0f;
    
    for (unsigned int col = 0; col < blocks_per_row; col++) {
        Vec101SuperBlock w_super = row_w_stream[col];
        
        for (unsigned int sub_blk = 0; sub_blk < 8; sub_blk++) {
            float micro_scale = (float)w_super.scales[sub_blk];
            vec101_block w_blk = w_super.blocks[sub_blk];
            vec101_block x_blk = batch_x_stream[col * 8 + sub_blk];
            
            uint64_t pos_prod[4];
            uint64_t neg_prod[4];
            int sum_p = 0;
            int sum_n = 0;
            for(int i=0; i<4; i++) {
                pos_prod[i] = (x_blk.w_pos_bits[i] & w_blk.w_pos_bits[i]) | (x_blk.w_neg_bits[i] & w_blk.w_neg_bits[i]);
                neg_prod[i] = (x_blk.w_pos_bits[i] & w_blk.w_neg_bits[i]) | (x_blk.w_neg_bits[i] & w_blk.w_pos_bits[i]);
                sum_p += __popcll(pos_prod[i]);
                sum_n += __popcll(neg_prod[i]);
            }
            
            row_sum += (float)(sum_p - sum_n) * micro_scale;
        }
    }
    
    float scale = s_stream[row];
    out_buffer[batch * num_rows + row] = row_sum * scale * x_scale;
}

struct BlockQ4_0 {
    int16_t d; 
    unsigned char qs[16];
};


"#;

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
    fn compute(&self, ctx: &vec101_context) {
        let ptx = compile_ptx(CUDA_SOURCE).expect("Failed to compile CUDA PTX");
        self.device
            .load_ptx(ptx, "vec101_module", &["vec101_gemv", "vec101_gemv_q4_0"])
            .expect("Failed to load PTX");

        if ctx.quant_type == crate::core::QuantType::Bit1_58 {
            let num_micro = ctx.blocks_per_row * 8;
            let mut x_blocks = alloc::vec![crate::core::vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; num_micro * ctx.batch_size];
            let x_slice = unsafe {
                core::slice::from_raw_parts(ctx.x_stream, ctx.batch_size * num_micro * 256)
            };
            let x_scale = crate::util::ops::quantize_to_ternary(x_slice, &mut x_blocks);

            let w_len = ctx.num_rows
                * ctx.blocks_per_row
                * core::mem::size_of::<crate::core::Vec101SuperBlock>();
            let s_len = ctx.num_rows * core::mem::size_of::<i32>();
            let out_len = ctx.batch_size * ctx.num_rows * core::mem::size_of::<i32>();

            let dev_w = self
                .device
                .htod_sync_copy(unsafe { core::slice::from_raw_parts(ctx.w_stream, w_len) })
                .unwrap();
            let dev_x = self
                .device
                .htod_sync_copy(unsafe {
                    core::slice::from_raw_parts(
                        x_blocks.as_ptr() as *const u8,
                        x_blocks.len() * core::mem::size_of::<crate::core::vec101_block>(),
                    )
                })
                .unwrap();
            let dev_s = self
                .device
                .htod_sync_copy(unsafe {
                    core::slice::from_raw_parts(ctx.s_stream as *const u8, s_len)
                })
                .unwrap();
            let mut dev_out = self.device.alloc_zeros::<u8>(out_len).unwrap();

            let func = self
                .device
                .get_func("vec101_module", "vec101_gemv")
                .unwrap();
            let cfg = LaunchConfig {
                grid_dim: (ctx.num_rows as u32, ctx.batch_size as u32, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &dev_w,
                        &dev_x,
                        &dev_s,
                        &mut dev_out,
                        ctx.blocks_per_row as u32,
                        x_scale as f32,
                        ctx.num_rows as u32,
                    ),
                )
            }
            .unwrap();

            unsafe {
                self.device
                    .dtoh_sync_copy_into(
                        &dev_out,
                        core::slice::from_raw_parts_mut(ctx.out_buffer as *mut u8, out_len),
                    )
                    .unwrap();
            })
                .unwrap();
            let dev_x = self
                .device
                .htod_sync_copy(unsafe {
                    core::slice::from_raw_parts(ctx.x_stream as *const u8, x_len)
                })
                .unwrap();
            let dev_s = self
                .device
                .htod_sync_copy(unsafe {
                    core::slice::from_raw_parts(ctx.s_stream as *const u8, s_len)
                })
                .unwrap();
            let mut dev_out = self.device.alloc_zeros::<u8>(out_len).unwrap();

            self.device
                .htod_sync_copy_into(
                    unsafe { core::slice::from_raw_parts(ctx.out_buffer as *const u8, out_len) },
                    &mut dev_out,
                )
                .unwrap();

            let func = self
                .device
                .get_func("vec101_module", "vec101_gemv_q4_0")
                .unwrap();
            let cfg = LaunchConfig {
                grid_dim: (ctx.num_rows as u32, ctx.batch_size as u32, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            let x_scale = 1.0f32;
            unsafe {
                func.launch(
                    cfg,
                    (
                        &dev_w,
                        &dev_x,
                        &dev_s,
                        &mut dev_out,
                        ctx.blocks_per_row as u32,
                        x_scale,
                        ctx.num_rows as u32,
                    ),
                )
            }
            .unwrap();

            unsafe {
                self.device
                    .dtoh_sync_copy_into(
                        &dev_out,
                        core::slice::from_raw_parts_mut(ctx.out_buffer as *mut u8, out_len),
                    )
                    .unwrap();
            }
        }
    }
}
