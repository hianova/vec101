#![allow(unused)]
use vec101::{vec101_context, core::QuantType, core::Vec101SuperBlock};

#[test]
#[cfg(target_arch = "aarch64")]
fn test_neon_gemv_complexity() {
    let n: usize = std::env::var("COVOPT_N")
        .unwrap_or_else(|_| "10".to_string())
        .parse()
        .unwrap_or(10);

    let blocks_per_row = n;
    
    let batch_size = 5;
    
    let mut w_stream = vec![0u8; blocks_per_row * core::mem::size_of::<Vec101SuperBlock>()];
    let mut x_stream = vec![0i8; blocks_per_row * 2048 * batch_size];
    let mut s_stream = vec![1i32; 1];
    let mut out_buffer = vec![0i32; 1];

    let ctx = vec101_context {
        quant_type: QuantType::Bit1_58,
        w_stream: w_stream.as_mut_ptr() as *const u8,
        x_stream: x_stream.as_mut_ptr() as *const i8,
        s_stream: s_stream.as_mut_ptr() as *const i32,
        out_buffer: out_buffer.as_mut_ptr(),
        kv_blocks: core::ptr::null(),
        num_blocks: 0,
        block_size: 0,
        batch_size,
        num_rows: 1,
        blocks_per_row,
        num_threads: 1,
        tree_mask: core::ptr::null(),
        tree_size: 0,
        hardware_handle: core::ptr::null_mut(),
    };

    let mut row_sums = vec![0i32; batch_size];

    unsafe {
        vec101::compute::neon::process_row_neon_gemm(0, &ctx, &mut row_sums);
    }
}
