#![allow(unused)]
use vec101::{core::QuantType, core::Vec101SuperBlock, vec101_context};

#[test]
#[cfg(target_arch = "aarch64")]
fn test_neon_gemv_complexity() {
    let n: usize = std::env::var("COVOPT_N")
        .unwrap_or_else(|_| "10".to_string())
        .parse()
        .unwrap_or(10);

    let n = std::hint::black_box(n);
    let blocks_per_row = n;

    let batch_size = 5;

    static mut W_STREAM: [u8; 1000 * core::mem::size_of::<Vec101SuperBlock>()] = [0u8; 1000 * core::mem::size_of::<Vec101SuperBlock>()];
    static mut X_STREAM: [i8; 1000 * 2048 * 5] = [0i8; 1000 * 2048 * 5];
    static mut S_STREAM: [i32; 1] = [1i32; 1];
    static mut OUT_BUFFER: [i32; 1] = [0i32; 1];
    
    let mut w_stream = unsafe { &mut W_STREAM[..blocks_per_row * core::mem::size_of::<Vec101SuperBlock>()] };
    let mut x_stream = unsafe { &mut X_STREAM[..blocks_per_row * 2048 * batch_size] };
    let mut s_stream = unsafe { &mut S_STREAM[..] };
    let mut out_buffer = unsafe { &mut OUT_BUFFER[..] };

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

    let mut row_sums = [0i32; 5];

    unsafe {
        vec101::compute::neon::process_row_neon_gemm(0, &ctx, &mut row_sums);
    }
    std::hint::black_box(row_sums);
}
