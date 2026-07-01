use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vec101::types::vec101_context;
use vec101::compute::vec101_compute;

fn bench_overhead(c: &mut Criterion) {
    let mut ctx = vec101_context {
        quant_type: vec101::types::QuantType::Bit1_58,
        w_stream: core::ptr::null(),
        x_stream: core::ptr::null(),
        s_stream: core::ptr::null(),
        out_buffer: core::ptr::null_mut(),
        batch_size: 1,
        num_rows: 1,
        blocks_per_row: 0,
        num_threads: 1,
        tree_mask: core::ptr::null(),
        tree_size: 0,
        block_size: 16,
        kv_blocks: core::ptr::null(),
        num_blocks: 0,
        hardware_handle: core::ptr::null_mut(),
    };
    
    let w_stream = vec![0u8; 1024];
    let x_stream = vec![0i8; 2048];
    let s_stream = vec![1.0f32; 1];
    let mut out_buffer = vec![0.0f32; 1];
    
    ctx.w_stream = w_stream.as_ptr();
    ctx.x_stream = x_stream.as_ptr();
    ctx.s_stream = s_stream.as_ptr();
    ctx.out_buffer = out_buffer.as_mut_ptr();

    c.bench_function("vec101_compute_overhead", |b| b.iter(|| {
        unsafe { vec101_compute(black_box(&ctx)) };
    }));
}

criterion_group!(benches, bench_overhead);
criterion_main!(benches);
