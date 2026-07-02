use vec101::{vec101_block, vec101_compute, vec101_context};
use vec101::core::{Vec101SuperBlock, BlockQ4_0};

use crate::common::{XorShift32, naive_fp32_compute, naive_q4_0_compute};

fn cosine_similarity(a: &[i32], b: &[i32]) -> f64 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        let x_f = *x as f64;
        let y_f = *y as f64;
        dot_product += x_f * y_f;
        norm_a += x_f * x_f;
        norm_b += y_f * y_f;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[test]
fn test_vec101_correctness() {
    let mut rng = XorShift32::new(42);
    let batch_size = 6;
    let num_rows = 10;
    let blocks_per_row = 1; // 1 SuperBlock = 2048 features
    let total_blocks = num_rows * blocks_per_row;

    let mut w_stream = vec![Vec101SuperBlock { scales: [128; 8], offsets: [0; 8], _padding: [0; 32], blocks: [vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 8] }; total_blocks];
    for super_block in &mut w_stream {
        for block in &mut super_block.blocks {
            for w in &mut block.w_pos_bits {
                *w = rng.next_u64();
            }
            for w in &mut block.w_neg_bits {
                *w = rng.next_u64();
            }
        }
    }

    let mut x_stream = vec![0i8; batch_size * blocks_per_row * 2048];
    for x in &mut x_stream {
        *x = rng.next_i8();
    }

    let mut s_stream = vec![0i32; num_rows];
    for s in &mut s_stream {
        *s = rng.next_i32();
    }

    let mut out_expected = vec![0i32; batch_size * num_rows];
    naive_fp32_compute(
        batch_size,
        num_rows,
        blocks_per_row,
        &w_stream,
        &x_stream,
        &s_stream,
        &mut out_expected,
    );

    let mut out_actual = vec![0i32; batch_size * num_rows];
    let ctx = vec101_context {
        quant_type: vec101::core::QuantType::Bit1_58,
        w_stream: w_stream.as_ptr() as *const u8,
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        batch_size,
        num_rows,
        blocks_per_row,
        num_threads: 4,
        tree_mask: core::ptr::null(),
        tree_size: 0,
        block_size: 16,
        kv_blocks: std::ptr::null(),
        num_blocks: 0,
        hardware_handle: core::ptr::null_mut(),
    };

    unsafe {
        vec101_compute(&ctx);
    }

    let sim = cosine_similarity(&out_expected, &out_actual);
    println!("Cosine Similarity: {}", sim);
    #[cfg(feature = "gpu-metal")]
    let threshold = 0.75;
    #[cfg(not(feature = "gpu-metal"))]
    let threshold = 0.99;
    assert!(sim > threshold, "Similarity {} is below threshold {}", sim, threshold);
}

#[test]
fn test_lock_free_mailbox() {
    use vec101::core::LockFreeMailbox;
    let mailbox = LockFreeMailbox::new();
    assert_eq!(mailbox.try_pop(), None);
    assert_eq!(mailbox.try_push(42), Ok(()));
    assert_eq!(mailbox.try_push(24), Err(24));
    assert_eq!(mailbox.try_pop(), Some(42));
    assert_eq!(mailbox.try_pop(), None);
}

#[test]
fn test_memory_tracker() {
    use vec101::memory_tracker::{ScopedResource, check_memory_leaks, check_thread_drops};
    assert!(check_memory_leaks());
    assert!(check_thread_drops());
    {
        let _resource = ScopedResource::new();
        assert!(!check_memory_leaks());
        assert!(!check_thread_drops());
        {
            let _resource2 = ScopedResource::new();
        }
    }
    assert!(check_memory_leaks());
    assert!(check_thread_drops());
}

#[test]
fn test_ffi_c_interface() {
    use vec101::vec101_compute_c;
    unsafe {
        vec101_compute_c(std::ptr::null());
    }
    
    // Also test with a real context
    let batch_size = 1;
    let num_rows = 2;
    let blocks_per_row = 1;
    let total_blocks = num_rows * blocks_per_row;
    let w_stream = vec![Vec101SuperBlock { scales: [128; 8], offsets: [0; 8], _padding: [0; 32], blocks: [vec101_block { w_pos_bits: [0; 4], w_neg_bits: [0; 4] }; 8] }; total_blocks];
    let x_stream = vec![0i8; batch_size * blocks_per_row * 2048];
    let s_stream = vec![1i32; num_rows];
    let mut out_actual = vec![0i32; batch_size * num_rows];
    let ctx = vec101_context {
        quant_type: vec101::core::QuantType::Bit1_58,
        w_stream: w_stream.as_ptr() as *const u8,
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        batch_size,
        num_rows,
        blocks_per_row,
        num_threads: 1,
        tree_mask: core::ptr::null(),
        tree_size: 0,
        block_size: 16,
        kv_blocks: std::ptr::null(),
        num_blocks: 0,
        hardware_handle: core::ptr::null_mut(),
    };
    unsafe {
        vec101_compute_c(&ctx);
    }
}

#[test]
fn test_tiled_attention() {
    use vec101::attention::IntegerTiledAttention;
    let seq_len = 8;
    let head_dim = 4;
    let q = vec![1i8; seq_len * head_dim];
    let k = vec![1i8; seq_len * head_dim];
    let v = vec![1i8; seq_len * head_dim];
    let output = IntegerTiledAttention::compute_attention_i8(&q, &k, &v, seq_len, head_dim, 4);
    assert_eq!(output.len(), seq_len * head_dim);
}

#[test]
fn test_vec101_q4_0_correctness() {
    let mut rng = XorShift32::new(42);
    let batch_size = 6;
    let num_rows = 10;
    let blocks_per_row = 1;
    let q4_blocks_per_row = blocks_per_row * 8;
    let total_q4_blocks = num_rows * q4_blocks_per_row;

    let mut w_stream = vec![BlockQ4_0 { d: 128, qs: [0; 16] }; total_q4_blocks];
    for block in &mut w_stream {
        block.d = 128;
        for q in &mut block.qs {
            *q = rng.next() as u8;
        }
    }

    let in_features = q4_blocks_per_row * 32;
    let mut x_stream = vec![0i8; batch_size * in_features];
    for x in &mut x_stream {
        *x = rng.next_i8();
    }

    let mut s_stream = vec![0i32; num_rows];
    for s in &mut s_stream {
        *s = rng.next_i32();
    }

    let mut out_expected = vec![0i32; batch_size * num_rows];
    naive_q4_0_compute(
        batch_size,
        num_rows,
        blocks_per_row,
        &w_stream,
        &x_stream,
        &s_stream,
        &mut out_expected,
    );

    let mut out_actual = vec![0i32; batch_size * num_rows];
    let ctx = vec101_context {
        quant_type: vec101::core::QuantType::Q4_0,
        w_stream: w_stream.as_ptr() as *const u8,
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        batch_size,
        num_rows,
        blocks_per_row,
        num_threads: 4,
        tree_mask: core::ptr::null(),
        tree_size: 0,
        block_size: 16,
        kv_blocks: std::ptr::null(),
        num_blocks: 0,
        hardware_handle: core::ptr::null_mut(),
    };

    unsafe {
        vec101_compute(&ctx);
    }

    let sim = cosine_similarity(&out_expected, &out_actual);
    println!("Q4_0 Cosine Similarity: {}", sim);
    let threshold = 0.99;
    assert!(sim > threshold, "Q4_0 Similarity {} is below threshold {}", sim, threshold);
}

#[test]
fn test_vec101_q4_0_correctness_gemv() {
    let mut rng = XorShift32::new(42);
    let batch_size = 1;
    let num_rows = 10;
    let blocks_per_row = 1;
    let q4_blocks_per_row = blocks_per_row * 8;
    let total_q4_blocks = num_rows * q4_blocks_per_row;

    let mut w_stream = vec![BlockQ4_0 { d: 128, qs: [0; 16] }; total_q4_blocks];
    for block in &mut w_stream {
        block.d = 128;
        for q in &mut block.qs {
            *q = rng.next() as u8;
        }
    }

    let in_features = q4_blocks_per_row * 32;
    let mut x_stream = vec![0i8; batch_size * in_features];
    for x in &mut x_stream {
        *x = rng.next_i8();
    }

    let mut s_stream = vec![0i32; num_rows];
    for s in &mut s_stream {
        *s = rng.next_i32();
    }

    let mut out_expected = vec![0i32; batch_size * num_rows];
    naive_q4_0_compute(
        batch_size,
        num_rows,
        blocks_per_row,
        &w_stream,
        &x_stream,
        &s_stream,
        &mut out_expected,
    );

    let mut out_actual = vec![0i32; batch_size * num_rows];
    let ctx = vec101_context {
        quant_type: vec101::core::QuantType::Q4_0,
        w_stream: w_stream.as_ptr() as *const u8,
        x_stream: x_stream.as_ptr(),
        s_stream: s_stream.as_ptr(),
        out_buffer: out_actual.as_mut_ptr(),
        batch_size,
        num_rows,
        blocks_per_row,
        num_threads: 4,
        tree_mask: core::ptr::null(),
        tree_size: 0,
        block_size: 16,
        kv_blocks: std::ptr::null(),
        num_blocks: 0,
        hardware_handle: core::ptr::null_mut(),
    };

    unsafe {
        vec101_compute(&ctx);
    }

    let sim = cosine_similarity(&out_expected, &out_actual);
    println!("Q4_0 GEMV Cosine Similarity: {}", sim);
    let threshold = 0.99;
    assert!(sim > threshold, "Q4_0 GEMV Similarity {} is below threshold {}", sim, threshold);
}
