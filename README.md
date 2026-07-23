# vec101

**vec101** is a bare-metal, highly specialized heterogeneous inference engine designed for LLMs (Large Language Models) and Embodied AI on extreme edge devices.

## Tech Stack
- **Bit1_58 Quantization**: Ternary weights (-1, 0, 1) mapping natively to branchless CPU instructions for unprecedented throughput.
- **Memory Architecture**: Relies heavily on pre-allocated scratch buffers (`Vec101EngineBorrow`). *(Note: `CpuBackend::compute` currently performs some allocations on the hot path, which is targeted for refactoring).*
- **Integer-Only Tiled FlashAttention**: A revolutionary custom Attention mechanism that completely replaces Floating Point Unit (FPU) usage with fixed-point `i8/i32` arithmetic, making it ideal for MCUs and low-power ARM cores.
- **Heterogeneous Backend**: `vec101_context` abstracts execution across pure CPU (NEON/AVX2), Metal GPU, and CUDA backends via conditional compilation flags.

## Example

```rust
#![no_std]

use vec101::core::context::{vec101_context, QuantType};
use vec101::compute::ComputeContextBuilder;
use vec101::engine::Vec101EngineBorrow;
use vec101::core::types::Vec101SuperBlock;

fn run_inference() {
    // 1. Pre-allocate all memory to guarantee Zero-Allocation at runtime
    // For Bit1_58, weights are packed into superblocks
    let mut w_stream = vec![Vec101SuperBlock::default(); 1024]; 
    let mut x_stream = vec![0i8; 2048];
    let mut out_buffer = vec![0i32; 1024];

    // 2. Build the cross-FFI C-compatible context
    let ctx = ComputeContextBuilder::new()
        .batch_size(1)
        .num_rows(1024)
        .quant_type(QuantType::Bit1_58)
        .build(w_stream.as_mut_ptr(), x_stream.as_mut_ptr(), out_buffer.as_mut_ptr())
        .expect("Failed to initialize vec101 context");

    // 3. Mount the borrow-engine (Zero heap overhead)
    let mut engine = Vec101EngineBorrow::new(ctx);
    
    // 4. Dispatch to the optimal backend (CPU/NEON/Metal/CUDA)
    // Dynamic routing between GEMV (Decoding) and GEMM (Prefill) based on batch_size
    engine.compute();
}
```
