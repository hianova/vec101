# vec101 🚀

A highly optimized, `no_std`, `no_alloc` library for computing 1.58-bit (ternary) weights multiplied by continuous INT8 activations. `vec101` acts as a branchless inference engine capable of extracting maximum hardware utilization across x86_64, Apple Silicon (NEON/Metal), and NVIDIA GPUs (CUDA).

## Features

- **Extreme SIMD Performance**: Core loops are heavily vectorized utilizing AVX2 (`_mm256_maddubs_epi16`) and NEON (`sdot` via `vdotq_s32`), effectively circumventing standard `if/else` execution penalties.
- **Cross-Platform GPU Backends**: 
  - **Apple Metal**: Exploits Unified Memory Architecture (UMA) for true Zero-Copy memory mapping and utilizes `popcount` on native Metal Shading Language (MSL) compute shaders.
  - **NVIDIA CUDA**: Compiles pure Rust directly to PTX utilizing the cutting-edge `cuda-oxide` framework to leverage NVidia hardware `popcount`.
- **Zero Allocations & `no_std`**: Completely heapless runtime. The computation context is strictly `no_std` and pointer-driven, avoiding standard library primitives and locks.
- **INT8 Operator Fusion**: `SwiGLU` and `RMSNorm` are meticulously designed to stay within the INT8 integer space via dynamic Lookup Tables (LUTs) and fixed-point scaling, eliminating FP32 bottlenecks entirely.

## PERFORMANCE

By transforming matrix multiplication into a flattened continuous dual-rail bitmask processor, `vec101` significantly reduces `L1-dcache-misses` and completely avoids scalar branching.

- Native Apple Silicon M1 (CPU NEON): **~20.24 tok/s** end-to-end decoding rate for a 3B Parameter Model.
- Micro-benchmark (1.28M ternary accumulations): **~135.28 µs** on CPU SIMD vs. **1.87 ms** scalar baseline (13.8x Speedup).

### Running Benchmarks

To verify the core engine latency (using `criterion`):
```bash
cargo bench
```

To run the end-to-end simulated LLM decoding loop:
```bash
cargo run --bin run_llm --release
```

## Architecture Details

For comprehensive details on engineering decisions, ternary bitmask memory layouts (`w_pos_bits`/`w_neg_bits`), and GPU integration, please refer to the [SPEC.md](SPEC.md) and [PERF.md](PERF.md) documents.
