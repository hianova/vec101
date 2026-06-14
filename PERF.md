# vec101 Performance Metrics

This document tracks the end-to-end decoding performance and micro-benchmarks of the `vec101` inference engine. All benchmarks are run under `cargo bench`.

## Test Environment
- **Architecture**: ARM64 (Apple Silicon M1)
- **Target Feature**: NEON SIMD
- **Parallelism**: 8 Threads (Custom `no_std` Spin-Latch Executor)
- **Workload**: 11,718,750 blocks of 1.58-bit weights (~3 Billion parameters).

---

## Core Compute Benchmark (`cargo bench --bench perf_bench`)

This benchmark strictly measures the single-thread CPU execution time (or synchronous GPU execution time) of the `vec101_compute` GEMV loop over 2.05M parameters (1000 rows x 2048 features).

| Implementation Type | Batch Size | Time Taken | Normalized Time per Batch | Speedup (vs Naive) |
| ------------------- | ---------- | ---------- | ------------------------- | ------------------ |
| Naive FP32 (Scalar) | 1          | 1.89 ms    | 1.89 ms                   | 1.00x              |
| **ARM NEON SIMD**   | 1          | 136.22 µs  | 136.22 µs                 | **13.8x**          |
| **ARM NEON SIMD**   | 16 (GEMM)  | 502.23 µs  | 31.38 µs                  | **60x** (batch)    |
| **GPU Metal**       | 1          | 3.62 ms    | 3.62 ms                   | 0.52x              |
| **GPU Metal**       | 16 (GEMM)  | 12.22 ms   | 763.75 µs                 | 2.47x              |

### Analysis of Core Compute
1. **NEON Acceleration**: The optimized dual-rail bitmask representation with NEON vector instructions (`vpaddlq_s8` + inline `sdot`) completely eliminates branching, achieving a massive **13.8x** speedup over the scalar FP32 implementation.
2. **GEMM Batching Efficiency**: For NEON SIMD, batching 16 inputs increases total execution time to 502.23 µs, resulting in a normalized time of **31.38 µs** per batch item (a further **4.3x** speedup over Batch=1 NEON). This is achieved through our Zero-Allocation hot loop optimization that maximizes L1 cache hits.
3. **GPU Metal Micro-workload Penalty**: For small-scale workloads (2.05M parameters), the GPU is slower than NEON due to the overhead of CPU-side ternary activation quantization, shader source compilation, and command buffer dispatch latency executed synchronously on every call.

---

## End-to-End Inference Loop Benchmark (`cargo bench --bench benchmark`)

This benchmark measures the full autoregressive decoding loop simulating **BitNet b1.58 3B Scale** using **INT8 Operator Fusion** and our custom **Spin-Latch Multi-threading** (averaged over 5 iterations).

| Backend | Decode Speed (TPS) | Prefill Latency (TTFT, Batch=128) | Description |
| ------- | ------------------ | --------------------------------- | ----------- |
| **CPU (M1 NEON + Spin-Latch)** | **20.24 tok/s** | **1.81 s** | Natively runs over 8 threads in `no_std`. |
| **GPU (Apple Metal)** | **13.46 tok/s** | **4.93 s** | Offloads execution to the GPU using unified memory. |

### Analysis of End-to-End Loop
1. **CPU Decode Dominance**: At the 3 Billion parameter scale, the 8-thread CPU NEON backend correctly utilizing the Spin-Latch executor achieves **20.24 tokens/sec** decode speed, overtaking the Apple Metal GPU. The CPU's Unified Memory access coupled with zero-allocation L1-cache optimizations makes it the dominant backend for Single Batch decoding.
2. **Prefill (TTFT) Trade-off**: The CPU backend achieves a significantly faster prefill time (1.81s vs 4.93s) because the prefill phase (Batch=128) requires quantizing a large volume of activations on the CPU and transfers to/from the GPU. The CPU NEON implementation operates completely in-place, bypassing these dispatch bottlenecks.

