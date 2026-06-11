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
| Naive FP32 (Scalar) | 1          | 1.91 ms    | 1.91 ms                   | 1.00x              |
| **ARM NEON SIMD**   | 1          | 134.69 µs  | 134.69 µs                 | **14.19x**         |
| **ARM NEON SIMD**   | 16 (GEMM)  | 383.49 µs  | 23.97 µs                  | **79.68x** (batch) |
| **GPU Metal**       | 1          | 3.62 ms    | 3.62 ms                   | 0.53x              |
| **GPU Metal**       | 16 (GEMM)  | 12.22 ms   | 763.75 µs                 | 2.50x              |

### Analysis of Core Compute
1. **NEON Acceleration**: The optimized dual-rail bitmask representation with NEON vector instructions (`vpaddlq_s8` + inline `sdot`) completely eliminates branching, achieving a massive **14.19x** speedup over the scalar FP32 implementation.
2. **GEMM Batching Efficiency**: For NEON SIMD, batching 16 inputs increases total execution time to only 383.49 µs, resulting in a normalized time of **23.97 µs** per batch item (a further **5.6x** speedup over Batch=1 NEON due to excellent cache locality of weights).
3. **GPU Metal Micro-workload Penalty**: For small-scale workloads (2.05M parameters), the GPU is slower than NEON due to the overhead of CPU-side ternary activation quantization, shader source compilation, and command buffer dispatch latency executed synchronously on every call.

---

## End-to-End Inference Loop Benchmark (`cargo bench --bench benchmark`)

This benchmark measures the full autoregressive decoding loop simulating **BitNet b1.58 3B Scale** using **INT8 Operator Fusion** and our custom **Spin-Latch Multi-threading** (averaged over 5 iterations).

| Backend | Decode Speed (TPS) | Prefill Latency (TTFT, Batch=128) | Description |
| ------- | ------------------ | --------------------------------- | ----------- |
| **CPU (M1 NEON + Spin-Latch)** | **5.08 tok/s** | **2.01 s** | Natively runs over 8 threads in `no_std`. |
| **GPU (Apple Metal)** | **13.46 tok/s** | **4.93 s** | Offloads execution to the GPU using unified memory. |

### Analysis of End-to-End Loop
1. **GPU Decode Dominance**: At the 3 Billion parameter scale, the GPU Metal backend achieves **13.46 tokens/sec** decode speed, representing a **2.65x** throughput improvement over the 8-thread CPU NEON backend. The compute load dominates the setup/dispatch overhead, letting the GPU compute cores and UMA memory bandwidth shine.
2. **Prefill (TTFT) Trade-off**: The CPU backend achieves a faster prefill time (2.01s vs 4.93s) because the prefill phase (Batch=128) requires quantizing a large volume of activations on the CPU and transfers to/from the GPU. The CPU NEON implementation operates completely in-place, bypassing these dispatch bottlenecks.

