# vec101 Performance Metrics

This document tracks the end-to-end decoding performance and micro-benchmarks of the `vec101` inference engine. All benchmarks are run under `cargo run --release`.

## Test Environment
- **Architecture**: ARM64 (Apple Silicon M1)
- **Target Feature**: NEON SIMD
- **Workload**: 500,000 blocks of 1.58-bit weights (256 ternary weights per block).

## Core Compute Benchmark (`cargo bench`)

This benchmark strictly measures the execution time of the `vec101_compute` core loop over 128M ternary weight accumulations.

| Implementation Type | Time Taken | Speedup |
| ------------------- | ---------- | ------- |
| Naive FP32 (Scalar) | 100.67 ms  | 1.00x   |
| **ARM NEON SIMD**   | **27.21 ms**   | **3.70x**   |

### Analysis
The transition to NEON Pairwise Accumulations (`vpaddlq_s8`) coupled with the dual-rail ternary bitmask representation (`w_pos_bits`, `w_neg_bits`) entirely circumvents branching penalties, extracting full hardware utilization on Apple Silicon chips.

---

## End-to-End Inference Loop Benchmark

This benchmark measures the full autoregressive decoding loop simulating **Gemma 4 MTP (Multi-Token Prediction)** Speculative Decoding alongside pure **INT8 Operator Fusion** (which skips FP32 activation vectors entirely).

| Metric | Result | Description |
| ------ | ------ | ----------- |
| **TTFT (Time To First Token)** | **~4.78 ms** | Time taken for the Prefill phase (1 token). |
| **Decoding Speed (TPS)** | **~192.43 tok/s** | Total Tokens Per Second outputted by the Speculative Decoding engine. |

### Analysis
The ~192 tok/s throughput fundamentally relies on the **INT8 Operator Fusion** implementation, which calculates `RMSNorm` and `SwiGLU` directly within the `i8` integer space. This mathematical fusion explicitly eliminated over 100ms of quantization bottleneck inherent in conventional 1-bit inference engines.
