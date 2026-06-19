# vec101 TODO

> [!NOTE]
> All items from the previous heavy refactoring have been fully implemented.
> **Verified**: No compromises or mock structures exist in the codebase. (e.g., `bench_simd` now uses the real `vec101::types::BlockQ4_0` structure).

## Pending Tasks
- [ ] Prepare for Application Layer / LLM Runner integration.

## 調查cache 命中為何overhead 666ns 理論上28-44ns 最多加上運算成本

## 調查CPU GPU gap
現狀： Batch=128 的 Prefill，CPU 花了 1.81 秒，GPU 花了 4.93 秒。（注意：你上一篇提到的 0.164ms 冷啟動，應該是指 mmap load 的初始化時間；而這裡的 1.81s 是真實輸入 128 個 Tokens 去做第一次大型 GEMM 的運算時間）。

挑戰與優化方向：

為什麼 GPU 在 Prefill 輸這麼慘？ 照理說，Batch=128 的 Prefill 是 Compute Bound，GPU 應該要贏過 CPU。這說明你目前 Metal 實作中的「動態激活值量化 (Activation Quantization)」嚴重拖垮了流水線。

解法 (Metal Shader 優化)： 你必須把「動態量化 (Dynamic Quantization, FP16 -> INT8/Ternary)」的邏輯直接寫成 Metal Compute Shader，實現 Operator Fusion (算子融合)。讓 GPU 在讀取 KV Cache 和 Activation 的瞬間，在 GPU 的 SRAM 內直接完成量化並與 1.58-bit 權重相乘。只要消除了 CPU-GPU 之間的同步量化開銷，Metal 在大 Batch 的 Prefill 速度絕對能反超 CPU 幾倍甚至十幾倍。

CPU Prefill 的極限： CPU 花了 1.81s 處理 128 個 Tokens 的 Prefill，速度大約是 70 tok/s。對於 3B 規模的模型來說，這個數字還有優化空間。可以檢查在大型矩陣相乘時，快取未命中 (Cache Miss Rate) 是否因為矩陣分塊 (Block Tiling / Loop Unrolling) 沒針對 Apple M1 的 L2 Cache (12MB/16MB) 進行最佳化而飆高。

# bench againt llama.cpp

## Hardware Optimizations (vec101)
- [ ] **Vectorize FP32 De-quantization & Scalar Operators**: Convert the `norm_i8` to `f32` conversion loop, Softmax, and RoPE computations to ARM NEON SIMD intrinsics.
- [ ] **Fuse Dynamic Activation Quantization in Metal**: Implement a customized MSL Compute Shader to fuse the `FP16 -> INT8` dynamic quantization natively with the GEMM operation to eliminate VRAM roundtrips.
