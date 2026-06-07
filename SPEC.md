# vec101 Architecture Spec (1.58-bit Edition)

`vec101` eliminates the concept of traditional "matrices" or "neural network layers" entirely. Instead, it processes 1D data streams optimized for 1.58-bit (ternary) weights:
1. **W_Stream (Dual-Rail)**: Highly compressed 1.58-bit weights. Separated into positive `w_pos_stream` and negative `w_neg_stream` bits (256 bits each per block).
2. **X_Stream**: Contiguous INT8 activation arrays.
3. **I_Stream**: Routing indices indicating where the result of a computed block should be scattered.
4. **S_Stream**: Floating point scaling factors applied per block.

## Memory Layout

To perfectly align with Cache Lines (64 Bytes), the `vec101_block` encodes 256 ternary weights using a dual-rail bitmask format:

```rust
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct vec101_block {
    // 256 weights packed into dual 256-bit masks.
    // +1 = w_pos_bits: 1, w_neg_bits: 0
    // -1 = w_pos_bits: 0, w_neg_bits: 1
    //  0 = w_pos_bits: 0, w_neg_bits: 0
    pub w_pos_bits: [u64; 4],
    pub w_neg_bits: [u64; 4],
}
```

## The Hot Loop
The core compute loop (`vec101_compute`) utilizes branchless SIMD programming tailored to the platform architecture:

### AVX2 (x86_64)
- Uses `_mm256_maddubs_epi16` and `_mm256_madd_epi16`.
- Extracts `x_pos` and `x_neg` via `_mm256_and_si256` using the expanded dual-rail masks.
- Horizontally sums 32-byte chunks instantly using `madd` fused instructions.

### NEON (ARM aarch64 - M1/M2/M3)
- **Weight Pre-Decoding**: 1.58-bit ternary weights are decoded into continuous `i8` arrays ONCE per row to bypass intensive runtime bit-fiddling in the hot loops.
- **Signed Dot Product (`sdot`)**: Uses the highly advanced `sdot` instruction (`vdotq_s32`) within inline `asm!` to fuse four `i8` multiplications and `i32` accumulations instantly, achieving peak M1/M2/M3 vectorization throughput.
- **Vectorized Operator Fusion**: `SwiGLU` and `RMSNorm` leverages `vqtbl4q_u8` (Dynamic Vectorized LUT Lookups) and saturating narrow conversions (`vqmovn_s32`) to process INT8 scaling, clamping, and quantization fully within NEON vector registers.

## Pure INT8 Operator Fusion
All linear layer outputs are intercepted before generating `Vec<f32>` arrays. `RMSNorm` and `SwiGLU` have been meticulously designed to take `&[i8]` inputs and directly output `&[i8]` values + a dynamic scale, completely obliterating the `f32` conversion penalty.

- **Dynamic LUT for SwiGLU**: Since inputs are already quantized to `i8`, the `silu` function only has 256 possible input states. `vec101` dynamically builds a 256-entry `i8 -> i8` Lookup Table **once per token** based on the incoming scale. The inner loop of 4096 dimensions is then reduced to a single O(1) table lookup and an `i8 * i8` integer multiplication, eliminating 4096 float calculations.
- **Dual-Pass Fixed-Point RMSNorm**: Model weights are pre-quantized to `i8`. The engine calculates standard deviation squares using native `i32` accumulation (`sum(x_i8 * x_i8)`). The inverse RMS is then folded into a highly precise fixed-point multiplier (e.g. `(prod * M) >> 15`), ensuring the element-wise scaling remains 100% within the integer domain.

## `no_std` Multi-threading (Spin-Latch Executor)
In contrast to standard engines that rely heavily on `std::sync` primitives or heavy libraries like `rayon`, `vec101` implements a custom row-chunking executor that relies solely on `core::sync::atomic::AtomicUsize`.
- **Zero-Lock Synchronization**: Threads synchronize execution completion purely via atomic spin-latches (`fetch_sub`), minimizing context switch latency.
- **Pointer Security**: The computation context (`vec101_context`) safely traverses threads via raw `usize` address boundaries, preventing `*const T` struct locking behaviors.
- **Dual Compatibility**: Through the `std` feature flag, `vec101` falls back seamlessly between concurrent thread spawning (`std::thread::spawn`) and sequential execution for highly restrictive bare-metal hardware.

## Serialization Format
Real model weights (e.g. BitNet b1.58) are serialized into the `Safetensors` format by extracting sub-layers into:
- `{layer}.w_pos_stream` (int64 arrays)
- `{layer}.w_neg_stream` (int64 arrays)
- `{layer}.s_stream` (float32 arrays)
