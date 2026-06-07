# vec101 🚀

A highly optimized, `no_std`, `no_alloc` library for computing 1D compressed (1-bit) weights multiplied by continuous INT8 activations, primarily leveraging x86_64 AVX2 SIMD instructions for extreme latency reduction.

## Features

- **Extreme Performance**: Branchless, non-divergent hot loops written with AVX2 intrinsics.
- **Hardware-aligned Layout**: Weight blocks (`vec101_block`) are meticulously padded to 32 bytes to cleanly fit in an `__m256i` register.
- **Zero Allocations**: Fully `#![no_alloc]` compliant.
- **Thread Tracking & Memory Checking**: Built-in atomic-based `ScopedResource` logic to track leaks explicitly without a heap allocator.

## PERFORMANCE

By transforming a matrix multiplication into a flattened continuous stream processor, `vec101` reduces `L1-dcache-misses` significantly. 
The internal logic maps highly compressed 1-bit flags (0 for -1, 1 for +1) to bytes without any branches, bypassing standard `if/else` checks.

**Expected Speedup:** 
- The latency is targeted to be **5x to 10x faster** than a pure FP32 double-for-loop equivalent.
- Cache misses are greatly minimized due to prefetching (`_mm_prefetch`) and sequential linear layout.

### Running Benchmarks

To verify the latency (using `criterion`):
```bash
cargo bench
```

To verify the cache miss improvements (using `perf` on a Linux x86_64 host):
```bash
cargo build --release --bin benchmark
perf stat -e cache-misses,instructions,cycles target/release/benchmark
```

## Architecture Details

See `SPEC.md` for in-depth engineering decisions, memory layouts, and the strategy to prevent cache thrashing via Dualcache-ff.
