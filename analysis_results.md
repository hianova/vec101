# Analysis of `vec101` Usage and Encapsulation Recommendations

This document outlines the usage of the `vec101` 1.58-bit ternary inference engine across the `./` (`Universal-Project`) and `../itc` (`itc`) codebases, identifying common patterns, architectural pain points, and proposing suitable APIs for encapsulation.

---

## 1. Survey of `vec101` Usage

Our investigation scanned all active crates inside `Universal-Project` and `itc`. The following 5 crates depend on and invoke `vec101`:

### A. `brains/ModelGo` (Universal-Project)
- **Files**: [speculative_engine.rs](file:///Users/kuangtalin/Documents/Universal-Project/brains/ModelGo/src/science/speculative_engine.rs), [llama.rs](file:///Users/kuangtalin/Documents/Universal-Project/brains/ModelGo/src/assembly/llama.rs), [engine.rs](file:///Users/kuangtalin/Documents/Universal-Project/brains/ModelGo/src/assembly/engine.rs).
- **Context**: Speculative decoding, 0-token Neural Heuristic pruning, and Llama block inference.
- **Usage Pattern**:
  - `engine.rs` utilizes the heap-allocating `ComputeContextBuilder` and `Vec101Engine`.
  - `speculative_engine.rs` and `llama.rs` manually populate raw fields in `vec101_context` (e.g., `w_stream`, `x_stream`, `s_stream`, `out_buffer`, `tree_mask`) and execute `vec101_compute(&ctx)` via raw pointers to customize dynamic output buffers and tree-structured speculative tokens.

### B. `brains/RobotGo` (Universal-Project)
- **Files**: [neural_physics.rs](file:///Users/kuangtalin/Documents/Universal-Project/brains/RobotGo/src/bin/neural_physics.rs), [react.rs](file:///Users/kuangtalin/Documents/Universal-Project/brains/RobotGo/src/react.rs).
- **Context**: Embedded physical-trap loops (ESP32-C6 targets) and differentiable GNN dynamics.
- **Usage Pattern**:
  - `react.rs` uses `Vec101Engine`.
  - `neural_physics.rs` defines `Vec101SuperBlock` and input/output buffers as **stack-allocated arrays** (`activations`, `out_buffer`) to guarantee a **heapless (`no_alloc`)** runtime. It manually constructs `vec101_context` using raw array pointers and calls `unsafe { vec101_compute(&ctx); }`.

### C. `ENLIGHTEN` (itc)
- **Files**: [core.rs](file:///Users/kuangtalin/Documents/itc/ENLIGHTEN/src/core.rs), [lib.rs](file:///Users/kuangtalin/Documents/itc/ENLIGHTEN/src/lib.rs).
- **Context**: Liquid-KAN network layers combined with `vec101` SIMD operations.
- **Usage Pattern**:
  - Builds `Vec101Engine` using `ComputeContextBuilder` at runtime, runs `.compute()`, and extracts outputs via `.get_output().to_vec()`.
  - Incorporates a software fallback when `vec101_compute` outputs zero or fails.

### D. `GENESIS` (itc)
- **Files**: [engine.rs](file:///Users/kuangtalin/Documents/itc/GENESIS/src/llm/engine.rs).
- **Context**: Multi-layer weight sequence execution.
- **Usage Pattern**:
  - Manually constructs `vec101_context` and loops through layers (`model.layers.N.weight`), mutating `ctx.w_stream = ptr` on each iteration to perform successive layer runs.

### E. `KYBERNA` (itc)
- **Files**: [dog_trainer.rs](file:///Users/kuangtalin/Documents/itc/KYBERNA/src/bin/dog_trainer.rs).
- **Context**: Mock GNN decision mapping.
- **Usage Pattern**:
  - Calls `vec101_compute` with a zeroed context: `let ctx: vec101_context = mem::zeroed();`.

---

## 2. Key Pain Points & Vulnerabilities

1. **Unowned Heap Allocations**: The current safe wrapper (`Vec101Engine`) forces heap allocation (`alloc::vec!`) for output buffers. This prevents embedded or real-time systems (like `RobotGo` or deep hot loops in `ModelGo`) from using the safe wrapper, forcing them into unsafe raw pointer manipulation.
2. **Unsafe Layer Swapping**: Callers looping over sequential layer weights (`GENESIS`, `ModelGo`) must manually swap raw `w_stream` pointers inside the loop, leading to potential alignment and out-of-bounds safety risks.
3. **Zeroed Context Crash Risks**: Passing `mem::zeroed()` to `vec101_compute` (like in `KYBERNA`) is hazardous if internal SIMD loops access null pointers.
4. **Parameter Inconsistency**: Incorrect computation of `blocks_per_row`, `num_rows`, and `batch_size` relative to actual buffer lengths results in buffer overflows or segmentation faults.

---

## 3. Recommended Encapsulations & Interfaces

We recommend introducing the following three safe interfaces in `core/vec101`:

### 💡 1. Zero-Allocation Lifetime-Bound Engine (`Vec101EngineBorrow`)
To support `no_alloc` environments, introduce a safe wrapper that borrows caller-provided memory instead of allocating on the heap:

```rust
pub struct Vec101EngineBorrow<'a> {
    ctx: vec101_context,
    _marker_x: core::marker::PhantomData<&'a [i8]>,
    _marker_out: core::marker::PhantomData<&'a mut [i32]>,
}

impl<'a> Vec101EngineBorrow<'a> {
    pub fn new(
        w_stream: &'a [u8],
        x_stream: &'a [i8],
        s_stream: &'a [i32],
        out_buffer: &'a mut [i32],
    ) -> Result<Self, &'static str> {
        // Validate alignments and lengths here
        // ...
        Ok(Self { ... })
    }

    pub fn compute(&mut self) {
        unsafe { vec101_compute(&self.ctx); }
    }
}
```

### 💡 2. Layer Sequence Runner (`LayerSequenceEvaluator`)
Encapsulate multi-layer sequential evaluations to hide pointer-swapping loops:

```rust
pub struct LayerSequenceEvaluator<'a> {
    ctx: vec101_context,
    layer_weights: &'a [*const u8],
}

impl<'a> LayerSequenceEvaluator<'a> {
    pub fn new(ctx: vec101_context, layer_weights: &'a [*const u8]) -> Self {
        Self { ctx, layer_weights }
    }

    pub fn evaluate_all(&mut self) {
        for &w_ptr in self.layer_weights {
            self.ctx.w_stream = w_ptr;
            unsafe { vec101_compute(&self.ctx); }
        }
    }
}
```

### 💡 3. Safe Dummy/Noop Mode
Provide a safe builder method or noop-runner to prevent callers from using `mem::zeroed()`. Calling `.noop()` returns a mocked execution response safely without dereferencing invalid memory.
