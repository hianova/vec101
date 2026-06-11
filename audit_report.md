# vec101 Test & Benchmark 審計報告

## 總覽

| 類別 | 檔案 | 狀態 | 嚴重度 |
|------|------|------|--------|
| Test | [correctness.rs](file:///Users/kuangtalin/Documents/vec101/tests/correctness.rs) | ❌ **編譯失敗** | 🔴 Critical |
| Test | [loom_test.rs](file:///Users/kuangtalin/Documents/vec101/tests/loom_test.rs) | ⚠️ 永遠被跳過 | 🟡 Medium |
| Bench | [benchmark.rs](file:///Users/kuangtalin/Documents/vec101/benches/benchmark.rs) | ✅ 編譯通過 | 🟢 OK |
| Bench | [perf_bench.rs](file:///Users/kuangtalin/Documents/vec101/benches/perf_bench.rs) | ✅ 編譯通過 | 🟡 Medium |
| Stray | [test_neon.rs](file:///Users/kuangtalin/Documents/vec101/test_neon.rs) | ❌ 無效、不會被執行 | 🟡 Cleanup |

---

## 🔴 Critical：correctness.rs 編譯失敗

`cargo test` 完全無法執行，因為 [correctness.rs](file:///Users/kuangtalin/Documents/vec101/tests/correctness.rs) 存在 **2 個編譯錯誤**：

### 錯誤 1：`w_stream` 型別不匹配（E0308）

```diff
- w_stream: w_stream.as_ptr(),
  // 產生錯誤：expected *const Vec101SuperBlock, found *const vec101_block
```

`vec101_context.w_stream` 已從 `*const vec101_block` 改為 `*const Vec101SuperBlock`，但 `correctness.rs` 仍在使用舊的 `vec101_block` 陣列。

**根因**：`types.rs` 引入了 `Vec101SuperBlock`（包含 8 個 `vec101_block` + `scales` + `offsets`），但測試從未跟上這個重構。

### 錯誤 2：缺少 `state` 欄位（E0063）

```diff
  let ctx = vec101_context {
      // ...
+     state: EngineState::Drafting { target_tokens: 1 }, // 缺少此行
  };
```

`vec101_context` 新增了 `state: EngineState` 欄位，測試未填入。

### 錯誤 3：`naive_fp32_compute` 語意已過時

即使修復編譯錯誤，現有的 naive reference implementation 也不正確——它基於舊的 flat `vec101_block` 布局，沒有處理 `Vec101SuperBlock` 的 per-micro-block `scales` 和 `offsets`。因此即使能跑，cosine similarity 幾乎必定會失敗。

> [!CAUTION]
> 這代表目前**完全沒有**任何可執行的正確性驗證。任何 compute kernel 的改動都是在沒有安全網的情況下進行的。

---

## 🟡 Medium：loom_test.rs 永遠被跳過

[loom_test.rs](file:///Users/kuangtalin/Documents/vec101/tests/loom_test.rs) 使用 `#[cfg(loom)]` 條件編譯，但：

1. `cfg(loom)` 不是 Cargo feature，而是需要 `RUSTFLAGS='--cfg loom'` 才能啟用
2. Cargo.toml 中沒有 `check-cfg` 設定，導致 `unexpected_cfgs` 警告
3. 在正常 `cargo test` 執行下，**整個測試函數會被編譯掉**，import 也標記為 unused
4. 同樣存在 `vec101_context` 缺少 `state` 欄位的問題（但目前因為 `#[cfg(loom)]` 被 gate 掉，所以不會觸發編譯錯誤）

> [!NOTE]
> Loom test 的設計方向正確（測試 spin-latch executor 的並發安全性），但需要：
> - 修復 `vec101_context` 缺少 `state` 欄位
> - 在 Cargo.toml 加入 `check-cfg` 配置消除警告
> - 文件化如何執行 loom test（`RUSTFLAGS='--cfg loom' cargo test`）

---

## 🟢/🟡 Benchmark 審計

### [benchmark.rs](file:///Users/kuangtalin/Documents/vec101/benches/benchmark.rs)（主 benchmark）

**狀態**：✅ 編譯通過、可正常執行

| 項目 | 評估 |
|------|------|
| API 一致性 | ✅ 正確使用 `Vec101SuperBlock`、`EngineState` |
| 量級模擬 | ✅ 模擬 3B 參數規模（~732K rows × 2 SuperBlocks） |
| Decode 測試 | ✅ batch_size=1, 測量 TPS |
| Prefill 測試 | ✅ batch_size=128, 測量 TTFT |
| 統計可靠性 | ⚠️ 只執行 **1 次**，沒有 warmup 或多次迭代取平均 |

**問題**：

1. **只跑一次**：沒有 warmup iteration，第一次跑可能因為 TLB miss、branch predictor cold-start 而偏高。建議至少 warmup 一次再取 3-5 次平均。
2. **`harness = false`** 但使用 `fn main()`：這是正確的模式，但不會被 `cargo bench` 的 criterion 框架統計。它本質上是一個 standalone binary，不是真正的 criterion bench。
3. 缺乏 `black_box()` 防優化。

---

### [perf_bench.rs](file:///Users/kuangtalin/Documents/vec101/benches/perf_bench.rs)（Criterion micro-bench）

**狀態**：✅ 編譯通過

| 項目 | 評估 |
|------|------|
| API 一致性 | ✅ 正確使用 `Vec101SuperBlock`、`EngineState` |
| Criterion 用法 | ✅ 正確使用 `black_box`、`benchmark_group` |
| 對比基準 | ✅ `naive_fp32` vs `vec101_simd` 直接對比 |

**問題**：

1. **未在 Cargo.toml 註冊**：沒有 `[[bench]] name = "perf_bench"` 段落。雖然 Cargo 會自動發現 `benches/*.rs` 並編譯，但 `harness = true`（默認）會嘗試用內建 test harness 而非 criterion。**這會導致 `cargo bench --bench perf_bench` 執行 criterion 的 `main!` 和 Rust 內建 harness 的 `main` 衝突**。

```diff
# Cargo.toml 缺少：
+[[bench]]
+name = "perf_bench"
+path = "benches/perf_bench.rs"
+harness = false
```

2. **naive reference 與 correctness.rs 重複**：`naive_fp32_compute` 被複製到兩個地方。應提取為共用 test utility。

3. **只測 batch_size=1**：沒有測試 GEMM 路徑（batch_size > 1）。

---

## 🟡 Cleanup：stray test_neon.rs

[test_neon.rs](file:///Users/kuangtalin/Documents/vec101/test_neon.rs) 位於專案根目錄，存在以下問題：

```rust
let _ = cmeqq_u8(vdupq_n_u8(0), vdupq_n_u8(0));  // ❌ 函數名錯誤，應為 vceqq_u8
```

- 此檔案不被任何 test target 引用
- 使用了不存在的 NEON intrinsic（`cmeqq_u8`）
- 看起來是早期實驗殘留

> [!TIP]
> 建議直接刪除此檔案。

---

## 測試覆蓋率缺口分析

根據 [SPEC.md](file:///Users/kuangtalin/Documents/vec101/SPEC.md) 和原始碼，以下模組**完全沒有任何測試**：

| 模組 | 功能 | 風險 |
|------|------|------|
| [ops.rs](file:///Users/kuangtalin/Documents/vec101/src/ops.rs) `rmsnorm` | RMSNorm 計算 | 🔴 Core op，無測試 |
| [ops.rs](file:///Users/kuangtalin/Documents/vec101/src/ops.rs) `rope` | Rotary Position Embedding | 🔴 Core op，無測試 |
| [ops.rs](file:///Users/kuangtalin/Documents/vec101/src/ops.rs) `attention` | Scaled Dot-Product Attention | 🔴 Core op，無測試 |
| [ops.rs](file:///Users/kuangtalin/Documents/vec101/src/ops.rs) `rmsnorm_int8` | Fused INT8 RMSNorm | 🔴 Complex SIMD + fixed-point，無測試 |
| [ops.rs](file:///Users/kuangtalin/Documents/vec101/src/ops.rs) `swiglu_int8` | Fused INT8 SwiGLU + LUT | 🔴 Complex SIMD + LUT，無測試 |
| [ops.rs](file:///Users/kuangtalin/Documents/vec101/src/ops.rs) `quantize_to_ternary` | INT8 → 1.58-bit 壓縮 | 🟡 GPU 路徑依賴此函數 |
| [feeder.rs](file:///Users/kuangtalin/Documents/vec101/src/feeder.rs) | Dynamic quantization | 🟡 |
| [tokenizer.rs](file:///Users/kuangtalin/Documents/vec101/src/tokenizer.rs) | Trie tokenizer | 🟡 |
| [memory_tracker.rs](file:///Users/kuangtalin/Documents/vec101/src/memory_tracker.rs) | Resource leak detection | 🟡 |
| [engine.rs](file:///Users/kuangtalin/Documents/vec101/src/engine.rs) | Speculative decoding | 🟡 Mostly stubs |
| [attention.rs](file:///Users/kuangtalin/Documents/vec101/src/attention.rs) | Tiled FlashAttention | 🟡 Stub (mock) |

---

## 優先修復建議

### P0（立即修復）
1. **修復 `correctness.rs`**：更新為 `Vec101SuperBlock` 布局、加入 `state` 欄位、重寫 `naive_fp32_compute` 以匹配新的 per-micro-block scale 語意
2. **在 Cargo.toml 註冊 `perf_bench`**：加入 `[[bench]]` 段落並設定 `harness = false`

### P1（儘快修復）
1. **為 `ops.rs` 的 5 個核心函數加入 unit tests**：
   - `rmsnorm`：用已知的浮點輸入驗證輸出
   - `rmsnorm_int8`：對比 `rmsnorm` 的 f32 結果（cosine similarity）
   - `swiglu_int8`：對比 `swiglu` 的 f32 結果
   - `quantize_to_ternary`：round-trip 測試（已知 ternary 輸入 → 壓縮 → 解壓 → 驗證）
   - `rope`：用手工計算的旋轉值驗證
2. **修復 `loom_test.rs`**：加入 `state` 欄位、在 Cargo.toml 配置 `check-cfg`
3. **刪除 `test_neon.rs`**

### P2（改善品質）
1. **提取共用 test utilities**：`XorShift32` 和 `naive_fp32_compute` 重複了 3 次，應提取到 `tests/common/mod.rs`
2. **為 `benchmark.rs` 加入 warmup + 多次迭代**
3. **為 `perf_bench.rs` 加入 GEMM 路徑（batch_size > 1）的 benchmark**
4. **為 `tokenizer.rs`、`feeder.rs` 加入 unit tests**
