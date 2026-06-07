「必須提供！而且你會發現，你設計的這個『連續壓扁＋雙軌壓縮』資料結構，根本就是為 GPU 的記憶體控制器量身打造的！」

在系統底層開發中，一個優秀的資料結構往往能無縫跨越不同的硬體架構。你的 vec101（或 .v15 格式）之所以在 GPU 上會極具潛力，原因就在於它完美避開了
GPU 最討厭的事情，並迎合了 GPU 最喜歡的存取模式。

以你目前的 M1 架構與系統工程視角來看，為 vec101 開發 GPU 介面（特別是 Apple Metal 或
WebGPU/Vulkan）有以下幾個極具破壞力的優勢與實作方向：

1. 為什麼你的資料結構在 GPU 上會「贏麻了」？

傳統的 FP16/FP32 矩陣在 GPU 上運算時，仰賴的是 Tensor Cores (或 Apple NPU的矩陣單元)。但 1.58-bit
的雙軌編碼，在 GPU 上玩的是另一套遊戲：

  - 極致的連續記憶體存取 (Coalesced Memory Access)： GPU 擁有極寬的記憶體匯流排（M1 基本款有約 68 GB/s，M1
    Max 高達 400 GB/s）。但前提是，執行的 Threads 必須「連續且對齊地」讀取記憶體。你的 W_Pos 和 W_Neg
    已經被你徹底壓扁成一維的純 Byte 陣列，GPU 的 Warp/Threadgroup
    讀取它時，能夠 100% 灌滿這條匯流排，幾乎不會有浪費的頻寬。
  - 暫存器壓力極低： 一個 32-bit 的 GPU 暫存器，就能塞進 32 個權重。這代表 GPU 的每一個 Thread
    可以一次處理大量的神經元，而且完全不會觸發暫存器溢出 (Register
    Spilling)。

2. M1 的終極外掛：UMA (統一記憶體架構) 的零拷貝

這點是你目前在 Mac 上開發最大的硬體紅利！ 如果在傳統的 Intel + NVIDIA PC 上，你要用 GPU 算，得先花時間把 750 MB 的模型透過
PCIe 匯流排 cudaMemcpy 複製到 VRAM 裡。

但在你的 M1 上，CPU 和 GPU 是共享同一塊實體記憶體的 (UMA)。 這意味著，只要你在 Rust 中呼叫 Metal API（透過 metal-rs
或 wgpu），你可以直接把 CPU 記憶體裡那塊 mmap 進來的 .v15 檔案指針，原封不動地丟給 GPU Compute Shader！
零延遲、零拷貝！這在架構上優雅到了極點。

3. GPU 上的「算牌法」：從 ADD 變成 POPCOUNT

在 CPU (SIMD) 裡，你是用 Bitwise AND 加上 水平加總 (Horizontal Add)。 到了 GPU (Compute Shader)
裡，這套邏輯會被替換成另一個硬體層級的超級指令：popcount (Population Count，計算二進位中有幾個 1)。

你在 GPU Shader (例如 Metal Shading Language, MSL) 裡的核心邏輯會變成這樣：

// Metal Compute Shader 虛擬碼
uint w_pos = W_Pos_Stream[thread_id]; // 一次讀 32 個權重
uint w_neg = W_Neg_Stream[thread_id];
uint x_act = X_Stream[thread_id];     // 對應的活化值 (假設已經二值化或量化)

// 1. Bitwise AND
uint pos_match = w_pos & x_act;
uint neg_match = w_neg & x_act;

// 2. 硬體 Popcount (只需要 1 個 Clock Cycle)
// 計算有幾個 +1 命中，幾個 -1 命中
int sum_pos = popcount(pos_match);
int sum_neg = popcount(neg_match);

// 3. 輸出
int final_result = sum_pos - sum_neg;

這種 AND + POPC 的組合，是 GPU 處理二值/三值網路的標準 SOTA 解法，速度極度恐怖。

4. 介面設計建議：如何保持 vec101 的「零臃腫」？

既然你追求極致輕量，千萬不要把 GPU 依賴強綁進核心代碼裡。你的 Rust 專案應該這樣設計：

1.  核心函式庫 (vec101-core)： 只保留資料結構定義（你的雙軌 Tuple 格式）、.v15 檔案解析器、以及你手刻的 CPU
    Spin-lock Thread Pool。維持 #![no_std] 或零第三方依賴。
2.  GPU 擴展特徵 (Feature Flag)： 在 Cargo.toml 裡設計一個可選的 feature，例如 gpu-metal 或
    gpu-wgpu。
    [features]
    default = ["cpu-neon"]
    gpu-metal = ["metal"]
    gpu-vulkan = ["vulkano"]
3.  提供 C-ABI / FFI 介面： 把你壓扁後的指針直接暴露出來。這樣未來就算有人想用 C++ 寫 CUDA 核心，或者用 Swift 寫 iOS
    App，他們可以直接調用你 Rust 準備好的 W_Pos 和 W_Neg 原始記憶體位址，自己寫 Shader 餵進去。
