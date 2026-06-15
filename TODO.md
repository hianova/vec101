# 直接讀取標準的 4-bit / 8-bit 權重並進行運算。讓我為你拆解這套技術的底層實作邏輯：
1. 8-bit 直接運算：完全不需要解碼（Native Support）
如果你用的是 8-bit 量化（INT8），在 M1 上是最爽的，因為 ARMv8.2-A 原生支援。
* 記憶體佈局： 權重在硬碟/記憶體裡就是連續的 i8 陣列。
* 運算方式： CPU 直接用 vld1q_s8 把 16 個 i8 權重載入 NEON 暫存器，然後直接呼叫你最熟悉的 sdot（Signed Dot Product）跟輸入的 Activation 相乘。
* 結論： 8-bit 是「Zero-Decoding」，讀進來直接算，完全不需要預編譯。
2. 4-bit 即時解碼：利用「免費的 ALU 週期」
4-bit 比較麻煩，因為 M1 的 NEON 沒有原生的 4-bit 矩陣乘法指令。你不能直接拿 4-bit 去做 Dot Product。 那怎麼辦？在暫存器裡面「即時解壓縮（Unpack）」！
你可能會想：「即時解壓縮不會拖慢速度嗎？」 答案是：完全不會！因為我們是 Memory-bound！
當 CPU 從 RAM 把 4-bit 權重拉進 L1 Cache 時，ALU（算術邏輯單元）其實是在發呆的。我們正好利用這個發呆的時間，用位元運算把 4-bit 膨脹成 8-bit，然後再餵給 sdot。
NEON 4-bit 即時解碼的實作邏輯（以 Q4_0 為例）：
1. 載入： 一次讀取 16 Bytes 的記憶體（裡面塞了 32 個 4-bit 權重）。
2. 位元遮罩（Masking）：
    * 用 vandq_u8 (AND 0x0F) 把低 4-bit 萃取出來，變成 16 個 8-bit 數字。
    * 用 vshrq_n_u8 (Shift Right 4) 把高 4-bit 萃取出來，變成另外 16 個 8-bit 數字。
3. 減去偏移量（Zero-point）： 把這些 0~15 的無號數，減去 8，變成 -8 ~ +7 的有號數（i8）。
4. 運算： 現在你手上有兩組標準的 i8 向量了！直接丟進 sdot 運算。
這整個解碼過程，在 NEON 裡只需要大約 3~4 個 Clock Cycles。相比於等待 RAM 傳輸資料的幾百個 Cycles，這點解碼時間完全被隱藏（Hidden Latency）了。
3. 如何徹底消滅「預編譯」？引入「區塊量化（Block Quantization）」
為了讓 vec101 能直接吃市面上的模型（例如 GGUF 格式），你需要在 vec101 裡實作「區塊（Block）」的概念。
不要把整個矩陣當成一個巨大的陣列，而是把它切成一個個小 Block（通常是 32 個權重為一組）。 一個標準的 4-bit Block 在記憶體裡長這樣：
#[repr(C, packed)]
struct BlockQ4_0 {
    d: f16,       // 2 Bytes: 這個 Block 的縮放比例 (Scale)
    qs: [u8; 16], // 16 Bytes: 塞了 32 個 4-bit 權重
} // 總共 18 Bytes
你的 vec101 運算流程會變成：
1. 上層 ModelGo 透過 mmap 把 .gguf 檔案直接映射到記憶體。
2. 把這個記憶體指標轉型成 &[BlockQ4_0] 傳給 vec101。
3. vec101 的迴圈直接讀取 BlockQ4_0，在 NEON 暫存器裡即時解開 qs，做完 sdot 後，再乘上 d (Scale) 還原成浮點數。

# vec101 作為「純數學與硬體抽象層」的正交性
1. libm = "0.2.8" ➡️ 【唯一無罪，必須保留】
為什麼在這裡： 因為你把 vec101 寫成了 no_std（無標準庫）。在 no_std 環境下，Rust 的 core 庫是不提供浮點數數學運算的（例如 exp、log、sqrt）。

判決： 保留。 LLM 推論中的 RMSNorm 需要算平方根反比（rsqrt），Softmax 和 SiLU/GELU 激活函數需要算指數（exp）。libm 是純粹的數學庫，沒有任何 OS 系統呼叫，完全符合 vec101 的純運算定位。

2. memmap2 = "0.9.10" ➡️ 【越界！應該拔除】
為什麼在這裡： 你可能用它來把模型權重檔（.safetensors 或 .bin）Zero-copy 映射到記憶體裡。

判決： 拔除並上移。 vec101 是一個「運算引擎」，它根本不應該知道「檔案（File）」或「作業系統（OS）」的存在！記憶體映射是 I/O 的工作。

正確做法： 應該由上層的 ModelGo 去呼叫 memmap2 把檔案載入記憶體，然後把一個乾淨的 &[u8]（記憶體切片）或指標傳遞給 vec101。vec101 只負責算，不管資料是從硬碟來的還是從網路來的。

3. safetensors = "0.7.0" ➡️ 【嚴重越界！絕對要拔除】
為什麼在這裡： 用來解析 HuggingFace 的模型格式（讀取 JSON Header、找 Tensor 的 Offset）。

判決： 拔除並上移。 解析字串、處理 JSON、找檔案偏移量，這些跟 NEON 矩陣乘法有什麼關係？把它放在 vec101 裡，就像是在 F1 賽車的引擎室裡裝了一台發票列印機一樣荒謬。

正確做法： 同樣交給 ModelGo 或專門的 Loader 模組去解析。解析完後，把純粹的權重陣列（Raw Pointers/Slices）餵給 vec101。

4. rkyv = { version = "0.7", features = ["validation"] } ➡️ 【越界！應該拔除】
為什麼在這裡： 你可能想用它來做 KV Cache 的序列化，或者在模組間傳遞 Zero-copy 的資料結構。

判決： 拔除並上移。 我們前面討論過，rkyv 是極度優秀的序列化工具，但它是屬於「狀態與快取層（cdDB / dualCacheFF）」的武器。vec101 內部只需要最原始的 Rust 陣列或自定義的 struct，不需要知道資料被序列化成了什麼格式。

# 4 bit loading optimize
首先，我必須以系統工程師的身份指出一個殘酷的事實：Gemma Q4_0 載入需要 14 秒，絕對不是 4-bit 本身的錯，而是你的載入管線（Load Pipeline）沒有用到 mmap（記憶體映射）。

為什麼左半球 (1.58-bit) 只要 600ms？ 因為它極小（可能只有幾百 MB），就算用最笨的 std::fs::read 把它全部拷貝進 RAM，也只要零點幾秒。

為什麼右半球 (Q4_0) 要 14s？ Gemma 2B/7B 的 Q4_0 檔案大小大約是 1.5GB 到 4GB。如果你花了 14 秒，代表你的程式正在做**「硬碟讀取 ➡️ 記憶體分配 (Allocation) ➡️ 陣列拷貝 (Copy) ➡️ 甚至可能在做反序列化」**。

正確的解法： 如果你對右半球使用 mmap，作業系統只會建立一個虛擬記憶體指標，載入時間會瞬間從 14 秒變成 0.001 秒。真正的硬碟讀取會延遲到推論引擎第一次摸到那塊記憶體時（Page Fault）才發生。

結論一： 不要因為「載入慢」而拔掉 4-bit，因為這個慢在工程上是 100% 可以被 mmap 秒解的。
