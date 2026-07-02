# Vec101 Engine Integrated Benchmark

* **Model Scale**     : 40960000 Parameters per layer (10000 x 4096)
* **Hardware Backend**: ARM NEON / Generic CPU Fallback

| Scenario | Quantization | Batch | Threads | Latency (ms) | Throughput (tok/s) |
|----------|--------------|-------|---------|--------------|--------------------|
| Framework Overhead       | BitNet (1.58b) |     1 |       1 |     0.000004 |                  - |
| Decode (Single-Thread)   | BitNet (1.58b) |     1 |       1 |        3.629 |              275.6 |
| Decode (Multi-Thread)    | BitNet (1.58b) |     1 |       8 |        1.005 |              995.0 |
| Prefill TTFT (Batch=128) | BitNet (1.58b) |   128 |       8 |       21.670 |             5906.8 |
| Decode (Single-Thread)   | GGUF (Q4_0)  |     1 |       1 |        0.173 |             5785.3 |
| Decode (Multi-Thread)    | GGUF (Q4_0)  |     1 |       8 |        0.571 |             1752.5 |
| Prefill TTFT (Batch=128) | GGUF (Q4_0)  |   128 |       8 |        5.300 |            24152.6 |

*Note: Throughput is measured in tokens/sec for the configured layer size.*
