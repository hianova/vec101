# Vec101 Engine Integrated Benchmark

* **Model Scale**     : 40960000 Parameters per layer (10000 x 4096)
* **Hardware Backend**: ARM NEON / Generic CPU Fallback

| Scenario | Quantization | Batch | Threads | Latency (ms) | Throughput (tok/s) |
|----------|--------------|-------|---------|--------------|--------------------|
| Framework Overhead       | BitNet (1.58b) |     1 |       1 |     0.000001 |                  - |
| Decode (Single-Thread)   | BitNet (1.58b) |     1 |       1 |        2.680 |              373.2 |
| Decode (Multi-Thread)    | BitNet (1.58b) |     1 |       8 |        1.152 |              868.4 |
| Prefill TTFT (Batch=128) | BitNet (1.58b) |   128 |       8 |       24.381 |             5250.1 |
| Decode (Single-Thread)   | GGUF (Q4_0)  |     1 |       1 |        0.177 |             5635.8 |
| Decode (Multi-Thread)    | GGUF (Q4_0)  |     1 |       8 |        0.577 |             1733.9 |
| Prefill TTFT (Batch=128) | GGUF (Q4_0)  |   128 |       8 |        5.980 |            21404.9 |

*Note: Throughput is measured in tokens/sec for the configured layer size.*
