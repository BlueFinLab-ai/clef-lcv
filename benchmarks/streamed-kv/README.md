# Streamed active KV benchmark

RTX 3070 Ti 8GB, one GPU, Flash 9B compact NF4, FP16 activations.

Same uncached synthetic early/middle/late facts; 8192 initial chunk. Warmup excluded. Three repetitions per standard cell; long probes single observations.

| Mode | Tokens | Runs | Median seconds | Input tokens/sec | Peak allocated GPU VRAM, GiB |
|---|---:|---:|---:|---:|---:|
| GPU resident · 1K chunks | 16,350 | 3 | 7.55 | 2,164 | 6.20 |
| GPU resident · 1K chunks | 32,734 | 3 | 17.75 | 1,844 | 6.77 |
| Whole-layer CPU RAM offload · 1K chunks | 16,350 | 3 | 9.45 | 1,730 | 5.98 |
| Whole-layer CPU RAM offload · 1K chunks | 32,734 | 3 | 25.11 | 1,303 | 5.98 |
| Whole-layer CPU RAM offload · 1K chunks | 65,502 | 3 | 73.09 | 896 | 6.48 |
| Streaming v1 · 1K chunks / 4K blocks | 16,350 | 3 | 8.83 | 1,852 | 6.01 |
| Streaming v1 · 1K chunks / 4K blocks | 32,734 | 3 | 22.86 | 1,432 | 6.01 |
| Streaming v1 · 1K chunks / 4K blocks | 65,502 | 3 | 62.88 | 1,042 | 6.48 |
| Streaming contiguous · 1K chunks / 4K blocks | 16,350 | 3 | 8.03 | 2,036 | 6.01 |
| Streaming contiguous · 1K chunks / 4K blocks | 32,734 | 3 | 18.28 | 1,791 | 6.01 |
| Streaming contiguous · 1K chunks / 4K blocks | 65,502 | 3 | 44.83 | 1,461 | 6.48 |
| Streaming contiguous · 4K chunks / 4K blocks | 16,350 | 3 | 7.35 | 2,225 | 6.01 |
| Streaming contiguous · 4K chunks / 4K blocks | 32,734 | 3 | 16.47 | 1,987 | 6.01 |
| Streaming contiguous · 4K chunks / 4K blocks | 65,502 | 3 | 39.77 | 1,647 | 6.48 |
| Streaming contiguous · 8K chunks / 8K blocks | 32,734 | 3 | 16.28 | 2,010 | 6.05 |
| Streaming contiguous · 8K chunks / 8K blocks | 65,502 | 3 | 38.72 | 1,692 | 6.48 |
| Streaming contiguous · 8K chunks / 8K blocks (long) | 98,270 | 1 | 68.80 | 1,428 | 7.29 |
| Streaming contiguous · 8K chunks / 8K blocks (long) | ~131,072 | — | HTTP 503 | — | — |

No truncation, sparse token selection, activation quantization, or multi-GPU execution.

Streaming v1 uses head-major pinned CPU storage; contiguous version uses token-major CPU storage and an in-place FP32 accumulation.

Whole-layer baseline restores complete per-layer history and appends CPU tensors with cat.

Streaming changes both transfer/storage strategy and attention partitioning. Larger-chunk comparisons also include reduced transfer frequency.

End-to-end throughput includes preprocessing, transfers, hidden restoration and native head. Backbone throughput uses language_ms only.

Peak allocated GPU VRAM is PyTorch allocation, not nvidia-smi reservation. Staging buffers exclude GQA/query/output/head workspace.

Synthetic facts and numerical attention tests are not a broad model accuracy benchmark.

Full 27B, images, ROCm, other GPU families and concurrent streamed requests were not validated.

Production settings are restored; active offloading remains opt-in and normal discovery limits are not enlarged.

[Recorded summary and numerical tests](results.json). Configuration: see [runtime optimizations](../../docs/runtime-optimizations.md#experimental-streamed-active-kv).
