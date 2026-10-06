# 131K context benchmark

Flash 9B processed 131,198 input tokens on the 8GB RTX 3070 Ti: median 105.17 seconds, 1,247 input tokens/sec, and 6.11 GiB peak allocated GPU VRAM across 3 runs. All tested beginning/middle/end facts were recovered.

| Head preparation | Input tokens | Runs | Median seconds | Input tokens/sec | Peak allocated GPU VRAM, GiB |
|---|---:|---:|---:|---:|---:|
| head_gpu_control | 16,350 | 1 | 7.56 | 2,164 | 6.04 |
| head_gpu_control | 65,502 | 1 | 39.09 | 1,676 | 6.48 |
| head_gpu_control | ~131,232 | 1 | HTTP 503 | — | — |
| head_cpu_prepared | 16,350 | 3 | 7.39 | 2,214 | 6.04 |
| head_cpu_prepared | 65,502 | 3 | 39.32 | 1,666 | 6.05 |
| head_cpu_prepared | 131,198 | 3 | 105.17 | 1,247 | 6.11 |

Synthetic text, one GPU family and Flash only; images/Full/ROCm/concurrency not validated.

Native-head control uses one observation per length; prepared-head uses three repetitions.

No native integrated control above 131K fits this card. Small CUDA tests check all typed head operations.

This is a passing tested length, not the maximum capacity or a broad accuracy guarantee.

Production discovery and defaults remain unchanged; the isolated fixed ceiling was 140000.

[Recorded summary and numerical test](results.json) · [Configuration](../../docs/runtime-optimizations.md#experimental-cpu-backed-head-preparation).
