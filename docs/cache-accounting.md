# Tensor-only cache accounting

The shared Full and Flash runtime counts retained tensor storage across language
checkpoints and independent image features. Each budget scan uses one shared
deduplication set, so views and immutable hidden chunks shared by multiple
checkpoints or features count once. Mutable hybrid attention snapshots remain
independent and are still cloned before use.

Previously, scans traversed the complete checkpoint dictionaries, including long
CPU token ancestry tuples. That traversal added CPU work proportional to retained
token metadata during preparation, trimming, retention, finish and health checks.
Accounting now visits only each checkpoint's `cache` and `chunks` fields plus the
independent feature tensors. Per-checkpoint byte metadata uses the same tensor-only
fields. Exact-prefix matching still inspects token ancestry when finding a hit.

This does not introduce approximate byte counters. Storage sizes, shared-storage
deduplication, automatic budgets, memory reserves, pressure eviction and retention
limits retain their existing semantics. Reported occupancy measures retained
tensor storage; it differs from total process memory, model weights, active
request workspace and the CUDA allocator's unused reserved blocks.

## Validation on October 4, 2026

CPU tests compare the new result with the original complete-object traversal.
They cover shared views, parent/child checkpoints, features aliasing checkpoint
storage, independent copies, protected eviction, feature eviction, retention,
clearing and metadata that raises if accounting tries to inspect it.

On the inference host, eight synthetic 64-layer cache states with 24,000 token
metadata entries each took a median **124.23 ms** per original scan versus
**2.61 ms** for tensor-only accounting: **47.5× faster**, with identical byte
totals. These small CPU tensor fixtures measure Python bookkeeping, not model
throughput. The same test on the development machine measured 59.39 versus
1.19 ms.

A separate controlled GPU test loaded Full Clef NF4 once on the spare RTX 3090.
Four real text requests of 6,867 tokens each populated four retained checkpoints
using 2,267.5 MiB. The test alternated the old and new accounting methods across
four rounds, reversing method order between rounds. All 32 measured requests hit
the same retained prefixes; selected answers matched and probabilities differed
by less than 0.00001. Original and new storage totals were identical.

Median time including preparation, model/head inference and finish fell from
**584.3 ms to 275.9 ms**, a **52.8% reduction** or **2.12× speedup**. This is a
warm-cache text workload on one GPU, without HTTP or image preprocessing. Gains
depend on retained cache contents; it is not a general 2× claim for cold requests.

## Production email rerun

The same frozen 100-email input, category guide and criteria were sent to each
updated service twice with four outstanding HTTP calls. All 600 calls succeeded
and selected the same categories as the saved pre-fix model results. All 300
second-pass calls reported prefix hits. Encoded token lengths were unchanged.

| Service | Earlier cached repeat | Updated first pass | Updated repeat |
| --- | ---: | ---: | ---: |
| Full, RTX 3090 | 207.49 s | 160.13 s | 155.92 s |
| Flash, RTX 3070 Ti | 106.97 s | 129.25 s | 100.17 s |
| Flash, RTX 2080 Ti | 126.92 s | 107.00 s | 85.52 s |

These production comparisons are observational, with different retained cache
contents, restart history and temperatures. The earlier 2080 Ti run reached
91 °C and reported thermal slowdown; the updated run was substantially cooler.
Use the controlled A/B above to attribute speedup to the code change. Reference
agreement remained 100/100 for Full and 95/100 for each Flash; these are frozen
assistant labels, not independently verified human accuracy.

Raw timing, deployment and validation records are kept outside this repository.

Reproduce the CPU checks without model weights or CUDA:

```sh
python scripts/test_cache_accounting.py
python scripts/benchmark_cache_accounting.py --entries 8 --tokens 24000 --repeats 25
```

Each deployed
service keeps its previous runtime beside the replacement as
`optimized_inference.py.before-cache-accounting-20261004`. Stop the service,
restore that service's saved file and restart it to roll back. Process-local
caches rebuild after a restart.
