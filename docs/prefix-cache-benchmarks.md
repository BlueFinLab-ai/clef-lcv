# Branching prefix-cache benchmarks

Validated October 3, 2026. Each process used one GPU; one outstanding request, no batching. Native joint head and model checkpoints unchanged.

| Model | Uncached 100 emails | Cold discovery | Warm repeat | Warm reduction |
| --- | --- | --- | --- | --- |
| Full | 321.4 s | 196.6 s | 192.6 s | 40.1% |
| Flash | 196.2 s | 125.3 s | 103.7 s | 47.2% |

Full used the spare 3090 at 250 W; Flash used the actual 8 GB 3070 Ti at 290 W. Timings include model/cache execution and formatting with pre-encoded inputs, not HTTP. Baselines came from the earlier unchanged uncached path in the same session. Cold discovery includes learning cache boundaries; warm repeats retain branches and warmed kernels. Single runs, not statistical confidence intervals.

All 100 choices matched each model’s uncached reference in both cached trials. Agreement with frozen assistant labels remained 100/100 Full and 95/100 Flash; those labels are not human-verified ground truth. Maximum probability differences were 1.55 percentage points Full and 0.12 Flash.

## Alternating contexts

| Model | One entry, 8 requests | Auto bank, 8 requests | Speedup | Retained storage |
| --- | --- | --- | --- | --- |
| Full | 13.86 s | 2.21 s | 6.26× | 732.5 MiB |
| Flash | 8.60 s | 2.16 s | 3.98× | 334.2 MiB |

Four distinct synthetic contexts were warmed once before timing eight alternating requests. One entry produced zero hits; the bank produced eight. Choices were unchanged.

## Validation and rollout

The test services (Full on an RTX 3090, Flash on an RTX 3070 Ti and an RTX 2080 Ti) were updated with rollback snapshots. Each passed 18 live HTTP checks, including repeated-branch promotion and deepest-prefix hits: text branches, 16 images, changed media, pooling and 1024 fidelity. Both models passed 19 synthetic GPU branching checks, with no choice changes in the final run. Maximum synthetic probability differences were 3.12 percentage points Full and 0.20 Flash. CPU ancestry/isolation, accounting and eviction checks passed.

Automatic VRAM sizing and branching checkpoints are deployed; queued batching remains experimental. KV/recurrent snapshots remain copied rather than paged/shared. See [runtime optimizations](runtime-optimizations.md) for settings, exact-match semantics, media-group boundaries, the 8,192-token cache cutoff and OOM fallback.
