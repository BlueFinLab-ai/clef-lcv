# NVIDIA queue batching comparison

October 5, 2026: [aggregate results](summary-2026-10-05.json) compare the same
100 public synthetic emails with batch caps one, two and four, without a batch
collection delay. First passes warmed GPU kernels/caches; the second pass is
reported. Client concurrency was held fixed across caps. Raw development logs
and operator captures stay outside the published tree.

At four clients, cap two reduced 100-email time by 6.5% on Flash/3090 (400 W),
5.5% on Full/3090 (250 W), and 2.3% on Flash/3070 Ti (290 W). The last is small
enough to warrant caution: these are single warmed passes, not a confidence
interval. All 2000 trial responses preserved the model's serial category choice.
Agreement with the GPT Sol 6.1 reference labels was 96/100 for Flash and 100/100
for Full. Full was tested on a second RTX 3090 at 250 W, not at 400 W.

Cap four was slightly slower than cap two on Flash/3090 and increased p95 latency.
The selected SM86 default is two; other families stay opt-in. Full was tested in
isolation using the same queue path. Each service stays on one GPU.

[Production behavior and bounds](../../docs/runtime-optimizations.md#opportunistic-cuda-text-batching).
[Public dataset](../email-sample/README.md).

## October 6 rerun

[Updated results](email-rerun-2026-10-06.json) compare the same fixture on Flash
3070 Ti (290W), Flash 3090 (400W), and Full 3090 (400W). Both 3090 model runs
used the same physical card sequentially, with the October 5 runtime.
Each configuration has first/warm serial passes and two four-client passes;
Flash/3090 additionally has two confirmation passes after a first-use batch
spike. The main warmed result uses those confirmation passes, with the original
53.61s and 40.61s observations retained in the saved results.

| Configuration | Warm serial, 100 emails | Warm four-client mean | Reference agreement |
|---|---:|---:|---:|
| Flash 9B · RTX 3070 Ti · 290W | 66.77s | 63.45s | 96/100 |
| Flash 9B · RTX 3090 · 400W | 44.25s | 41.56s | 96/100 |
| Full 27B · RTX 3090 · 400W | 125.24s | 119.31s | 100/100 |

All 1,400 measured requests succeeded. Every repeated/concurrent pass preserved
the corresponding model's serial category choices. Four clients are outstanding
HTTP calls; actual GPU batches remain capped at two and admission can choose
single requests. Average prefix reuse was 1,536 tokens: primarily the shared
category guide. These runs do not cache finished answers. Agreement uses the
GPT Sol 6.1 reference labels, not independently adjudicated human truth.

No sampled thermal flags coincided with nonzero sampled GPU utilization; the
3070 Ti reported eight software flags at sampled idle moments. This does not
establish throttling of active work. The original services were restored afterwards.
Loading and unrelated warm-up are excluded;
initialization or runtime GPU compilation can still affect first-use batches.

## October 6 cooling rerun

[Cooling rerun results](email-cooled-2026-10-06.json) preserve the same dataset,
committed runtime, model/GPU combinations, power limits, shared-guide warm-up,
and first/warm serial plus two four-client passes. All 1,200 requests succeeded,
and reference agreement stayed at 96/100 for Flash and 100/100 for Full.

| Configuration | Before: warm C4 mean | After: warm C4 mean | Peak temperature before → after |
|---|---:|---:|---:|
| Flash 9B · RTX 3070 Ti · 290W | 63.45s | 62.61s | 79°C → 59°C |
| Flash 9B · RTX 3090 · 400W | 41.56s | 40.23s | 75°C → 74°C |
| Full 27B · RTX 3090 · 400W | 119.31s | 119.16s | 76°C → 75°C |

The 3070 Ti temperature fell by 20°C; measured concurrent-time changes were
modest (about 1.3% for that card, 3.2% for Flash/3090, and 0.1% for Full/3090).
No thermal-slowdown flags coincided with nonzero sampled GPU utilization in
either session. These are two-pass means from separate sessions, not confidence
intervals or proof that cooling alone caused the small timing changes. Original
services were restored afterwards.
