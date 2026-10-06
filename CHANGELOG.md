# Changelog

Dated summaries of notable changes. Measurements and test details are in the
[validation record](docs/validation.md) and the linked documents.

## 2026-10-06

### Image caching
- When a request's later images change, are removed or are added, Flash and Full
  reuse the work for the unchanged leading images. Cache checkpoints are kept
  after each complete image, keyed by the exact images before it, in the existing
  GPU and RAM tiers.
- Restored images skip vision encoding; responses report `vision_images_reused`.
  For five 1536px images with the last one replaced, requests were 2.5–2.9 times
  faster than with vision caching alone.

### Startup
- Flash and Full warm a short text request and a synthetic 256px image on the
  serving GPU worker before HTTP readiness. Warmup failures stop startup;
  `/health.startup_warmup` reports timings. Use `--no-warmup` or `CLEF_WARMUP=0`
  to opt out. Synthetic requests are excluded from queue counters and caches.

### Benchmarks
- Reran the 100-email benchmark on the October 5 runtime: Flash on the RTX 3070 Ti
  and RTX 3090, and Full on the RTX 3090. With four clients after restoring
  cooling, 100 emails took 62.6 s, 40.2 s and 119.2 s respectively. All 2,600
  measured requests succeeded. See
  [queue batching results](benchmarks/nvidia-batching/README.md).

## 2026-10-05

### Caching and long context
- Exact-prefix cache indexed with a compressed radix tree, following SGLang's
  RadixAttention approach.
- Checkpoints evicted from GPU memory move to an automatically sized system-RAM
  tier (25% of available memory kept free, least-frequently-used eviction) instead
  of being dropped. Processed images share the same RAM allowance. There is no
  disk tier.
- Memory-adaptive context routing for Flash and Full: GPU native, then GPU tiled,
  then CPU-streamed attention, chosen per request. 131K-token requests were
  tested on both models. See [adaptive context](benchmarks/adaptive-context/README.md).

### Serving
- Opportunistic batching of small text requests, two at a time by default on SM86.
- Optional API keys, an automatic portal session, and a demo mode that masks
  server addresses in generated code.
- Runtime-only Docker image and prebuilt Triton host helpers.
- Experimental AMD RX 580 (ROCm) build. See [the ROCm experiment](experimental/rocm/README.md).

### Portal
- Reusable context and request input stay a steady size with long text, showing
  their size with Expand and Clear.
- Fixed-height timing area with Server, Queue wait, Input tokens, Cached input
  and Peak GPU tiles; JSON response tab.

## 2026-10-04

### Portal
- Side-by-side editor and response panes that scroll independently. Each answer
  sits beside its timing, shows its confidence change since the previous run of
  the same question, and dims when its question is edited.
- Answer options are entered as tags; the answer type is a three-way toggle.
- Run with Cmd/Ctrl+Enter or optional auto-run; mark answers correct or wrong.
- A recent-run chart replaces the run history table.
- Copyable curl and Python examples built from the current editor, using the
  server's own address and model ID.
- Image Scaling presets from Compact (256 × 256) to Original, preserving the
  source PNG, JPEG or WebP format. See [image scaling](docs/image-scaling.md).

### Service and packaging
- One implementation, `clef_service`, now serves both Flash and Full. It detects
  the selected GPU before loading and chooses the tested kernel policy.
- Docker image and Compose file with a persistent model and compiler cache.
  The first start downloads the pinned release and prepares NF4 weights; later
  starts reuse them. See [unified service and Docker](docs/unified-service.md).
- Preparation quantizes one tensor at a time, so Flash can be prepared on an
  8 GB GPU; the dense intermediate checkpoint is no longer needed.
- Build support for A100 (SM80), RTX 4090 (SM89), H100 (SM90) and RTX 5090
  (SM120) in addition to the tested SM75 and SM86 cards. See
  [GPU support](docs/gpu-support.md).
- Bounded FIFO request queue: one inference worker per GPU, up to 64 waiting
  requests, 429 on overload and 504 on an expired wait.
- Elastic GPU cache headroom: retention uses remaining VRAM, grows on demand and
  shrinks before larger requests. Cache accounting counts tensor fields only.
  See [long-request cache](docs/long-request-cache.md) and
  [cache accounting](docs/cache-accounting.md).
- Optional optimized linear-attention kernels for Flash. The RTX 3070 Ti uses
  FLA throughout; the RTX 2080 Ti uses an adaptive 512-token crossover. Frozen
  100-email repeat wall time fell from 100.17 to 75.54 s on the 3070 Ti and from
  85.52 to 81.69 s on the 2080 Ti, with all category choices preserved. See
  [optimized kernels](docs/optimized-kernels.md).

### Benchmarks
- Clef Flash measured on the RTX 2080 Ti (65.22 s serial, after restoring
  cooling) and the 400 W RTX 3090 (43.41 s), alongside the RTX 3070 Ti (59.98 s).
  All three gave identical category choices. See
  [Clef Flash on three GPUs](benchmarks/provider-comparison/README.md#clef-flash-on-three-gpus).
- Public 100-email synthetic benchmark and runner. See
  [the email sample](benchmarks/email-sample/README.md).
- Comparison runner for Clef, TypeSafe JEV and other SystemOne-compatible
  services. See [provider comparison](benchmarks/provider-comparison/README.md).

## 2026-10-03

- Saved working NF4 deployments of Clef 27B (Full) and Clef Flash 9B.
- Context limits computed per GPU from available memory instead of fixed caps.
  `/v1/models` reports the current limit. See
  [context budget discovery](docs/runtime-optimizations.md#context-budget-discovery).
- Incremental text and image prefill (8K initial, 4K continuation chunks). Full's
  image budget follows the computed limit (45,056 tokens at validation); Flash
  keeps a validated 24,576-token image budget. See
  [incremental prefill](docs/multimodal-prefill.md).
- Branching exact-prefix cache and independent preprocessing, token and image
  feature reuse. Reordered or partly changed image groups can reuse individual
  images. See [prefix cache benchmarks](docs/prefix-cache-benchmarks.md) and
  [input cache benchmarks](docs/input-cache-benchmarks.md).
- The 8,192-token GPU-cache cutoff was removed for long requests.
- Client-side image resizing with native processor `media_kwargs`. See
  [client image resizing](docs/client-image-resizing.md).
- Up to 16 images per request.
- Full's optimized runtime completed the repeated email benchmark about 26%
  faster than its fallback, keeping all 200 category decisions.
- Experimental long-context Flash tests found inconsistent fact retrieval even
  without chunking. See the [accuracy artifact](docs/long-context-accuracy-artifact.md).
