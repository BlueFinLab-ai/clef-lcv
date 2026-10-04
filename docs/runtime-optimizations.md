# Runtime optimizations

Both Full and Flash accept up to 16 images per request and use compact schema
labels and one image preprocessing pass. Combined context and VRAM limits still
apply; 16 high-fidelity images may exceed those limits.
Question instructions, answer options, and the official joint decision head are
retained. Overlength inputs return 413 without truncating context.

## Context budget discovery

Clients should query `GET /v1/models` and read `data[].max_input_tokens` for text.
The calibrated NF4 builds now compute this value from the selected GPU's idle
memory. It is a deployment extension; the 262,144-position architectural ceiling
is reported separately and does not imply that the GPU can serve that many tokens.
`/health` and admission use the same estimator. Overlength requests return HTTP
413 without truncation, with the accepted limit and actual encoded token count.

`context_limit.mode` is `memory_estimate` for calibrated builds. Its
`computed_max_input_tokens` equals `max_input_tokens`; the legacy configured cap
is shown for transparency with `configured_cap_applied=false`. Unsupported kernel,
model, or chunk configurations fall back to `configured` mode and explain why.
`--max-length N` selects fixed mode explicitly. Environment deployments can use
`CLEF_CONTEXT_LIMIT_MODE=fixed` and `CLEF_MAX_LENGTH=N` for the same behavior.

The calculation includes device free memory, PyTorch reserved-but-unallocated
blocks, and GPU prefix/image entries that long requests evict. It subtracts a
512 MiB safety reserve, then projects memory using measured workspace anchors
and model configuration. Flash KV is 32 KiB/token plus 8 KiB/token of retained
hidden vectors; Full KV is 64 KiB/token plus 10 KiB/token of retained vectors.
Temporary expanded GQA K/V, a layer's concatenation copy, and token bookkeeping
are included in the growth term with 25% additional slack. Anchors are 2,000 MiB
above weights at 24,576 tokens for Flash and 2,210 MiB at 16,384 for Full, rounded
above measured peaks. Capacity is rounded down to 4K tokens and bounded by the
architecture. If the measured initial-chunk workspace cannot fit, it advertises
zero instead of guessing a smaller safe policy.

For contexts above the reference, the projection is:
`workspace = reference_workspace + (tokens - reference_tokens) * growth_bytes_per_token`.
Below the reference it conservatively retains the workspace floor. Admission
samples idle memory under the inference lock. Discovery while busy returns the
last idle estimate (`context_snapshot_status=last_idle_busy`). Larger observed
chunked workspace raises the projection; an unexpected text or automatic-image
OOM reduces its runtime ceiling. Cache occupancy does not permanently reduce the long-text budget.

For requests containing images, use `max_input_tokens_with_images`. It includes
all text, image, question, option and template tokens. Full defaults to
`CLEF_MAX_IMAGE_LENGTH=auto`. With calibrated memory estimation and
`CLEF_IMAGE_PREFILL=1`, images use the computed budget and report
`image_limit_basis=computed_memory_estimate`. The legacy `CLEF_MAX_LENGTH` does
not clamp that budget. Measured image workspace and image OOMs feed the same
conservative projection and runtime backoff as text.

An explicit numeric `CLEF_MAX_IMAGE_LENGTH` remains an operator cap, bounded by
the current text admission limit. It is no longer silently clamped by the legacy
fallback `CLEF_MAX_LENGTH`. Flash retains its numeric default. Automatic image
mode falls back to `CLEF_MAX_LENGTH` when chunking or calibrated estimation is
unavailable, including explicit fixed mode. Check discovery after changing a
prefill policy; automatic mode never assumes an uncalibrated image capacity.

This remains an estimate, not a GPU reservation. External memory changes,
fragmentation, image layouts, and large question/answer schemas can cause an
otherwise accepted request to return 503 after the existing cache fallback.
Capacity does not establish answer accuracy at long context; the recorded
[accuracy artifact](long-context-accuracy-artifact.md) remains unresolved.

The browser Image Scaling default is Compact: fit within 256 × 256.
Original and display-resolution frames through 4K UHD are also available.
The browser downscales while preserving aspect ratio and source file format.
Original and already-small inputs bypass encoding. Native processor resizing
remains enabled only for patch alignment, with `min_pixels: 1024` and
`max_pixels: 20000000` to avoid a second fidelity budget. The upload and context
limits still apply. See [image scaling](image-scaling.md) for details.
Direct API calls use native processor defaults or explicit `media_kwargs`;
`image_fidelity` is now an optional deprecated server-resizing extension. See
[client resizing and API compatibility](client-image-resizing.md).

## Branching exact-prefix reuse

Prefix reuse is enabled by default with the memory-efficient attention backend.
It stores full-attention KV, linear-attention recurrent/convolution state, and
prefix hidden states. It recomputes each question schema and the joint decision
head. It does not cache answers. New or reordered questions can reuse the exact
same image/context. Requests resume from the longest saved matching checkpoint.
Changed text reuses only the identical leading portion; changed images, image
order, fidelity or pooling invalidate checkpoints that include those images,
while identical text before the images can still be reused. Model identity,
complete prefix token IDs and media identity are included;
entries do not survive process restarts. No pixels or states are written to disk.

The API accepts optional `context`, placed before images and changing `state`.
The browser exposes it as **Reusable context**. Its end and the end of the media
region are explicit checkpoints. Without this field, the engine discovers the
exact common leading token span of incoming requests and retained entries, and
builds a checkpoint there for later reuse. The first divergent request may need
to rebuild that shared section; later requests can hit it. Prefixes of fewer
than 128 tokens are not automatically discovered. The engine keeps bounded CPU
token metadata for recent inputs so a repeated branch can be promoted to a GPU
checkpoint. Novel unrelated inputs seed prefix discovery; unique suffixes after
a shared guide run together with their questions in one language forward.
This avoids an extra forward merely to save a branch that may never recur.
Explicit context boundaries can be shorter. State endpoints within 128 tokens
of an existing checkpoint are not separately retained. Checkpoints never split
the media region. Fixed interval checkpointing is disabled by default.

By default the cache retains up to 32 checkpoints with an automatic VRAM budget.
Both HTTP builds default to elastic caching: the automatic target uses all
device memory after accounting for model/other allocations, active request
workspace and adaptive extra headroom. Idle/short calls keep 256 MiB; calibrated
incremental prefill with positive projected workspace keeps at least 512 MiB. Idle requests release the workspace reservation; retained
tensors remain until memory or entry limits require eviction. Long requests
use calibrated projected workspace; short requests use observed workspace.
Already allocated working tensors are deducted from transient snapshot headroom.
Active branch clones are part of workspace, while retained copies are extra
allocations. Before large prefill, an eviction also returns unused allocator
segments to CUDA with `empty_cache()`. This is selective: calls without an
associated eviction do not flush the allocator. This target is a retention
budget, not a hard upper bound on request memory. Each
service stays on its one configured GPU. A numeric MiB setting
overrides automatic sizing. Checkpoints grow on demand; the cache does not
preallocate or fill idle memory with unused tensors. Least recently used entries
are evicted before retaining new ones or when request headroom becomes tight.
Actual allocator-free memory includes unused blocks reserved by PyTorch.

Mutable KV, recurrent and convolution states are cloned before each branch.
The short path shares immutable hidden-state chunks between related checkpoints
and counts their storage once. The long path copies only the completed prefix
from its full-request hidden buffer, so retaining a checkpoint does not pin
unused suffix storage. KV/recurrent snapshots remain independent copies; this is
not paged attention or zero-copy KV sharing. Entries are not
retained when memory is tight. There is no fixed token-count cutoff for caching.
Memory availability includes unused blocks reserved by PyTorch, so a previous large request does not prevent
subsequent small requests from retaining a prefix. An OOM
clears the cache and retries once without caching. First requests pay cache build
costs. Similar wording alone is not a cache hit. Chunked NF4 processing can
shift probabilities slightly; near-tied choices can change. Disable caching
when comparing against the unchunked numerical reference.

API requests can set `prefix_cache: false`. `usage.prefix_cache` reports `hit`,
`miss`, `not_retained`, `disabled`, or `memory_fallback`; `prefix_build_ms` and
`prefix_tokens` describe processing. `reused_prefix_tokens` is the actual saved
work; `prefix_tokens` is the total state-prefix length, not the hit length.
`prefix_checkpoints_saved` counts new snapshots, and `vision_prefix_reused`
reports skipped GPU vision encoding. CPU image preprocessing can independently
reuse processed pixels. Long requests also report `cache_memory_budget_mib` and
`cache_limit_reason`; `request_memory_budget` means the projected request leaves
no retention budget. `checkpoint_exceeds_memory_budget` means the snapshot is
larger than the remaining pool; `insufficient_snapshot_headroom` means copying
it would violate transient free-memory checks.
`/health` exposes occupancy, budget, observed workspace, evictions and hit counters.
Elastic cache stats also expose `phase`, `request_workspace_mib`, `reserve_mib`,
`prefill_reserve_mib`, `effective_reserve_mib` and retention rejection counters.
Usage includes `cache_prepare_ms` (the complete preparation cost),
`cache_evicted_mib` (live retained tensors removed), `cache_headroom_mib` and
`cache_allocator_release_ms` (included in preparation time). The evicted byte
count is distinct from unused allocator memory returned to CUDA. Idle budget and the active budget reported in usage can
differ. A larger idle pool does not guarantee retention at the maximum context:
the live request and its retained snapshot must fit at the same time.
Server timing includes decoding, preprocessing, GPU work and answer formatting;
browser timing additionally includes client image preparation, JSON serialization,
upload/network time and request parsing. The GUI reports client preparation and
encoded request size separately.

## Optional 2×2 pooling

Select **Enable 2×2 image pooling** in the browser, or set `image_pooling: true`
in the API request. Default is false. Each image is encoded at the selected
fidelity, then its merged spatial features are averaged to a smaller grid before
language processing. Placeholder tokens, schema spans, multimodal token types,
and positional indices are adjusted. Multiple images retain their order.

This is experimental, untrained spatial pooling. Small text and fine visual detail
may be lost. Vision encoder computation is unchanged; language computation uses
roughly one quarter of the image tokens. `usage.unpooled_input_tokens` and
`usage.input_tokens` show the reduction. Pooling can be combined with prefix reuse.

## Incremental text prefill

Both HTTP builds use an initial 8,192-token chunk followed by 4,096-token
continuations for text-only encoded requests longer than the initial chunk size.
This is internal processing of one complete HTTP request. It does not stream
additional context from the client. Short requests retain the existing prefix
cache behavior. Long image requests also use this policy when image prefill is
enabled; otherwise they retain the single-pass multimodal path.

The language model carries one mutable request-local hybrid attention state
between chunks. All token outputs remain in a preallocated buffer, and the full
unchanged native head runs once at the end. Working KV/recurrent state is released
before the head. Memory-budgeted persistent checkpoints can now seed this path;
mutable hybrid state is cloned per request. No truncation, offloading or sharding
is used. Chunking reduces temporary workspace; working KV and hidden
outputs still grow with total input length. The recorded [accuracy artifact](long-context-accuracy-artifact.md)
remains an independent limitation.

Set `CLEF_PREFILL_INITIAL_CHUNK_TOKENS` and `CLEF_PREFILL_CHUNK_TOKENS` to change the
sizes; a continuation size of `0` disables this path. Smaller 1K continuations are
available for capacity experiments. The deployed 4K policy favors the measured
speed improvement at 24K. The calibrated 8K/4K policy now uses computed text
limits. Other chunk policies retain configured limits until separately calibrated.
Each service remains on one GPU.

Usage reports `prefill_mode` (`chunked`, `single_pass`, or `prefix_cache`). Chunked
requests additionally report `prefill_chunks`, `prefill_initial_chunk_tokens`,
`prefill_chunk_tokens`, `language_ms`, and `head_ms`. Their `prefix_cache` status is
the actual hit/miss/disabled state; CPU token preprocessing reuse remains available.
`/health` reports the configured prefill sizes and image-prefill setting.

Before long text or image processing, the projected workspace trims cache
occupancy. A sufficiently large request can evict all GPU entries; a smaller
request can retain them. Long observed workspace is tracked separately as
`observed_chunked_workspace_mib`, so a maximum-length call does not permanently
disable retention on subsequent smaller calls. The existing `observed_workspace_mib`
continues to reserve workspace for non-chunked requests that can coexist with
retained prefix and image-feature entries.

CLI options include `--prefill-initial-chunk-tokens`, `--prefill-chunk-tokens`,
`--max-image-length`, and `--context-reserve-mib`. For computed text admission on
the calibrated Flash build with an 8 GB 3070 Ti:

```sh
.venv/flash/bin/python scripts/clef.py serve flash --gpu 0 \
  --max-image-length 8192 \
  --prefill-initial-chunk-tokens 8192 --prefill-chunk-tokens 4096
```

`scripts/test_chunked_prefill_api.py` captures a synthetic reference before an
update and checks native answer/probability agreement afterward, long text at the
advertised limit, image bypass, both overlength limits, and short-cache recovery.
It accepts `--url`, `--project`, `--checkpoint`, `--reference-tokens`, `--reference`,
and `--phase baseline` or `--phase verify --output FILE`. Run in a profile's Python
environment; it loads only the local tokenizer/CPU helpers and calls the already
running service rather than loading a second GPU model.

## Incremental image prefill

Set `CLEF_IMAGE_PREFILL=1` or pass `--image-prefill` to enable chunking for mixed
text/image requests longer than the first chunk. The portable default is off;
the validated live Full and Flash deployments explicitly enable it. Short image
requests retain the existing native/prefix-cache path.

The new path leaves processed pixel tensors on CPU and transfers one complete
image at a time for vision encoding. Original ordering is retained. Optional 2x2
pooling follows vision encoding, using the same grid and feature transformation
as the existing path. Spatial rotary positions are computed for the complete
mixed sequence, then sliced with each language chunk. Only the current chunk's
text embeddings are allocated, and image features are inserted at their matching
placeholder positions. A boundary can cross an image-token block without losing
features or resetting its spatial coordinates. KV/recurrent state continues
across chunks; all token outputs go to the unchanged native decision head.

Long image requests can retain language checkpoints and independent image
features within the shared memory budget. A checkpoint covering all images skips
vision and pixel transfer; otherwise missing features are encoded one image at a
time. Reordering images invalidates the language prefix but can reuse raw features.
CPU preprocessing reuse remains available. See [long-request cache validation](long-request-cache.md). The new path reports `multimodal_prefill`, `vision_ms`,
`vision_batch_images`, and `image_chunk_splits` in addition to standard prefill
usage. Vision features are still native FP16/BF16 activations; weights are unchanged.

Both Flash deployments retain a 24,576-token image ceiling. Full uses the computed
budget (45,056 at validation) after [controlled quality tests](full-context-quality.md)
showed no wrong answers through that length, including sixteen images. The old
16K cap has been removed. A 1-point probability change alone is not a quality ceiling.
When reverting the 3070 Ti to single-pass image processing, also restore its image
ceiling to 8,192. Turning off chunking alone does not change an operator's cap.

For the validated Flash configuration with automatic text discovery:

```sh
CLEF_MAX_LENGTH=24576 CLEF_MAX_IMAGE_LENGTH=24576 CLEF_IMAGE_PREFILL=1 \
  .venv/flash/bin/python scripts/clef.py serve flash --gpu 0
```

For Full with automatic image admission:

```sh
CLEF_IMAGE_PREFILL=1 CLEF_MAX_IMAGE_LENGTH=auto \
  .venv/full/bin/python scripts/clef.py serve full --gpu 0
```

These environment recipes leave text-limit mode at `auto`. The CLI `--max-length`
option instead explicitly chooses fixed mode. Recheck `/v1/models` on your GPU.

## Controls

| Setting | Default | Purpose |
|---|---|---|
| `CLEF_IMAGE_PREFILL` | `0` | Enable incremental long-image prefill; explicitly enabled on the validated services |
| `CLEF_CONTEXT_LIMIT_MODE` | `auto` | Compute the text limit on calibrated builds; `fixed` enforces the legacy cap |
| `CLEF_CONTEXT_RESERVE_MIB` | `512` | Extra memory reserve for the context estimate, separate from cache retention settings |
| `CLEF_MAX_LENGTH` | profile fallback | Fixed/fallback text cap; does not clamp the calibrated auto estimate |
| `CLEF_MAX_IMAGE_LENGTH` | `auto` for Full; legacy cap for Flash | `auto` follows calibrated memory with image chunking; a positive integer sets an explicit image cap |
| `CLEF_PREFILL_INITIAL_CHUNK_TOKENS` | `8192` | First long-text prefill chunk; also the threshold for using the incremental path |
| `CLEF_PREFILL_CHUNK_TOKENS` | `4096` | Continuation size; `0` disables incremental text prefill |
| `CLEF_PREFIX_CACHE` | `1` | `0` disables prefix reuse process-wide |
| `CLEF_PREFIX_CACHE_MIB` | `auto` | Automatic budget, or an explicit maximum in MiB |
| `CLEF_PREFIX_CACHE_ENTRIES` | `32` | Maximum retained checkpoints |
| `CLEF_PREFIX_CACHE_RESERVE_MIB` | `256` | Extra idle/short-call headroom; added to workspace under elastic accounting |
| `CLEF_PREFIX_CACHE_PREFILL_RESERVE_MIB` | `512` | Extra headroom for positive projected prefill workspace; never below the configured minimum reserve |
| `CLEF_PREFIX_CACHE_GPU_UTILIZATION` | `1.0` | Automatic memory target; model, other allocations and reserves are subtracted |
| `CLEF_PREFIX_CACHE_ELASTIC` | `1` | Both HTTP builds: release idle workspace and account transient copies without double reservation; `0` selects legacy accounting |
| `CLEF_PREFIX_CHECKPOINT_TOKENS` | `0` | Optional fixed checkpoint spacing; 0 uses explicit and discovered boundaries |
| `CLEF_IMAGE_POOLING` | `0` | `1` enables pooling by default; requests can override |
| `--image-pooling` | off | CLI equivalent of enabling pooling by default |
| `--no-prefix-cache` | off | CLI equivalent of disabling prefix reuse |

These controls apply to `scripts/clef.py serve full` and `serve flash`.
Other attention backends bypass prefix reuse; the efficient backend was the best
simple combination in the experiments. Quantization, fused projections and
compilation were not changed because they did not improve measured language time.

## Verification

Run `scripts/test_optimizations.py full` or `flash` in the GPU runtime with
`CLEF_DATA_DIR` and `CUDA_VISIBLE_DEVICES` set. It compares fresh, cold-cache,
warm-cache and changed-question inference for all three answer types, text,
single-image inputs, four-image pooling and 1024 fidelity. It checks selected
options and probability differences. The synthetic inputs require no private data.

`scripts/test_prefix_index.py` checks exact ancestry, media isolation, immutable
states, eviction and budget arithmetic on CPU. `scripts/test_prefix_branching.py`
validates text and image branches, changed questions, pooling, automatic boundary
discovery and pressure eviction on GPU. Probability differences must stay below
3.5 percentage points; choice changes are accepted only within a near tie bounded
by twice that request's measured probability difference. The separate 100-email
benchmark requires all choices to match each model's uncached baseline.

## Independent input reuse

Full and Flash now support a separate process-local input cache, enabled by default. Exact text fragments reuse token IDs. Each exact image upload is validated, decoded, EXIF-normalized and processed once per processor/fidelity combination; later requests reuse the processed CPU tensors. Image tensors are assembled in the requested order and preserve the original tokenizer/schema layout.

The GPU cache holds each image’s unpooled vision features in the model’s native activation precision. A reused image can appear under a different leading context, at a different position, alongside new images, or with 2×2 pooling toggled. The short path batches missing images; the incremental path processes misses
one image at a time. Cache keys distinguish these batch policies. Duplicate images
are encoded once when their features fit the retention budget. Pooling is applied after retrieval, so the same raw features support either setting. Changing fidelity requires new image processing and features. The language forward and native decision head still run unless an exact language-prefix checkpoint also matches.

CPU image preprocessing is bounded to 256 MiB / 128 entries; text tokens to 8 MiB / 1,024 entries. Independent features are capped at 256 MiB / 64 entries, **inside** the existing shared adaptive GPU cache budget. Language checkpoints are evicted before the smaller independent features under GPU pressure. Cache entries grow on demand, and OOM retry clears GPU caches and retries with both prefix and feature reuse disabled. Long requests use their projected workspace to decide retention; the previous 8K GPU-cache bypass is removed. CPU input reuse remains available.

Set API `input_cache: false` to bypass all new layers. Set `prefix_cache: false` independently to measure image reuse with language-prefix reuse disabled. An entirely uncached reference needs both flags false. Global settings are `CLEF_INPUT_CACHE`, `CLEF_PREPROCESS_CACHE_MIB`, `CLEF_PREPROCESS_CACHE_ENTRIES`, `CLEF_TOKEN_CACHE_MIB`, `CLEF_TOKEN_CACHE_ENTRIES`, `CLEF_IMAGE_FEATURE_CACHE_MIB` and `CLEF_IMAGE_FEATURE_CACHE_ENTRIES`; defaults are in the deployment environment example. Zero byte/entry limits disable retention for that layer. The CLI retains its existing preprocessing path; these new caches are integrated into the HTTP adapters.

Usage reports preprocessing milliseconds, processed-image hits/misses, token hits/misses and independent-feature hits/misses. A language-prefix hit that already includes images skips vision entirely; in that case independent-feature counts are zero and `vision_prefix_reused` is true. Health exposes current CPU occupancy and GPU feature occupancy/hit counts. CPU caching still requires receiving and hashing the upload; it does not reduce network transfer. The short path still transfers pixel tensors during collation. The incremental path leaves pixels on CPU and transfers only vision misses.

Keys isolate processor/tokenizer instances, exact source bytes, processing parameters, model/vision instance, image grid, device and activation dtype. Serving objects and weights are immutable for their process lifetime; restart clears all caches after changing model/processor configuration. Re-encoding identical pixels in another JPEG produces a miss. No source files, answers, or cache tensors are written to disk. These layers support text/JSON and still images; extraction for PDFs, audio or other formats is outside the current HTTP API. They do not provide semantic similarity matching.

NF4 vision matrix kernels can produce slightly different values when only a subset of a previous image batch is computed or when duplicates reuse features from a smaller batch. Processor pixels and token layout are exact; decision probabilities are not guaranteed bit-identical. See the input-cache benchmark report for measured deltas and selected-option agreement. Disable input reuse for the original batched reference.


## Request queue

Both HTTP builds use one FIFO model worker, running CPU preprocessing and GPU
inference serially on their configured single GPU. Concurrent callers await
responses asynchronously without occupying the inference thread. FIFO order is
by submission after the complete body and schema validation, not the opening
order of slow-upload TCP connections. This implementation buffers requests; it
does not combine records into GPU batches or overlap processor access.

Default bounds are 64 waiting jobs plus one active job, five minutes maximum
queue wait, 256 MiB total admitted wire payload and 224 MiB per request body.
Admission count and byte limits also cover requests being received/parsed and
responses still completing. The middleware checks actual chunks as well as
Content-Length before JSON parsing; base64 image bytes count toward the budget.
Wire bytes bound retained input volume, not process RSS: JSON strings, decode
copies and processor tensors require additional RAM. Only the active worker
creates decoded/preprocessed media. Existing input-cache capacities still apply.
A 224 MiB body can accommodate sixteen 10 MiB images after base64 expansion
plus typical text; these remain subject to native per-image and context limits.

Queue count or aggregate payload exhaustion returns 429 with `Retry-After: 1`
and `error: queue_full`; an oversized wire body returns 413 with
`error: request_body_too_large`. Queue wait expiry returns 504 with
`error: queue_timeout`. Capacity/token admission is recomputed inside the worker,
after the preceding request finishes, using reclaimable caches and normal
workspace rules. Genuine image/token overlength remains 413, including the exact
encoded token count and accepted limit for a token violation. Queueing cannot
make an oversized individual request fit. Worker errors do not stop subsequent
jobs. A closed/unavailable queue returns 503 with `queue_shutdown`.

A disconnected or cancelled pending call is removed and its budget released.
An already active CUDA call cannot safely be interrupted: it finishes, then its
result is discarded if the caller has left. Its payload remains charged until
completion. The lifespan close rejects pending work and waits for the active
worker. Abrupt process termination loses queued jobs; this is not a durable job
API and the client receives no guarantee of delivery after disconnection.

Usage adds `queue_wait_ms` and `server_total_ms`; existing `latency_ms` still means
processing time, including preprocessing and cache preparation, excluding queue
wait. Total is measured after endpoint submission and excludes HTTP upload,
JSON/schema parsing and response transmission. Processed JSON error responses
such as token 413 carry `X-Clef-Queue-Wait-Ms` and `X-Clef-Server-Time-Ms` headers.
The GUI displays the wait in the timing description, uses a ten-minute response
timeout, and shows detailed 413 counts. External callers should choose timeouts
longer than queue wait plus expected inference duration and avoid automatic
retries that can duplicate active work.

`/health.queue` exposes active/waiting/admitted counts, admitted wire MiB,
limits, accepting state and completed/failed/cancelled/timed-out/rejected counts.
GPU/cache discovery uses the previous idle snapshot while inference is running;
health does not iterate mutable cache dictionaries concurrently with the worker.

| Control | Default | Meaning |
|---|---:|---|
| `CLEF_QUEUE_MAX_WAITING` | `64` | Maximum waiting inference jobs; admission also bounds in-flight parsing |
| `CLEF_QUEUE_WAIT_SECONDS` | `300` | Maximum wait before execution; active inference has no queue deadline |
| `CLEF_QUEUE_PAYLOAD_MIB` | `256` | Shared wire-byte budget for admitted requests, including active calls |
| `CLEF_REQUEST_BODY_MIB` | `224` | Maximum wire body for one decision call |

Run `scripts/test_request_queue.py` for CPU/ASGI checks of FIFO, one worker,
bounds, streaming bodies, deadlines, exceptions, disconnects and shutdown.
`scripts/test_queue_api.py` checks real-model concurrent mixed text/JPEG calls,
cache reuse, queued token 413, actual TCP disconnection and context stability.
It generates synthetic pictures and saves no request bodies.
