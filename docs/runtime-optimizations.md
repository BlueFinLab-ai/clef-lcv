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

CPU image preprocessing now uses the shared automatic RAM policy described below; text tokens remain bounded to 8 MiB / 1,024 entries. Independent features are capped at 256 MiB / 64 entries, **inside** the existing shared adaptive GPU cache budget. Language checkpoints are evicted before the smaller independent features under GPU pressure. Cache entries grow on demand, and OOM retry clears GPU caches and retries with both prefix and feature reuse disabled. Long requests use their projected workspace to decide retention; the previous 8K GPU-cache bypass is removed. CPU input reuse remains available.

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

`gpu_time_ms` uses device events around each inference attempt, including a
failed attempt followed by an OOM retry. CPU input preparation, admission and
cache preparation before inference are outside this window. It is elapsed
device-stream time, including host launch gaps, inference cache work and
transfers, rather than a sum of kernel execution times. Batched members share
the batch window; interleaved work is included in its parent's window. The
API adds `cpu_time_ms` and `cpu_time_estimated:true`: elapsed handler processing
minus the GPU window, plus preprocessing done by queued lookahead, bounded by
the request's server elapsed time minus GPU time. Inline preprocessing is
already in the handler window and is not added twice. This is an estimate of
elapsed non-inference work, not CPU utilization; in-model cache work and launch
gaps remain in the GPU window. Shared batch processing windows are attributed
to each member, not divided by batch size.

The portal's CPU Time also includes browser image preparation. Overhead Wait
is `max(0, browser_total_ms - gpu_time_ms - displayed_cpu_time_ms)`, covering
upload, proxy/network transit, remaining queue wait and unmeasured work. Queue
wait can overlap lookahead CPU preparation; it must not be added separately.
An older GPU-timed server uses an equivalent estimate from existing usage
fields. Without GPU timing, the portal labels `latency_ms` Processing and
shows browser preparation only in CPU Time, avoiding double-counting server CPU.

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

## Experimental ROCm settings

The RX 580 image has a separately validated math-attention caching policy,
repeated complete-prefix promotion, query-tiled vision attention, offline
rocBLAS replay, and smaller prefill chunks above 4K. See the
[ROCm settings and measured limits](../experimental/rocm/README.md). These flags
are opt-in in the shared runtime and are enabled by the experimental Dockerfile.


## Opportunistic CUDA text batching

The worker can take compatible consecutive text requests already in the inbound
queue. There is no collection timer: an isolated request starts immediately.
The oldest request anchors the group; images and other incompatible requests
form a FIFO barrier and are processed through the existing single-request path.
The validated SM86/optimized-kernel builds default to cap two for both model
profiles. Other CUDA families stay at one until explicitly enabled.
Maximum batch size is selected with `--batch-size 1|2|4` or
`CLEF_BATCH_MAX_SIZE`. Size one disables grouping. Actual groups can be smaller.

CPU encoding supplies actual token counts. Admission limits padded tokens,
record length and padding ratio, then estimates KV, hybrid recurrent state,
hidden/attention buffers and temporary workspace against reclaimable memory
plus a reserve. The estimate learns from measured batch peaks. Aggregate OOM
splits a group into smaller groups and finally independent calls; it does not
lower the single-request context ceiling. Invalid and oversized members retain
individual 400/413 responses without failing neighbours.

Each batched row has independent KV, convolution and recurrent state. Exact
common prefix checkpoints can be shared read-only, then cloned for execution.
Padded end states are not retained as per-request checkpoints. Requests with a
better individual cached prefix use the single path, preserving existing reuse.
Exact repeated states can use that path to promote their own checkpoints.
This works with generic Clef text/state/context and typed questions, not just emails.

Images, optional pooling, long prompts and their incremental prefill remain
single-request operations in this initial version. RX 580 batching stays off.
`/health` reports queue grouping and actual GPU batch counts separately. Responses
report `usage.batch_size`, padding, worker wait and any fallback reason. Batch
latency is the group's work duration, not that duration divided by row count.


## CPU-backed prefix checkpoints

The service retains evicted language checkpoints in a bounded, process-local
CPU RAM tier. GPU retention remains elastic. The CPU index searches both tiers
for the longest exact matching saved checkpoint, including model, token ancestry,
image identity, scaling and pooling. Semantic similarity is not a cache hit.
A checkpoint contains full-attention KV, linear recurrent/convolution state, and
final hidden vectors required by Clef's unchanged decision head.

`CLEF_PREFIX_HOST_CACHE_MIB` defaults to `auto` for both Flash and Full. The
cache grows on demand; it does not allocate or reserve its full budget upfront.
The budget is recomputed on request preparation, restoration, retention and
completion from Linux `MemAvailable`, constrained by remaining memory in visible
cgroup v1/v2 ancestors (including v2 `memory.high`). Swap is never counted.
The default leaves 25% of allocatable RAM free:

```text
available = min(system MemAvailable, visible cgroup headroom)
capacity = available + this service's current CPU prefix tensor bytes
cache limit = 0.75 × capacity
```

Adding back this tier's own allocations keeps its budget stable as it fills;
RAM used by other services and non-cache workloads reduces the budget. This is
a per-process live budget, not a reservation of 75% of total installed RAM for
every service. Under pressure the worker trims retained snapshots before CPU
copies or request processing. Idle caches are trimmed when the next request
runs; this is not an OS memory reservation or a guarantee against unrelated
workloads exhausting RAM between samples. No memory reading means no automatic
cache admission. Only cgroup ancestors visible inside the container can be
inspected; use an explicit container memory limit when isolation is required.

Set `CLEF_PREFIX_HOST_CACHE_MIB` to a positive MiB value for a fixed maximum
(still constrained by sampled headroom), or `0` to disable the RAM tier.
`CLEF_PREFIX_HOST_CACHE_RESERVE_FRACTION` defaults to `0.25` and accepts values
from 0 inclusive to 1 exclusive. `CLEF_PREFIX_HOST_CACHE_ENTRIES=auto` removes
the default entry-count cap; a positive integer adds a cap, and `0` disables
the tier. Limits apply to retained tensor storage; token metadata and temporary
copy buffers add overhead covered by the free-memory margin.

RAM eviction uses least-frequently-used (LFU) checkpoints; least recent reuse
breaks ties. Newly built checkpoints start at zero uses, actual GPU/CPU prefix
reuse increments usage, and GPU use counts survive offload to RAM. Metadata
lookups and duplicate snapshot writes do not count as reuse. Usage is cumulative
for a retained checkpoint's lifetime, without aging. GPU eviction continues to
use its existing elastic/LRU policy. RAM entries vanish on process restart or
cache clear. There is no disk storage or new runtime compiler. Image-feature
caching and CPU preprocessing caches retain their separate policies. A larger
RAM tier uses compressed radix lookup by default; the optional shared CPU
segments described below reduce duplication among related checkpoints.

Restoration runs inside the single-GPU worker after workspace reservation. It
checks allocation headroom before transfer and recomputes on an admission miss.
CPU copies become request-owned GPU state directly; the forward path does not
clone that attention state again. CPU snapshots remain immutable across branches.
Short text batches whose best checkpoint is on CPU fall back to individual
requests for this first version; ordinary GPU-prefix batching is unchanged.

Transfers are synchronous in this initial implementation. The default transfer path remains synchronous. Optional CPU lookahead, pinned
staging/layer-wise restore, and shared CPU segments are described below. GPU
paged attention and cost-based transfer-versus-recomputation selection remain
future work. A restored CPU checkpoint stays in RAM; new completed checkpoints may be
retained on GPU when the elastic budget permits. CPU cache capacity does not
increase the memory available for an active GPU request or its context limit.

`usage.prefix_cache_tier` reports `gpu`, `cpu`, `miss` or `recompute`;
`prefix_restore_ms` measures host-to-device restoration; `prefix_offload_ms`
measures CPU snapshot copies during the forward. Queue wait remains
separate. `cache_offload_ms` measures GPU eviction transfers during request
preparation (included in `cache_prepare_ms`). Direct CPU retention during the
forward is included in inference time. `/health` exposes host occupancy, limits,
hits, offloads, evictions, failures, restore time and restore rejections under
`optimizations.prefix_cache.host`. Answer and token-count semantics are unchanged.
Host stats also include `budget_mode`, `eviction_policy`, `reserve_fraction`,
`reserve_mib`, `available_mib`, and `cgroup_available_mib`; an unlimited entry
cap is reported as `max_entries: null`. Health reads update the estimated cap,
but eviction occurs in the serialized GPU worker rather than the health thread.

CPU regression checks: `python scripts/test_host_prefix_cache.py` and
`python scripts/test_host_memory.py`.


## Cache indexing and processed-image RAM

`CLEF_PREFIX_INDEX=radix` is the default. A compressed exact-token radix index
tracks GPU checkpoints, CPU checkpoints and metadata-only observations. It owns
no tensors. Eviction/clear remove index sources immediately; a checkpoint in
both tiers remains indexed until both copies disappear. Model identity, complete
ancestry, media identity, pooling and the no-mid-image boundary rules are
unchanged. `scan` retains the prior lookup implementation for controlled comparisons.

`CLEF_PREPROCESS_CACHE_MIB=auto` and `CLEF_PREPROCESS_CACHE_ENTRIES=auto` replace
the old 256 MiB / 128-image defaults. Processed CPU images and CPU prefix states
share one service-local RAM allowance using the 25% reserve fraction. By default
processed images can use at most 25% of that allowance, configured through
`CLEF_PREPROCESS_CACHE_RAM_FRACTION`. Prefix states can use the remainder, or the
whole allowance when image occupancy is low. The banks can evict each other to
make room, rather than independently claiming the same available RAM. Both use
LFU with oldest actual access breaking ties. Fixed MiB caps and zero to disable
remain supported. Tokenization's separate small bounded cache stays unchanged.

The processed-image cache contains exact processor output, not compressed image
files or language state. It can hit despite changed prompt beginnings. Model
backbone work still requires an exact prefix hit. Cached pixels are assembled
in original image order; no resizing, image format or answer-schema change is
introduced by these optimizations.

## Bounded preparation and transfer experiments

`CLEF_CPU_PREPARE_OVERLAP=1` enables one CPU preparation worker and at most one
queued preparation slot. The first request dispatches immediately. Already
waiting requests may tokenize/preprocess while the GPU worker runs, without
GPU work on the preparation thread. Wire tickets survive cancelled CPU work
until it actually completes. Preparation errors remain isolated per request.
`CLEF_CPU_PREPARE_MIB` defaults to 1024 and bounds estimated preparation workspace;
it is also capped at one eighth of sampled available RAM. Unknown/oversized
images remain inline. Cached image snapshots can qualify by their known size,
with 25% extra workspace slack. Per-request cache counters are thread-local.

CPU overlap defaults off: the initial cached-original-image concurrency test
showed no meaningful throughput gain. Enable only after benchmarking the target
workload. It does not add a batch-collection delay or change FIFO ordering.

`CLEF_PREFIX_ASYNC_RESTORE=1` enables experimental CUDA-only per-layer cache
restoration. A dedicated copy stream and I/O worker use two pinned staging
buffers totaling `CLEF_PREFIX_RESTORE_STAGING_MIB` (128 MiB by default). Hidden
vectors and global metadata become ready before use; decoder-layer hooks wait
for their corresponding cache events. GPU computation stays on the existing
single inference worker. The full RAM cache is not pinned. The CUDA attention
cache still materializes as dense tensors on the GPU, so this cannot extend an
active request beyond its GPU memory limit. Copy plans release source/destination
references explicitly, without waiting for cyclic garbage collection.

Async restore defaults off: the first benchmark preserved answers but did not
show a speed gain. ROCm uses the established synchronous path.

## Shared CPU checkpoint segments

`CLEF_PREFIX_SHARED_HOST_BLOCKS=1` enables immutable CPU segments for full-attention
K/V and final hidden vectors. Sharing is allowed only from the checkpoint that
was explicitly reused for the current request, with matching model, token
ancestry and media namespace. New suffix segments and all mutable convolution/
recurrent state are copied independently. No unrelated prompts are blended.

Physical RAM accounting charges each retained CPU storage once; `logical_mib`
reports the sum of complete checkpoint sizes and `shared_saves` counts snapshots
that reused parent storage. Evicting an ancestor preserves storage referenced by
descendants. Restoration recreates independent dense GPU state; this is CPU
storage sharing, not a paged GPU attention backend or copy-on-write GPU KV pool.
It reduces retained memory and duplicate GPU-to-CPU offload, while transfer
performance still depends on segment sizes and the hardware.

## Chunk-boundary short-request scheduling

`CLEF_CHUNK_INTERLEAVE=1` enables experimental CUDA scheduling at text/image
language-prefill chunk boundaries. Only the oldest queued, CPU-prepared text
request can interleave; images remain FIFO barriers. The default maximum child
length is 2048 input tokens (`CLEF_INTERLEAVE_MAX_TOKENS`) and at most four
children can run per parent (`CLEF_INTERLEAVE_MAX_PER_PARENT`). Eligibility also
requires the calibrated short-request workspace plus GPU headroom to fit beside
the parent's live state. If it does not fit, the parent continues and the queued
request waits normally.

This flag enables limited short-text preparation even when general CPU overlap
is off. Child execution runs uncached to bound extra state, on the same GPU
thread, then resumes the parent's independent cache/hidden state. It preserves
FIFO candidate order and prevents recursive interleaving. Parent workspace
reservation and peak-memory accounting survive child execution. Short replies
can complete before the long request; the long request can take longer by the
amount of interleaved work. This is a latency/fairness policy, not a promise of
higher single-request prefill throughput. It remains opt-in pending broader
mixed-workload validation.


Measured October 5 results are recorded in the local efficiency-stage report:
processed original-size image warm latency improved from 2.44 s to 0.95 s
(medians of three repeats); shared Full checkpoints retained 2.16 GiB for
7.19 GiB logical data, and Flash retained 1.11 GiB for 3.83 GiB logical data.
The current NVIDIA deployments enable shared CPU segments explicitly. New builds
keep that flag opt-in until the target model/backend is validated. General CPU
lookahead and async restore showed no throughput gain on this image workload and
remain off. The mixed-text interleave observation reduced the short request's
latency from 27.16 s to 7.52 s while increasing the parent from 27.42 s to 28.26 s;
it remains opt-in. These synthetic results do not establish broad accuracy or
throughput guarantees. The RX580 deployment was not changed in this round.


## Experimental active context offloading

`CLEF_ACTIVE_CONTEXT_OFFLOAD=hidden` moves completed backbone hidden vectors to
request-owned CPU RAM during incremental prefill, then releases working KV state
before restoring the hidden matrix for Clef's unchanged native decision head.
`kv_hidden` additionally keeps each full-attention layer's growing K/V history
in CPU RAM. Pre-hooks upload only the current layer; post-hooks copy the new K/V
tail back and append it to CPU history. Linear convolution/recurrent state remains
on GPU. This mode does not quantize activations or discard earlier tokens.

The default is `none`. These modes are experimental, CUDA-only, and initially
validated with Flash on an RTX 3070 Ti. They require incremental prefill. Prefix
retention, independent GPU feature retention, batching experiments and chunk
interleaving must remain off during controlled evaluation. The path is separate
from RAM prefix caching between requests. Hooks are removed on success or failure;
CPU snapshots here belong to the active request rather than the reusable cache.

The normal context estimator is not calibrated for this path and therefore
falls back to configured limits. Do not treat an experimental configured limit
as measured capacity. The isolated benchmark uses fixed limits only to probe
actual failures; production discovery and deployed defaults remain unchanged.
Host admission conservatively requires the projected CPU state to fit within
half of currently sampled available RAM. GPU model weights, the current layer's
KV, attention/mask/GQA workspace and the final head still need to fit VRAM.

Usage reports `active_context_offload`, `hidden_offload_ms`, `hidden_restore_ms`,
`active_hidden_mib`, and (for KV mode) `kv_upload_ms`, `kv_offload_ms`,
`kv_uploaded_mib`, `kv_offloaded_mib`, and `active_cpu_kv_mib`. Language time
includes synchronous transfers; the final hidden restoration is reported
separately. Smaller continuation chunks reduce GPU attention workspace but
increase repeated KV uploads and kernel/scheduling overhead. No compiler or new
package is required. The native model and decision head remain unchanged.


The isolated 8 GiB 3070 Ti experiment processed 98,270 text tokens with
`kv_hidden`, an 8192-token initial chunk and 1024-token continuations: 140.1 s
and 7.29 GiB peak allocated VRAM. The 65,502-token case took 73.7 s and peaked at
6.48 GiB. With the same 1K chunks but resident GPU state, 32,734 tokens passed
and 64K failed. With 4K chunks, the resident path passed 16K but failed 32K;
hidden-only offload passed 32K. Successful synthetic runs recovered facts at the
beginning, middle and end. These are single observations, not a claimed maximum
or a broad long-context quality result. The original service was restored.

### Experimental streamed active KV

`CLEF_ACTIVE_CONTEXT_OFFLOAD=kv_stream` replaces whole-layer KV restoration with
blockwise full attention. `CLEF_KV_STREAM_BLOCK_TOKENS=4096` controls the history
staging block independently of the initial/continuation prefill chunk sizes.
The default active-offload mode remains `none`.

The request owns preallocated pinned **CPU RAM** buffers for each full-attention
layer's K/V, stored in token-major order so history slices are contiguous. Two
bounded **GPU VRAM** buffers alternate uploads on a CUDA copy stream. Attention
on the current block can overlap upload of the following block. Every earlier
unmasked token is still read. Per-block outputs are combined with log-sum-exp
weights in FP32; the current chunk uses causal attention, while older history
blocks are fully visible. This is the full softmax calculation, subject to
ordinary floating-point differences; it is not selective or sparse attention.
Only new KV returns to CPU RAM after a layer. There is no growing CPU `cat`.

The recurrent layers remain native and GPU resident. Completed hidden vectors
remain in CPU RAM until backbone KV is released, then return to GPU VRAM for the
unchanged Clef head. Streaming avoids both complete per-layer GPU KV restoration
and the ordinary full-history causal-mask allocation. It does not eliminate
repeated history traffic: larger continuation chunks still reduce transfer volume.

The experiment is limited to one unpadded text request on NVIDIA CUDA and the
Qwen3.5 backbone. Images are explicitly rejected. Batching, prefix retention and
chunk interleaving must remain disabled. Attention methods/configuration are
restored when the request ends or fails. It calls PyTorch's internal efficient
SDPA operator with log-sum-exp output, so compatibility with a different PyTorch
release requires validation. It adds no compiler, GPU binary or dependency.

Usage includes `kv_stream_layout`, `kv_stream_block_tokens`,
`kv_stream_history_blocks`, `kv_gpu_staging_mib`, `kv_pinned_allocated_mib`,
`active_cpu_kv_mib`, `kv_uploaded_mib`, and `kv_offloaded_mib`. The staging-buffer
counter is not total GPU memory: query/output/GQA workspaces, weights, recurrent
state and the head also require GPU VRAM. `kv_offload_ms` includes waiting for
queued attention work; do not interpret it as isolated DMA transfer time.

Run `python scripts/test_streamed_kv.py` on a CUDA GPU to compare streamed GQA
attention with FP32 dense causal attention, including odd block tails. Benchmarks
must also compare complete model answers at matched lengths/chunk sizes and
track total time, language time, peak allocated GPU VRAM and transferred bytes.


Measured on Flash 9B compact NF4, RTX 3070 Ti, PyTorch 2.11.0+cu128:

| At 65,502 input tokens | Continuation chunk | Median total seconds | Input tokens/sec |
|---|---:|---:|---:|
| Whole-layer CPU RAM offload | 1,024 | 73.09 | 896 |
| Streamed, head-major CPU RAM | 1,024 | 62.88 | 1,042 |
| Streamed, contiguous CPU RAM | 1,024 | 44.83 | 1,461 |
| Streamed, contiguous CPU RAM | 4,096 | 39.77 | 1,647 |
| Streamed, contiguous CPU RAM | 8,192 | 38.72 | 1,692 |

Each standard cell is the median of three warmed, uncached requests. All modes
used an 8192-token initial chunk. The matched 1K control separates the storage/
streaming changes from the larger-chunk gain. Head-major versus contiguous also
includes an in-place FP32 accumulation change, so it is not a pure layout ablation.
The 8K configuration reached 98,270 tokens in 68.80 s (1,428 input tokens/sec)
with 7.29 GiB peak allocated GPU VRAM; this long probe was a single observation.
The approximately 131K probe failed with GPU OOM; the failure phase was not
instrumented. The unchanged native head and its hidden-vector workspace still
need GPU VRAM. No greater maximum than the passing 98K probe is established.

All passing runs recovered the beginning/middle/end facts. At matched lengths,
choice probabilities differed from the whole-layer baseline by at most 0.0001
at the API's reported precision. These synthetic facts do not establish broad
model accuracy. At 64K all offloaded variants peaked at 6.48 GiB GPU allocation,
so the faster path did not reduce the overall request peak there.

The experiment remains opt-in. See the [recorded benchmark](../benchmarks/streamed-kv/README.md)
for throughput, memory and transfer observations.

### Experimental CPU-backed head preparation

`CLEF_HEAD_CPU_PREPARE=1` enables tiled preparation of native head inputs when
active context offloading is enabled. `CLEF_HEAD_PREPARE_CHUNK_TOKENS=4096`
controls the preparation chunk independently of backbone/history block sizes.
It is off by default. The validated setup uses `kv_stream`, single unpadded text
requests, 8192-token initial/continuation chunks and 8192-token history blocks.
Prefix retention, batching and chunk interleaving remain disabled. The configured
input ceiling must also allow the requested total tokens; the ordinary GPU-only
context estimator is not calibrated for this experimental path.

The full hidden matrix stays in **CPU RAM** after backbone KV is released. Each
chunk moves to **GPU VRAM** for the original hidden normalization and memory
projection. These operations act independently on token rows. Normalized rows
return to CPU RAM, while the smaller full projected-memory matrix stays in GPU
VRAM. No token is omitted or summarized. Question/option spans and the last
normalized vector move to GPU VRAM for the original native mean reductions and
all subsequent evidence routing, field processing and scoring.

The upstream `JointSchemaHead.forward` and its trained weights are unchanged.
Request-local adapters supply its prepared normalized spans and projected memory;
two module forwarding methods are restored in `finally`, including on exceptions.
This is inference-only and requires the native head, one unpadded request, and
CUDA. It is not a compatibility claim for a compiled/custom head or concurrent
head calls. Host admission accounts for the additional normalized CPU RAM matrix.
No compiler or dependency is added.

The instrumented 131,198-token control completed the backbone and failed inside
`EvidenceRoutingLayer.memory_norm`: a 258 MiB allocation was requested with
253.38 MiB free. This identifies the previous ceiling as head workspace pressure.
With CPU-backed preparation, all three runs processed **131,198 tokens** on the
8GB RTX 3070 Ti. Median total time was **105.17 s**, **1,247 input tokens/sec**,
with **6.11 GiB** peak allocated GPU VRAM. Beginning/middle/end facts and yes/no
answers were stable across repetitions. At 64K, peak allocation fell from 6.48
to 6.05 GiB, while elapsed time remained about 39 seconds.

At 16K and 64K, prepared-head probabilities matched the native integrated control
at the API's reported precision. CUDA head tests in FP16/BF16 cover choice,
yes/no and score fields, cross-chunk spans and failed-call cleanup; maximum
logit difference was 0.000244. This is synthetic validation, not broad accuracy
coverage or proof of maximum context capacity. Full, images, ROCm, other GPU
families and concurrency remain unvalidated. Production discovery/defaults are
unchanged and the original RTX 3070 Ti service was restored after testing.

Usage adds `head_cpu_prepared`, `head_prepare_chunk_tokens`,
`head_cpu_prepare_ms`, `head_projected_memory_mib`, `head_normalized_cpu_mib` and
`head_span_upload_mib`. `head_ms` includes preparation and the native head.
`hidden_restore_ms` is zero because the full hidden matrix is not restored to GPU
VRAM. Optional `CLEF_OFFLOAD_TRACE=1` logs OOM stage and token count for controlled
diagnosis. See the [131K benchmark](../benchmarks/context-131k/README.md).

### Adaptive NVIDIA routing and cache reuse

CUDA startup selects `CLEF_ACTIVE_CONTEXT_OFFLOAD=auto` for both Flash and Full.
The earliest path that fits the complete request's projected memory needs is used:

1. `gpu_native`: existing GPU-resident execution, prefix cache and small batching.
2. `gpu_tiled`: KV stays in GPU VRAM, but full attention reads it in blocks;
   completed hidden/normalized vectors stay in CPU RAM for head preparation.
3. `cpu_streamed`: full KV stays in CPU RAM and bounded buffers stream history
   into GPU VRAM; the head uses CPU-backed preparation.

There is no fixed 16K cutoff. Available memory, GPU/model configuration, head
width, schema spans and cold-request peaks determine the switch. The configurable
`CLEF_ROUTING_RESERVE_MIB` margin defaults to 256 MiB. Warm suffix work cannot
lower the cold estimate. Calibration is saved atomically under the data directory,
keyed by GPU UUID, model revision/configuration, precision, Torch/CUDA versions,
prefill policy and linear kernels. An OOM backs off the failed mode at comparable
memory availability; freeing memory can reopen resident execution. The worker
retries the next path while preserving valid CPU snapshots. One GPU belongs to
each service, and calls remain serialized during temporary attention/head hooks.

`/health` and `/v1/models` expose `context_limit.routing_limits` for `gpu_native`,
`gpu_tiled` and `cpu_streamed`, plus the overall accepted estimate. Usage reports
`execution_strategy`, `context_switch_reason`, `context_fallbacks` and actual
`reused_prefix_tokens`. Cold and cache-assisted limits are reported separately; per-request admission
checks its actual matching prefix. Explicit fixed caps remain honored. Estimates are not
allocation reservations or model accuracy guarantees.

The existing prefix index serves every path. Larger requests borrow immutable
CPU snapshots, restore recurrent state into GPU VRAM, and seed owned KV/hidden
storage without first restoring complete KV to GPU. Streaming borrows immutable
CPU prefix blocks directly and allocates only the new KV suffix. Final body
checkpoints adopt append-only CPU storage without duplicating the full history;
recurrent state remains independently copied. New CPU checkpoints share
an ancestor while independently copying mutable recurrent state. GPU-native
prefixes can seed either path; attention partitioning can introduce ordinary
floating-point differences. GPU evictions spill through the CPU tier. CPU RAM
admission reclaims least-used unrelated snapshots while protecting the matched
prefix. Only bounded transfer/tail buffers are pinned, not full KV history.

Images retain global positions and GPU vision encoding. Tiled/streamed execution
moves completed image features into CPU RAM and loads current-chunk features into
GPU VRAM. Image-feature reuse and media identity checks remain, along with explicit
image caps. Prefixes cannot match across different image bytes or pooling settings.

The policy targets supported CUDA hardware (SM75+; Full still requires BF16 and
enough VRAM for model weights). Startup probes the installed attention operator;
if unavailable, native execution remains. Hardware-policy tests cover Turing,
Ampere, Ada, Hopper and Blackwell; those are compatibility checks, not actual-card
benchmarks. ROCm keeps its separate policy. Set `CLEF_ACTIVE_CONTEXT_OFFLOAD=none`
for legacy behavior; forced modes remain available for controlled calibration.

Large CPU snapshot admission credits allocated-but-unpublished bytes once.
This prevents free-RAM reduction during a copy from recursively shrinking its
own cache budget. Adoption and pending credits are covered by regression tests.
