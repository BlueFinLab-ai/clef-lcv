# Validation record

Saved October 3, 2026. Original deployed services were tested on their GPUs before
this project was packaged. This file contains aggregate measurements only.

| Full 27B on RTX 3090 | Fallback | Optimized | Category matches |
|---|---:|---:|---:|
| 100-email original prompt | 201.68 s | 148.70 s | 96/100 for both |
| Same 100 emails with category guide | 365.62 s | 269.47 s | 100/100 for both |

All 200 top categories were unchanged. Optimized measured time fell about 26%.
Times exclude warm-up and initial kernel compilation. These compare with frozen
assistant labels; they are not human-reviewed accuracy or held-out evaluations.
The guide was developed from the same sample. No private emails, identifiers,
subjects, bodies, per-email labels, or learned examples are included here.

Original full validation passed short typed requests, image fidelities, exact
16,384-token text, and four high-fidelity images at an exact 16,384-token combined
input. A 16,385-token request returned 413. Peak allocated memory for the largest
optimized check was approximately 20.27 GiB. FLA/causal-conv numerical comparisons
passed within the recorded BF16 tolerances.

Original Flash validation passed image inputs with compact embeddings. On the
2080 Ti it passed 24,576-token combined inputs, including four high-fidelity
images. The 3070 Ti profile retains its tested 8,192-token service cap.

Packaging changes are limited to portable filesystem paths, profile launch and
preparation scripts, the shared UI's model selection from `/health`, and a portable
kernel build recipe. `scripts/check_project.py` checks Python syntax, profile/schema
consistency, source provenance, local documentation links, required UI elements,
and accidental sensitive-file inclusion. JavaScript syntax is checked separately.

A clean download/quantization/kernel-build/serve cycle for this packaged repository
still needs GPU validation before a public release. The save operation did not
modify or restart any running service.

The packaged adapters were also imported in the existing Linux runtime in a
temporary directory. Both returned 200 for the homepage, JavaScript, CSS, and model
listing; both rejected an empty decision request with 422 and preserved the low
fidelity request default at the initial save. The current default is Standard
256², documented in [runtime optimizations](runtime-optimizations.md). Those checks used no model loading or GPU allocation.


## Implemented runtime optimizations (2026-10-03)

The shared runtime implements compact schema formatting, one preprocessing pass,
bounded prefix reuse and opt-in 2×2 image pooling. Image fidelity defaults to
Standard 256²; Low/Medium/High retain 128²/512²/1024².

Spare-GPU integration checks compared fresh, cold-cache, warm-cache and changed
questions for Full and Flash: five input cases each, all three native answer
types, text, single images and four-image pooling. All selected options matched.
Maximum probability differences were 2.30 percentage points for Full and 0.131
for Flash; cached probabilities are not guaranteed bit-identical.

Each live service passed 21 API checks, including the supplied photographs at
high fidelity, default fidelity, pooling, exact-prefix invalidation, reordered
questions, overlength rejection and invalid input rejection. The CPU helper tests
check multi-image grid/placeholder remapping, question spans, shared tensor storage
accounting and availability of unused PyTorch-reserved memory.

A follow-up live API benchmark used two visible-attribute questions, three measured
repeats per cell, two photos, Standard/High fidelity and pooling off/on. On the beach
photo at High fidelity, median fresh time was 1.427 s, fresh pooling time 0.927 s,
and warm-prefix time 0.506 s. These are service timings from the Full RTX 3090 at 400 W; no clock
or thermal normalization was applied. First requests can include kernel compilation
and cache-building costs and take substantially longer.


Final regression checks passed exact 16,384-token Full, 8,192-token 3070 Ti Flash
and 24,576-token 2080 Ti Flash requests. All three subsequently retained a new
small prefix and returned a cache hit on repeat. Inputs at or above 8,192 tokens
bypass prefix caching. Cache eligibility counts reusable allocator blocks, fixing
the case where a large request leaves unused reserved GPU memory.

Updates are running on the existing three services, with their GPU UUIDs,
power limits, checkpoints and context caps preserved. A rollback copy of each
adapter was retained on its host. The packaging checker, JavaScript syntax check,
upstream wrapper checksum and Git whitespace checks passed. A new clean download/
quantization/kernel-build cycle remains unvalidated, as described above.


## Sixteen-image request limit (2026-10-03)

Full and both Flash services accept up to 16 images per request. Each passed a
16-image request at default Standard 256² fidelity, with pooling off/on and cold/
warm prefix reuse; the synthetic red-image classification remained correct. Each
rejects 17 images with 422. OpenAPI and health metadata advertise the new limit.

The browser accepted 16 files, rejected an additional file, allowed the same file
to be selected after removing one, and completed a 16-image request. Combined
context caps, individual file-size and source-pixel limits are retained. Sixteen
high-fidelity images have not been validated and may exceed context or VRAM limits.

## Adaptive branching prefix cache (2026-10-03)

Full and both Flash services now retain multiple checkpoints with automatic VRAM
budgets, bounded eviction, CPU metadata for identifying repeated branches, and
longest exact-prefix matching. The GUI/API expose reusable context before images
and changing state. Both models passed 19 synthetic GPU checks; each deployed
service passed 18 HTTP checks, including branch promotion, deeper checkpoint hits,
16 images, changed media, pooling and high fidelity. No OOM fallback occurred in
these live checks. GPU process mapping confirmed one GPU per service and the spare
3090 was released. Deployed runtime checksums match the saved project.

Both cached 100-email trials per model preserved every uncached category choice.
Warm time fell from 321.4 to 192.6 seconds for Full and from 196.2 to 103.7 for
Flash. Alternating four contexts gave 6.26× Full and 3.98× Flash speedups against
a one-entry cache. See [benchmark methods and limits](prefix-cache-benchmarks.md).
KV/recurrent snapshot copying remains; paged sharing and the queue prototype are
separate future work. Existing model context caps and the 8,192-token cache cutoff
remain in force.

## Independent input reuse (2026-10-03)

Full and both Flash services now use bounded exact CPU preprocessing/token caches and independent GPU image-feature reuse. Both models passed 44 checks each: three exact processor comparisons and 41 native answer/probability comparisons, including reordered original photos, changed context, partial replacement, pooling toggles and 16 duplicate images. No selected options changed. Full max probability delta was 3.81 percentage points in the 16-duplicate stress test, 0.28 elsewhere; Flash max was 0.064 points. NF4 kernels can vary with image-batch shape. See [benchmark results and limits](input-cache-benchmarks.md).

All three updated services passed 14 live HTTP checks each, including bypass flags, changed fidelity, exact prefix layering, invalid image rejection and the 16-image cap. No GPU memory fallback occurred. Live runtime hashes match the project; NVIDIA process mapping confirms each service remains on its original single GPU. Default 256 pixel budget, optional pooling, model precision and input token caps remain unchanged.

CPU tests `test_input_cache.py`, `test_image_feature_index.py`, `test_prefix_index.py` and `test_inference_helpers.py` passed; project syntax/provenance and JavaScript syntax checks passed. The shared GUI loaded with the existing controls and current asset version. Private source photos and raw requests remain outside the saved project and source ZIP. Queue/microbatch scheduling remains separate future work.

## Model context budget discovery (2026-10-03)

Both HTTP adapters now advertise `data[].max_input_tokens` in `GET /v1/models`.
The field uses the same configured limit as `/health` and the existing 413
overlength check. Clients should check the actual service's metadata and budget
for the complete encoded request. No dynamic memory predictor or truncation was
added.

Live checks passed on all three services: Flash on the 3070 Ti advertised 8,192
tokens, Full on the 3090 advertised 16,384, and Flash on the 2080 Ti advertised
24,576. Each value matched `/health`; each service answered a normal 117-token
request and rejected an oversized request with the unchanged HTTP 413 status and
detail. Local discovery-function checks confirmed that both adapters follow
their configured limit. Project syntax, provenance, and documentation-link checks
passed. Rollback copies of the previous deployed adapters were saved before the
update; the earlier stable checkpoint and experimental archives are preserved.

## Incremental prefill serving integration (2026-10-03)

The tested text-only experiment is now integrated into both HTTP builds. Requests
longer than 8,192 encoded tokens use an 8K first chunk and 4K continuations, one
request-local hybrid state, and the unchanged native decision head over all token
outputs. Images use the existing multimodal path. Each service remains on its
original single GPU; model weights and compute precision are unchanged.

| Deployment | Text budget | Budget with images | Verified longest text | Chunks | Peak allocated |
|---|---:|---:|---:|---:|---:|
| Flash, RTX 3070 Ti | 24,576 | 8,192 | 24,576 | 5 | 6.80 GiB |
| Full, RTX 3090 | 16,384 | 16,384 | 16,384 | 3 | 19.06 GiB |
| Flash, RTX 2080 Ti | 24,576 | 24,576 | 24,576 | 5 | 6.80 GiB |

`GET /v1/models` and `/health` advertise `max_input_tokens` and the total-request
limit `max_input_tokens_with_images`. The browser displays both when they differ.
Both overlength paths retain the existing 413 response without truncation.

All three services passed synthetic short-text, long-text, and single-image
comparisons against their pre-update API responses. Selected answers matched in
all nine reference cases. Maximum probability drift was 0.15 percentage points
for Full at 16K and 0.07 for the 2080 Ti Flash at 10K; the 3070 Ti's unchanged 8K
path and all short/image references had zero drift. Additional long-text checks
at 10K and the advertised Flash limits completed and retrieved the tested start
and end facts correctly. These checks do not establish general long-context
accuracy; the separate recorded artifact remains unresolved.

Text and image-containing oversized requests returned 413 on every service.
Repeated short prompts recovered a prefix-cache hit after the long requests and
rejections. The initial pilot caught a cache-budget regression: reserving long
chunked workspace alongside short-request caches exhausted the 3070 Ti's cache
budget. Chunked requests now clear GPU caches and record their workspace
separately, preserving the existing reserve for non-chunked requests.

The 3070 Ti's final 24K HTTP observation was 15.18 seconds; the 2080 Ti's was
12.95 seconds. Full's first 16K chunked call took 25.85 seconds, followed by three
warm calls with a median of 14.65 seconds. Its pre-update single-pass observation
was 13.59 seconds and 20.06 GiB peak allocated. The new path therefore saved about
1 GiB at 16K, with a modest observed warm latency cost. These serving checks are
not a controlled repeated before/after speed benchmark. Driver/context memory
and allocator reservations are additional to peak allocated memory.

Project syntax/provenance/link checks, JavaScript syntax, and CLI option checks
passed. Per-file rollback backups were retained on the hosts before deployment;
the original stable checkpoint and isolated experimental source archives remain
preserved. `scripts/test_chunked_prefill_api.py` provides the synthetic regression
and admission checks for future deployments.


## Computed context discovery and admission — October 3, 2026

The previous Full 16K and Flash deployment values were configured caps. The
calibrated NF4 / efficient SDPA / 8K-first + 4K-continuation builds now advertise
the actual conservative memory projection, rather than treating those caps as
computed ceilings. The same estimator is used in both adapters and admission.

| Service | Computed text budget | Retained image-request budget | Measured text peak | Text request time |
|---|---:|---:|---:|---:|
| Flash, 3070 Ti 8 GB | 24,576 | 8,192 | 6.80 GiB | 15.25 s |
| Flash, 2080 Ti 11 GB | 65,536 | 24,576 | 9.38 GiB | 44.46 s |
| Full, 3090 24 GB | 45,056 | 16,384 | 22.07 GiB | 50.93 s |

Each measured text request contained exactly the advertised number of encoded
tokens and four typed questions. All produced finite answers and correct start/end
facts for that fixture. This does not resolve the previously recorded accuracy
artifact. These are single-request timings, not an inference-speed comparison
between models or a universal context-capacity guarantee.

Tests verified normal text and image answers against saved references, text
rejection at exactly one token over the computed limit, image overlength 413,
recovery of prefix-cache hits after long requests, and discovery during active
inference. Busy discovery retained the last idle estimate instead of shrinking
with temporary allocations. All 18 deployed source-file hashes matched and each
service remained mapped to exactly one GPU. Native browser inspection confirmed
the Full UI displayed “Estimated: 45,056 text input tokens; 16,384 with images.”

CPU tests verify KV and hidden-vector arithmetic, allocator slack and reclaimable
cache accounting, dynamic memory changes, architectural bounds, workspace
observations, OOM ceilings, and fixed/uncalibrated fallbacks. The portable syntax,
profile, provenance, UI/link checks and JavaScript syntax checks passed.

See `scripts/test_context_budget.py` and `scripts/test_computed_context_api.py`.
Raw deployment records are kept outside this repository.
The calibration, reserve, rounding, fallback controls, and limitations are detailed
in [context budget discovery](runtime-optimizations.md#context-budget-discovery).

## Incremental image prefill — October 3, 2026

Mixed image/text chunking is now enabled on Full and both Flash services. Tests
used the two supplied JPEGs at 256, 512 and 1024 fidelity, reversed image order,
optional pooling, boundaries inside image-token blocks, and sixteen-image
requests. Whole-image vision encoding uses one-image GPU batches; language
chunks preserve global spatial positions and all outputs for the native head.

All 21 Flash comparisons passed the selected-answer and 1.0 percentage-point
probability agreement gates. All 21 Full comparisons within 16K passed. Full's
24K two-image case exceeded the probability gate at 1.052 points, so its image
ceiling remains 16,384. Flash's 3070 Ti image ceiling increased from 8,192 to
24,576; the 2080 Ti retains 24,576. Computed text ceilings are unchanged.

All three deployments passed seven live image cases, one-token-over-limit 413
checks, prefix-cache recovery, and text regressions through their computed
limits. Eighteen deployed source hashes matched and single-GPU service mapping
was verified. CPU semantics, context-budget and portable project checks passed.
Original photos remain outside the project and source archive.

See [incremental image prefill](multimodal-prefill.md) for memory measurements,
timing limitations, reproduction scripts and rollback controls. Raw records are
kept outside this repository.

## Full 27B numerical isolation and context quality — October 3, 2026

Three native repetitions were identical. Freezing image features isolated a
0.461 percentage-point maximum change from 8K/4K language chunking; changing
vision batch size alone produced 0.415 points. Combining both reproduced the
original 1.052-point shift exactly, with unchanged selected answers. The largest
change was on the text urgency question. This numerical alarm did not establish
an accuracy failure or a context-quality boundary.

A separate suite tested 11 fixtures with varied log filler, facts at three
positions, two photo orders, rotated fact answers, and up to sixteen images.
All 120 known-answer checks across 26 successful trials passed, including all
102 chunked checks. Both chunk layouts passed through 45,056 encoded tokens;
the sixteen-image peak was 22.22 GiB. Native 32K text and image references ran
out of memory, so longer comparisons rely on known truth and agreement between
chunk layouts, not an unavailable native reference.

This study left production configuration unchanged. Idle discovery still
reported 45,056 text tokens and a configured 16,384-token image ceiling. The
spare GPU was released. See [Full context quality](full-context-quality.md) for
the measurements, interpretation, scope and reproduction scripts. Raw results
are kept outside this repository.


## Automatic Full image admission — October 3, 2026

The stale Full image ceiling has been removed. `CLEF_MAX_IMAGE_LENGTH=auto`
(default for the Full build) follows the computed memory budget when image
chunking and calibrated estimation are enabled. Discovery and admission now
agree at 45,056 total tokens for both text and image requests on the deployed
3090. Explicit numeric image caps remain available; automatic mode retains the
configured fallback for fixed, uncalibrated or non-chunked configurations.

The live Full RTX 3090 service passed maximum-length requests with two reversed photos
and sixteen high-fidelity images, matching the isolated reference answers and
probabilities within 0.1 percentage points. The sixteen-image call peaked at
22.22 GiB and took 56.38 seconds including preprocessing. Both text and images
returned 413 at exactly 45,057 tokens. Busy discovery retained the idle budget;
post-request discovery stayed at 45,056. The 8K text regression and short prefix
cache recovery passed. Six deployed source hashes matched the saved source.

CPU checks cover automatic images, explicit caps, memory changes, OOM backoff,
and safe fixed/uncalibrated/non-chunked fallbacks. Project checks passed. The
change is persistent in the Full service drop-in; per-file rollback copies were
saved before deployment. See `scripts/test_auto_image_budget_api.py`; raw records
are kept outside this repository.


## Long-request GPU caching — October 3, 2026

The fixed 8K cache bypass is removed in both HTTP adapters. The Full RTX 3090 service is
deployed and validated; live Flash services remain unchanged. Eleven image
slots (the two private photos repeated), 12,288 tokens and six known-answer
questions took 11.473 s cold and 0.742 s warm median on the live API,
15.5× faster, with identical recorded probabilities. Changed questions
reused the same state; reordered images and pooling reused independent features.
45,056 tokens with sixteen image slots passed after caches were evicted for
workspace, while 45,057 returned 413. Subsequent smaller calls regained cache
hits. Discovery stayed at 45,056; no OOM fallback occurred. Six deployed hashes
matched; the spare GPU was released. See [long-request cache validation](long-request-cache.md)
for timing definitions, isolated baseline checks, scope and reproduction.


## Client image resizing — October 3, 2026

The GUI now resizes before upload and sends native `media_kwargs` with
`do_resize: false`. Both HTTP adapters accept upstream processor options and
use native defaults when omitted. Legacy fidelity remains explicit/deprecated.
All three live services passed four fidelities, repeat equality/cache hits,
pooling, native/legacy budget equality, native defaults, text regression, 400
conflict rejection and 413 token admission. Standard fixture payloads shrank
93.7%; maximum measured client/Pillow probability shifts were below
0.6 points, with unchanged selected known answers. Fifteen deployed hashes
matched; one GPU per service was confirmed. Flash's long-cache rollout remains
separate. See [client resizing](client-image-resizing.md) for scope and reproduction.


## Image Scaling — October 4, 2026

The client menu now offers Original and display frames through 4K UHD, with
Compact 256 × 256 as default. All eight choices passed actual browser uploads;
56 geometry combinations passed CPU checks. Compact passed live on all three
services; HD, 4K and Original also passed on Full with uploaded dimensions
preserved and no processor resizing. Repeated preparation was reused. Static
files only were deployed with rollback copies and matching served hashes.
See [image scaling](image-scaling.md) for dimensions, encoding and test scope.


## Preserve source image format — October 4, 2026

The GUI downscales in the source PNG/JPEG/WebP format, with no automatic
conversion. Original and already-small sources bypass encoding and were
verified byte-identical across four fixture files. Browser scaling retained
PNG/WebP alpha. All eight JPEG presets passed browser transport checks, and
repeat requests reused prepared bytes. Native processor options now enable
patch alignment with min/max pixels of 1024/20 million; selected frame resizing
remains browser-side. All three live services passed Compact JPEG/PNG/WebP,
and Full also passed HD, 4K and Original with expected patch dimensions.
Geometry checks (56 combinations), encoder substitution/oversize rejection,
project validation and served file hashes passed. Only static files changed;
rollback copies were saved. See [image scaling](image-scaling.md).


## Elastic GPU cache — October 4, 2026

The Full RTX 3090 service now uses remaining VRAM for retention with a fixed 1 GiB reserve and
active workspace. The two original-photo request (23,805 tokens) retained its
language checkpoint; three live repeats had a 2.484 s median server time versus
24.511 s for the previous repeat baseline (9.9× faster), with identical recorded
answers and probabilities. Changing questions reused the prefix; short text
cache reuse also passed. Idle/request budgets were 5,456/2,322 MiB for this case.

The isolated 3090 comparison confirmed that a 100% target with old accounting
still rejected retention, while elastic accounting worked. A 45,056-token request
evicted caches, completed, and smaller calls rebuilt/hit afterwards. No OOM
fallback occurred. Live text at 45,057 tokens returned 413 and released the
workspace reservation back to idle. Both deployed source hashes matched the
installer manifest; the spare GPU was released. Flash's live policy and context
limits remain unchanged. CPU policy/index checks and project validation passed.
See [elastic cache details](long-request-cache.md#elastic-gpu-cache--october-4-2026)
for timing definitions, independent cache limits and targeted test scope.


## Adaptive cache headroom — October 4, 2026

The Full RTX 3090 service and both Flash services now run the shared long-cache runtime and
elastic retention with a 100% memory target, 256 MiB idle/short headroom and
at least 512 MiB for calibrated projected prefill. Before large processing,
cache eviction selectively returns unused allocator blocks to CUDA. The older
Full-only 1 GiB and unchanged-Flash statements above describe previous stages.

Isolated 3070 Ti and 3090 tests passed; live tests passed on all three services.
The 3070 Ti's former 24K pressure case completed in 15.727 s with no retry and
90 ms cache preparation (366.4 MiB evicted), while repeated 8K requests took
0.341–0.380 s. Full's 45K and the 2080 Ti's 64K text calls passed without retry.
Sixteen-image-slot mixed pressure and smaller text/image cache recovery passed.
Cold/warm choices matched and probability changes stayed within 0.2 percentage
points for the tested fixtures. Full's original two-photo request retained its
23,805-token state and repeated at a 2.296 s server median with identical answers.
These targeted fixtures do not settle the deferred long-context accuracy issue.

Discovery retained 24,576 text/image on the 3070 Ti, 65,536 text / 24,576 image
on the 2080 Ti and 45,056 text/image on Full. Text cap plus one returned 413.
CPU elastic, prefix-index, feature-index, inference-helper and context-budget
checks passed; project checks passed 72 files. Six source hashes per deployment
matched, and each unit stayed on its assigned single GPU.
Private temporary photo copies were removed. See
[adaptive cache measurements and controls](long-request-cache.md#adaptive-cache-headroom--october-4-2026).



## FIFO request queue — October 4, 2026

Deployed on the Full RTX 3090 service and both Flash services. There is one inference worker
per configured GPU, a 64-job waiting limit, a five-minute queue wait deadline,
a 256 MiB admitted wire-payload budget and a 224 MiB per-body limit. Requests
wait asynchronously; preprocessing, capacity checks, cache preparation and
inference execute inside the worker. GPU batching remains a separate prototype.

| Service | Valid calls waiting behind an 8K call | Observed queue wait | Eight simultaneous calls | Final text / image cap |
|---|---:|---:|---|---:|
| Full, RTX 3090 | 5 | 6.543–8.163 s | All 200 | 45,056 / 45,056 |
| Flash, 3070 Ti | 5 | 4.703–7.075 s | All 200 | 24,576 / 24,576 |
| Flash, 2080 Ti | 5 | 4.069–5.592 s | All 200 | 65,536 / 24,576 |

The backlog also contained one overlength request: six waiting calls in total.
Its token cap plus one correctly returned 413 after queueing, with the exact
count/limit and queue timing headers. All valid calls succeeded with choices
matching their serial controls; tested probability differences were within
0.2 percentage points. Synthetic red/blue JPEGs exercised native images and
optional 2×2 pooling without using private photographs. This is a functionality
check, not a throughput improvement or broad accuracy claim.

Each service passed a real TCP disconnection while queued: the pending call was
removed without execution, and admitted count/bytes returned to zero after the
active call finished. Image repeats recovered a prefix hit on all services.
The 3070 Ti initially missed its old image-prefix checkpoint after 8K pressure,
while reusing both independent image features; later repeats hit the prefix.
Retention remains subject to existing memory and entry limits.

An isolated spare 3090 first passed the same concurrency/disconnect tests.
CPU/ASGI tests passed FIFO execution on one thread, count/byte exhaustion,
streamed/known body limits, malformed JSON without admission leaks, deadline
expiry, task/client cancellation, active-call budget retention, worker-error
recovery and shutdown. The CPU suite passed locally and in the deployment's
Python environment. Project checks passed 75 files. Nine source/asset hashes
per service matched the saved project; units remain on their assigned single
GPUs. The GUI separates queue wait from processing,
allows ten minutes for the HTTP response, and shows detailed token 413 errors.

See [queue controls and semantics](runtime-optimizations.md#request-queue).

## Tensor-only cache accounting (2026-10-04)

Full and Flash now count retained tensor-bearing fields without walking CPU token
ancestry or other checkpoint metadata during VRAM budget scans. Shared storages
remain deduplicated across checkpoints and image features. Exact byte totals,
shared views and branches, protected eviction, feature eviction, cloned retention
and clearing passed the new CPU regression tests. Prefix indexing, feature
indexing, inference helpers, elastic cache policy and queue tests also passed.

A controlled spare-3090 A/B test used the same loaded Full NF4 model and four
retained 6,867-token text prompts, alternating only the accounting implementation.
All 32 requests were cache hits, every selected answer matched, probabilities
agreed within 0.00001 and storage totals matched. Median complete inference time
including preparation and finish fell from 584.3 to 275.9 ms. A CPU-only metadata
stress fixture measured 124.23 versus 2.61 ms per accounting scan on the host.
These are workload-specific measurements, described in
[cache accounting](cache-accounting.md).

The updated runtime hash matches all three active services, each on its previous
single GPU. Thirty recorded live requests passed image repeats, optional pooling,
reordered media and concurrent text calls. Queue count and payload bytes returned
to zero. The advertised text/image limits remain 45,056/45,056 for Full,
24,576/24,576 for 3070 Ti Flash and 65,536/24,576 for 2080 Ti Flash.

Two four-caller passes over the frozen 100-email dataset per service completed
600/600 calls successfully; all selected categories matched saved pre-fix model
results. Every second-pass call reported a prefix hit. Updated repeat totals
were 155.92 s for Full, 100.17 s for 3070 Ti Flash and 85.52 s for 2080 Ti Flash.
Historical comparisons are affected by retained cache contents and temperatures;
see the controlled measurement and limits in the linked accounting document.

Full also processed the two supplied original-resolution JPEGs together at
23,805 input tokens. The build took 45.87 s wall time; repeats took 2.05 and
1.94 s, both with image-containing prefix reuse, identical image-count answers
and no memory fallback. This is a functional long-image cache check, not a
controlled comparison of old and new image-inference speed. Warm processing
still included about 1.22–1.32 s of CPU image preprocessing.

## Flash optimized linear-attention kernels (4 October 2026)

A controlled comparison on the spare RTX 3090 switched native fallback and FLA
kernels on the same loaded FP16 compact NF4 Flash model. All 24 linear-attention
layers used FLA chunk/recurrent operations, causal-conv1d and gated normalization
in optimized mode. Full-attention SDPA, weights, image processing, memory budgets
and decision heads were unchanged. Ten input cases covered all native answer
types, exact 8K/12K/24K text, 256/512/1024 images, optional pooling, 16 images and
two original photographs. Selected answers matched; the maximum absolute numeric
answer difference was 0.0039. Compilation and per-case warm-up were excluded.

| Controlled spare-3090 task | Fallback | Optimized |
| --- | ---: | ---: |
| Short new text prefix | 273 ms | 129 ms |
| Short cached repeat | 296 ms | 139 ms |
| New 8,192-token text prefix | 3.17 s | 2.27 s |
| New 24,576-token text prefix | 11.37 s | 9.03 s |
| Two original photos, new 23,805-token prefix | 39.59 s | 34.48 s |
| Same original-photo prefix, cached repeat | 504 ms | 285 ms |
| Frozen 100 emails, first pass | 102.40 s | 72.20 s |
| Frozen 100 emails, repeat | 77.13 s | 58.11 s |

Per-request engine timings include preparation, forward/head and finishing,
excluding encoding and HTTP. Controlled email wall totals include encoding and
loop overhead and use the engine serially without HTTP queuing. Both email passes
kept all 100 choices; max probability difference was 0.0022. Agreement with frozen
assistant labels is not independently verified accuracy.

The pinned causal-conv1d source built successfully for SM75 using the CUDA
12.9.1 Ubuntu 24.04 recipe, with only architecture/compiler-thread build changes.
FP16 output, recurrent state, split-prefix and single-token continuation, and
convolution checks passed on SM86 and SM75; default BF16 checks passed on SM86.
The original Full environment and service were not replaced. Optional Flash
requirements and architecture-specific recipes are in
[optimized kernels](optimized-kernels.md).

Actual-GPU comparisons passed all ten cases on the 3070 Ti and 2080 Ti. On the
3070 Ti, new 8K text fell from 4.33 to 3.13 seconds and new 24K text from 15.20
to 11.74 seconds. Large prefixes were declined for retention by the unchanged
8 GB workspace budget; their repeat timings remain uncached. Short cached input
fell from 294 to 145 ms. Full FLA is enabled there.

The 2080 Ti showed a different crossover. After the user fixed cooling, three
interleaved cold measurements per path gave these medians:

| 2080 Ti new input | Native fallback | Full FLA | Native chunks + optimized conv/norm/recurrence |
| --- | ---: | ---: | ---: |
| Short | 228 ms | 202 ms | 230 ms |
| 8,192 tokens | 3.639 s | 4.983 s | 3.454 s |
| 24,576 tokens | 12.947 s | 14.470 s | 12.558 s |

All choices matched; max absolute numeric difference was 0.0018. Seven additional
mixed-path image, pooling, 16-image and cached-repeat cases passed. A separate
warmed FP16 GatedDeltaNet kernel comparison used 32 heads of dimension 128:
FLA was faster through 512 tokens (4.92 vs 7.42 ms at 512), approximately tied at
1K, and slower at 4K/8K (71.81 vs 44.91 ms at 8K). These microbenchmarks describe
one operation rather than complete request latency. Both implementations passed
output/state numerical checks at each tested length.

The 2080 Ti therefore uses an explicit adaptive chunk dispatch: FLA through 512
tokens and native chunks above that size, with optimized recurrence, convolution
and normalization. Short question suffixes and cached repeats can retain FLA's
benefit without using the slower large-chunk path. This threshold is measured on
this GPU/runtime and is configurable, not a claim about all architectures.

Production rollout preserved all three single-GPU services, token caps, queue
policy and cache memory reserves. The Full service was left running;
both Flash environments were upgraded separately with original environments and
adapter backups retained for rollback. Flash health reports selected chunk and
recurrent kernels and the adaptive threshold. Twenty live requests passed image
repeats, optional pooling, reordered media and concurrent FIFO calls, returning
to zero admitted requests and bytes.

The frozen 100-email task was rerun with four HTTP callers per Flash service:

| Service | Prior cached repeat | New first pass | New repeat | Repeat time reduction |
| --- | ---: | ---: | ---: | ---: |
| flash-3070ti | 100.17 s | 78.87 s | 75.54 s | 24.6% |
| flash-2080ti | 85.52 s | 86.61 s | 81.69 s | 4.5% |

All 400 requests succeeded, kept the previous category choices, and matched
95/100 frozen assistant reference labels in each pass. Token counts were identical.
These historical timings are observational and differ in compiler/cache history
and thermal conditions; the 2080 Ti cooling was fixed during this experiment.
The 3070 Ti shows the stronger email gain. No independently verified accuracy
claim is made. An additional 18 interleaved model trials passed the packaged
adaptive benchmark on the spare 3090, with max numeric difference 0.0014;
public fixed-mixed-path and kernel timing CLI modes were also exercised.

The final 2080 Ti adaptive API check processed both original JPEGs together at
23,805 input tokens: 41.33 s initial wall time and
2.01/1.93 s repeats. Both repeats reused the image-containing
prefix, kept the same image-count answer, and required no memory fallback. All
queues returned to zero. Kernel details are in
[optimized kernels](optimized-kernels.md).


## Unified startup and Docker — October 4, 2026

Full and Flash now use one serving module. GPU detection selects the tested
SM75 adaptive / SM86 FLA policy, Flash is the default, and `--full` selects the
larger model. A persistent cache supports pinned downloads, bounded single-GPU
NF4 preparation, locked atomic checkpoint publication and reuse on restart.
Docker/Compose use the same runtime with an SM75+SM86 convolution wheel.

Full on the 3090 and Flash on both the 3070 Ti and 2080 Ti passed container
text/vision/cache/pooling/16-image/context-error/queue checks. The 3070 Ti also
completed fresh Flash preparation and reused its checkpoint after restart.
The original services were restored after tests; a separate unified Flash demo
uses the previously spare 3090. See [detailed results](unified-service.md#validation).
No new email categorization or controlled throughput benchmark was performed.

## Expanded GPU targets — October 4, 2026

The convolution build now includes SM75/80/86/89/90/120. CUDA 12.9.1 compilation
and ELF inspection succeeded for all six targets, with wheel SHA256
`2406709c7d96b91abc1e82cbb547bfa834bd9bb860b59c3dbc4274e29c03727d`.
The installed bitsandbytes CUDA 12.8 binary was inspected separately and contains
SM80/89/90/120; Torch reports SM80/86/90/120, with SM86 binary compatibility on Ada.

FLA cross-compilation passed for A100 (SM80), RTX 4090 (SM89), H100 (SM90) and
RTX 5090 (SM120): 30 specializations per target, FP16/BF16, Flash/Full value-head
counts (32/48), initial and continuation prefill, recurrence and gated norm.
No target GPU kernels were launched, and these checks establish neither numerical
correctness nor throughput on those four cards. CPU policy tests cover both
profiles, A100 40/80 GB variants, missing cubins, reduced builds and independent
native-environment manifests. See [GPU support](gpu-support.md).

The final multiarch image passed numerical FP16 and BF16 kernel checks on the
3090, including continuation, single-token recurrence and convolution. Both
Full and Flash passed text/vision/exact-cache-reuse/pooling/16-image/>8K-prefill/
four-concurrent-call/error/discovery/UI checks in that image. Full advertised
45,056 input tokens on the test 3090. Application source hashes in the
container match the project source. No new target-card
throughput claims are made.

## Flash container on the RTX 2080 Ti — October 4, 2026

Clef Flash 9B moved from the native service to the unified Docker image on the
RTX 2080 Ti. The image was built with `CLEF_CUDA_ARCHES=75`, and native SM75
cubins were verified. The container sees only that GPU, at its existing 300 W
limit. It uses the existing compact NF4 checkpoint, mounted read-only, with FP16
compute, adaptive prefill (FLA up to 512 tokens, native PyTorch above that) and
optimized FLA recurrence.

Text, images, exact cache reuse, optional pooling, sixteen images, >8K prefill,
four concurrent queued calls, the overlength 413, request validation and a real
container restart all passed. Runtime source hashes matched the project.

The 100-email fixture then completed three passes (1, 4 and 4 outstanding calls)
with zero errors and 96/100 reference agreement in each. Times were 68.78, 65.50
and 67.53 seconds. The GPU reached 88 °C, with thermal slowdown in nine samples
late in the sustained workload. Every request body matched the earlier 3070 Ti
Flash run. See the
[initial results](../benchmarks/provider-comparison/results-flash-2080ti-initial-2026-10-04.json).

## Flash RTX 2080 Ti cooling rerun — October 4, 2026

With cooling restored, the same container and checkpoint completed three more
passes. Times were 65.22, 64.64 and 65.06 seconds (four-call mean 64.85 s).
Every pass kept 96/100 reference agreement, with zero errors or truncated
requests, and all 300 request hashes and choices matched the initial run.

Peak temperature fell from 88 °C to 70 °C and thermal-slowdown samples from nine
to zero. Mean GPU utilization was 94.2%, with about 6,384 MiB of GPU memory used
at the same 300 W limit. The serial set was 5.2% faster and the four-call mean
2.5% faster. Caches were not cleared, so extra warm cache state may also have
contributed. The service was healthy with an empty queue afterwards, and no
requests failed, timed out or were rejected. See the
[updated results](../benchmarks/provider-comparison/results-flash-2080ti-2026-10-04.json).

## Flash on the 400 W RTX 3090 — October 4, 2026

The RTX 3090 that served Full 27B was switched to Flash 9B, using the same
unified image and the cached pinned NF4 Flash checkpoint with FP16 compute and
SM86 FLA prefill and recurrence. The container sees only that GPU, at its
existing 400 W limit. Seventeen runtime and profile source hashes matched the
project. Text, vision, exact cache reuse, optional pooling, sixteen images, >8K
chunked prefill, four queued calls, discovery, the portal and the expected input
errors passed before benchmarking.

Three 100-email passes (concurrency 1, 4, 4) took 43.41, 42.31 and 42.67 seconds;
the four-call mean was 42.49 seconds. Each pass matched 96/100 reference labels,
with zero errors or truncations. Request hashes and all category choices matched
both earlier Flash runs, and probabilities differed by at most 0.18 percentage
points. Model loading and smoke checks were excluded; caches were not cleared.

Peak GPU temperature was 72 °C, memory use about 6,986 MiB and mean power draw
328 W. One software thermal flag was sampled at 68 °C, but a follow-up driver
report showed no active flag and zero accumulated slowdown. Memory temperature
was unavailable. This is recorded as a telemetry discrepancy, without diagnosing
overheating or a timing impact.

The queue was empty after the benchmark. Its one failed-request count came from
the intentional overlength API test beforehand. The Full container was stopped
and kept for rollback. See the
[results](../benchmarks/provider-comparison/results-flash-3090-2026-10-04.json).
