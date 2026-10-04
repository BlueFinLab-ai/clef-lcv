# Long-request cache validation — October 3, 2026

The fixed 8,192-token GPU-cache bypass is removed from both HTTP adapters.
Full Clef 27B on the primary RTX 3090 received the initial runtime on October 3.
Both Flash services received the adaptive implementation on October 4 and passed
live long-cache pressure/recovery tests below. Context admission limits remain
unchanged. Earlier sections preserve the policies measured at each stage.

## Live Full measurements

Single-GPU RTX 3090, 400 W configured limit; NF4 weights, BF16 compute,
efficient SDPA and fast FLA. Timings below are server latency including
preprocessing. The eleven-image fixture repeats the two supplied photographs;
it is not eleven distinct photographs. It contains 12,288 encoded tokens and
six questions about visible scene/clothing attributes and text urgency.
No face identification or matching was tested.

| Request | Server time | Result |
|---|---:|---|
| All caches disabled | 15.256 s | Reference |
| Cold GPU prefix, CPU input cache warm | 11.473 s | Two checkpoints retained |
| Repeat, median of three calls | 0.742 s | 11,917 prefix tokens reused; vision skipped |
| Changed question, prefix disabled | 10.869 s | Independent image features reused |
| Changed question, prefix enabled | 0.802 s | Same 11,917-token state reused |
| Reordered photographs | 10.935 s | Language miss; 11 feature hits, zero misses |
| Pooling toggled, held at 12K tokens | 11.430 s | Language miss; raw features reused |
| 45,056 tokens / 16 image slots | 53.373 s | Cache evicted for workspace; all known answers passed |
| Repeat 12K after maximum request | 0.756 s | Cache rebuilt and hit again |

The repeat median is **15.5× faster** than the cold GPU-prefix call
(93.5% less server time). Client wall time was 11.625 s cold
and 0.909 s warm median, including upload and HTTP overhead. Cold/warm and
changed-question/its-own-uncached-reference probabilities were identical in the
recorded responses. Selected answers passed all fixture truth checks. The maximum
request peaked at 22.36 GiB.
45,057 tokens returned HTTP 413; discovery remained at 45,056. Short text cache
recovery passed too. No OOM fallback occurred.

## Cache behavior and limits

The adapter reserves calibrated workspace for the complete request, then trims
GPU retention to fit. In the initial deployment, at 12K the live pool budget was
about 1.74 GiB; at 45K it was zero. The October 4 elastic policy below supersedes
that retention budget. The latter is a memory decision, not a new fixed token cutoff. Retained
entries are process-local and bounded by VRAM and entry limits. New checkpoints
clone mutable hybrid state and copy only their completed hidden prefix. The
native decision head still evaluates the complete request; answers are not cached.
The first request builds state and later exact-prefix requests reuse it.

Image order, fidelity, model configuration and pooling participate in language
cache identity. Independent raw image features can survive reorder/pooling changes,
subject to eviction, but changed fidelity needs new features. Prefix checkpoints
never end midway through the media region. The long path transfers pixels only
for missing features, one complete image at a time. Language checkpoints are
removed before small independent features under pressure. OOM recovery retains
the existing retry with GPU caching disabled. Restart clears all entries.

Usage now includes `cache_memory_budget_mib` and `cache_limit_reason` on the
incremental path, alongside reused token counts, feature hits and vision timing.
Explicit API cache flags and the global memory/entry controls remain available.
This is exact-prefix reuse with copied state, not paged attention or batching.

## Verification and reproduction

An isolated 3090 at a 250 W limit first compared the candidate against the old
uncached implementation. Selected answers matched; the largest probability shift
was 0.174 percentage points. Candidate cold/warm probabilities were identical.
Its warm GPU-processing calls took 0.702–0.703 s versus 12.925 s cold. These are
separate from the live 400 W service timings above. Reordered images, changed
questions, pooling, maximum context and post-pressure recovery passed there too.
The tests are targeted fixtures, not a broad accuracy or eleven-unique-photo study.

CPU checks cover global multimodal positions, pooling, checkpoint continuation,
independent branch state and complete-image-prefix vision skipping. Prefix-index
and feature-index checks passed. Six deployed file hashes matched the install
manifest. Each running service remains on its configured single GPU.

Run `scripts/test_multimodal_prefill.py`, `scripts/test_prefix_index.py` and
`scripts/test_image_feature_index.py` for CPU semantics. Model benchmarks use
`scripts/benchmark_long_cache.py`; live validation uses
`scripts/test_long_cache_api.py`. Both GPU scripts require the two private photos
via `--images-dir`. Original photos and encoded request payloads are not bundled.
To compare against an older `runtime/chunked_prefill.py`, pass it with
`--baseline-source`.
See each script's `--help` for model and output paths.

Measurements are kept outside this repository.
Per-file rollback copies were preserved on the server before deployment.

## Elastic GPU cache — October 4, 2026

The first elastic deployment targeted the remaining memory of the Full RTX 3090, subtracting model and other
allocations, a fixed 1 GiB reserve, and active request workspace. Idle workspace
is released; the pool grows on demand and shrinks through eviction before larger
requests. Existing working tensors count toward projected peak workspace, so
snapshot admission no longer reserves that workspace twice. A branch clone is
part of request workspace; retained snapshot copies remain additional allocations.
At that stage Flash's live cache policy was unchanged. Entry limits and independent CPU/feature
cache caps still apply. This does not increase the advertised context limit.

Two original private JPEGs produced 23,805 input tokens. The previous live
policy rejected their language checkpoint and repeat server times were 24.291
and 24.731 s (24.511 s median). With elastic caching on the same live RTX 3090,
three repeats took 2.484, 2.103 and 2.548 s (2.484 s median), a 9.9× improvement;
client wall median was 2.781 s. The stored prefix reused 23,729 tokens and skipped
GPU vision encoding. Recorded cold/warm answers and probabilities were identical.
The new post-restart cold call took 47.309 s because vision features also had to
be rebuilt; it is not comparable to the old repeats with warm vision features.

The idle GPU cache budget was 5,456 MiB (5.33 GiB), versus about 2,322 MiB during
this request. Actual retained occupancy was about 2,022 MiB after the two-photo
call. No unused cache memory is preallocated. Warm CPU preprocessing still
missed for these large originals, taking roughly 1.3–1.8 s per request; that
separate 256 MiB CPU cache was not changed. Input token counts remain the complete
logical request size, even on cache hits.

A second, isolated RTX 3090 at 250 W compared the old 90% target, the old accounting
at 100%, and the elastic policy. Raising the target alone still rejected the
snapshot for transient headroom. Elastic retention succeeded. A 45,056-token
request evicted the language snapshot, completed without OOM, and a subsequent
smaller request rebuilt it and regained a hit with identical recorded answers.
These isolated calls do not use independent image-feature keys; their roughly
55 s uncached times include repeated vision encoding and must not be mixed with
the live API baseline. Changed questions and short text cache reuse passed live.

Run `scripts/test_elastic_cache.py` for CPU checks of idle expansion, active
workspace, snapshot/branch copies, eviction, other device users, fixed budgets
and legacy behavior. Prefix and image-feature index regressions also passed.
Deployment kept per-file rollback copies.

## Small-GPU cache margins — October 4, 2026

An isolated Flash NF4 compact run on the actual 8 GB RTX 3070 Ti compared
elastic retention margins. Production Flash was stopped for the test and
restored with its original configuration afterward. These measurements do not
deploy elastic caching to Flash. The context admission reserve stayed at 512 MiB.

| Cache margin | Repeated 8,192-token processing | 24,576-token pressure |
|---|---:|---|
| 1,024 MiB | 4.565 s; reuse disabled after observed workspace | Not tested in this comparison |
| 512 MiB | 4.630 s; checkpoint not retained | 15.357 s; no retry |
| 256 MiB | 0.325 s; cache hit | 30.786 s; succeeded after cache-clear retry |

Times include direct model preprocessing/GPU work and exclude HTTP transport.
The 1 GiB comparison covers only the cold/repeated 8K pair. The 256/512 MiB
comparison additionally ran eight distinct 4K inputs, 24K pressure, 8K recovery,
and two private photographs downscaled to a 1024-pixel edge. Both photo repeats
hit the prefix cache (0.364/0.348 s respectively), with the same selected count.
Text cold/warm selected actions agreed; this is not a broad accuracy evaluation.

The 8K snapshot was about 343 MiB, while observed request workspace was about
1,989 MiB. A 1 GiB margin left no active retention budget. At 256 MiB, idle pool
budget was about 2,372 MiB, and the smaller active pool could retain that snapshot.
The warm 8K call was about 14× faster than the 1 GiB repeat. Idle budget remains
a capacity limit rather than a promise of that much retained occupancy.

The first 256 MiB pressure test, after retaining about 366 MiB of language
checkpoints, failed a 190 MiB CUDA allocation. The error reported 131 MiB device
free and 312 MiB of PyTorch reserved-but-unallocated memory. Repeating with the
HTTP adapter's cache-clear/uncached-retry behavior completed the same request.
Subsequent 8K calls rebuilt and hit again. The 512 MiB run avoided the allocation
failure but lost the 8K checkpoint beside its conservative high-water workspace.
These tests do not isolate allocator fragmentation from other transient effects.

This experiment recommended a 256 MiB cache margin for idle/smaller requests and
at least 512 MiB effective headroom near long-context pressure, with proactive
eviction preferred to redoing a failed prefill. The adaptive policy was not part
of that comparison; it is implemented and validated in the next section. A flat 256 MiB setting is an aggressive
option with the existing retry; zero reserve was not tested. Model memory and
projected request workspace must still be accounted for independently.
Raw records are kept outside this repository.


## Adaptive cache headroom — October 4, 2026

Implemented and deployed on the Full RTX 3090 and both Flash services. Both HTTP builds
now target 100% of usable device memory after model/other allocations, active
workspace and extra headroom: 256 MiB idle/short-call reserve; at least 512 MiB
when a calibrated incremental request declares projected workspace. A configured
larger minimum is preserved. This does not raise context admission limits.

Before inference, projected workspace trims the shared language/feature pool.
After an eviction for projected prefill, `empty_cache()` returns unused allocator
segments to CUDA. It does not discard retained live tensors, and is not called
on every request. Idle completion releases the workspace reservation even on
exception paths. The existing cache-clear/uncached retry remains available for
unexpected memory pressure, but none of these validation requests used it.

| Live service | Text / mixed-image limit | Maximum text server time | Evicted tensors | Cache preparation | Repeated 8K server time |
|---|---:|---:|---:|---:|---:|
| Flash, 3070 Ti 8 GB | 24,576 / 24,576 | 15.727 s | 366.4 MiB | 90.0 ms | 0.341–0.380 s |
| Flash, 2080 Ti 11 GB | 65,536 / 24,576 | 44.483 s | 1,808.8 MiB | 299.4 ms | 0.279–0.281 s |
| Full, RTX 3090 24 GB | 45,056 / 45,056 | 50.864 s | 3,603.4 MiB | 433.6 ms | 0.301–0.316 s |

Preparation includes allocator release: 60.8 ms on the 3070 Ti, 59.2 ms on the
2080 Ti and 120.0 ms on Full. The 3070 Ti's 15.727 s pressure call avoided the
retry observed with a flat 256 MiB margin (30.786 s in the prior isolated run).
These are separate trials, not a statistically controlled latency comparison.
The direct isolated adaptive test corroborated the same case at 15.741 s with
94.8 ms preparation and no retry. Its Full 250 W counterpart passed at 59.590 s.

Every live service passed eight distinct 4K inputs to fill retention, maximum
text processing, 8K cache recovery, two-image repeats, maximum mixed requests
with sixteen image slots, and image cache recovery. Known two-image counts and
maximum mixed scene choices were correct; tested cold/warm choices matched with
probability deltas no greater than 0.2 percentage points. Sixteen slots repeat
the two private fixture photos. This is a cache/admission regression, not a broad
accuracy evaluation, and does not resolve the recorded Flash long-context
accuracy artifact. Text inputs one token over the advertised limit returned
413 with the exact limit; discovery limits remained unchanged after pressure.

Full also retained both original photographs at 23,805 tokens after pressure.
The first call took 45.422 s; three repeats took 2.296, 2.480 and 2.185 s
(2.296 s server median), reused vision/prefix state and returned identical answers
and probabilities. A changed question hit the same prefix. Large original CPU
preprocessing remains a separate cost and its capacity is unchanged.

Post-test idle shared pool limits were about 2,372 MiB on the 3070 Ti,
5,370 MiB on the 2080 Ti and 6,224 MiB on Full. These are on-demand capacity
limits, not retained occupancy or preallocation promises. PyTorch may retain
unused allocator blocks, so `nvidia-smi` usage can be much higher than live cache
occupancy. `/health` reports both the shared pool and feature occupancy. The
32-checkpoint, 256 MiB feature and 256 MiB CPU image-cache limits remain in force.
Maximum-sized requests may consume most capacity and retain no checkpoint.

Controls: `CLEF_PREFIX_CACHE_ELASTIC=1`,
`CLEF_PREFIX_CACHE_GPU_UTILIZATION=1.0`,
`CLEF_PREFIX_CACHE_RESERVE_MIB=256`,
`CLEF_PREFIX_CACHE_PREFILL_RESERVE_MIB=512`. Usage reports preparation cost,
evicted tensor bytes, effective headroom and allocator-release time. The prefill
margin applies when positive projected workspace is supplied; uncalibrated
custom builds without projection still use observed short-call accounting.

Reproduce with `scripts/benchmark_adaptive_cache.py` (isolated model) and
`scripts/test_adaptive_cache_api.py` (live API). CPU checks passed elastic
accounting/selective allocator release, prefix and feature indexing, inference
helpers and context estimates. Project packaging checks passed 72 files.
Six deployed hashes per service matched the saved source and install manifests;
all three units are active on their configured single GPU.
Temporary remote photo copies were removed after testing; saved records contain
metrics and model answers, never image payloads or private prompt bodies.

Each service kept per-file copies of its previous runtime for rollback. Flash
and Full runtime snapshots differ, so each service restores its own copies.
