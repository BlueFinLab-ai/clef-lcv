# Incremental image and text prefill

The new path is enabled on Full and both Flash deployments. It processes long
mixed requests with an 8K first language chunk and 4K continuations, retaining
all outputs for the unchanged native decision head. Short requests retain their
existing processing and cache behavior. Default fidelity remains 256², with
optional 2x2 pooling.

| Deployment | Text estimate at validation | Current image-request budget |
|---|---:|---:|
| Flash, 3070 Ti 8 GB | 24,576 | 24,576, increased from 8,192 |
| Flash, 2080 Ti 11 GB | 65,536 | 24,576 |
| Full, 3090 24 GB | 45,056 | 45,056, computed from memory |

Each image budget covers the complete encoded request, including context,
state, image tokens, questions, options, and template. `/v1/models` labels it
`configured_chunked_ceiling` for Flash. Full uses `computed_memory_estimate`
with `CLEF_MAX_IMAGE_LENGTH=auto`, following the same memory budget as text.
Overlength inputs return 413 without partial processing or truncation.

## What crosses a chunk boundary

The JPEG is decoded and each entire image is vision-encoded before language
prefill. Pixel tensors stay on CPU until the current image needs the GPU.
One-image vision batches bound temporary memory. Optional pooling then uses the
existing transformation and updates the grid consistently.

Spatial positions are computed for the whole mixed sequence. Each language
chunk receives its corresponding position slice and image-feature rows inserted
at the correct placeholder tokens. Boundaries may cut an image-token block.
Hybrid attention state continues across those boundaries. All language outputs
are preserved, and working state/features are released before the native head.
Neither the image nor any context is discarded at a boundary.

## Measured reference comparisons

Two user-supplied JPEGs were used at 256, 512, and 1024 fidelity. Additional cases
reverse their order, apply pooling, place images across the 8K boundary, and
repeat the pair to create sixteen high-fidelity images. Tests ask six visible
attribute/incident questions; no person identification is used.

The spare 3090 ran native and chunked inference using the same profile weights,
activation precision, processor, and native head. Flash then ran the same cases
on the 3070 Ti, with native local comparisons for short requests. Long requests
were compared against the spare-GPU reference because their native workspace
exceeds the smaller card's memory. Total token IDs were checked by hash.

The predeclared agreement gate requires unchanged selected answers and at most
1.0 percentage point of probability drift. All 21 Flash comparisons passed on
both GPUs. All 21 Full comparisons within its 16K ceiling passed, with at most
0.743 percentage points of drift. Tiny 64/128 and 512/512 chunks deliberately
stress image boundaries; they are not production settings.

| Same-GPU reference case | Native peak | Chunked peak | Native seconds | Chunked seconds |
|---|---:|---:|---:|---:|
| Flash, 24K, two high-fidelity images | 10.03 GiB | 6.81 GiB | 12.16 | 11.70 |
| Flash, 24K, sixteen high-fidelity images | 10.67 GiB | 6.92 GiB | 14.51 | 14.30 |
| Full, 16K, two high-fidelity images | 20.16 GiB | 19.08 GiB | 29.27 | 17.84 |
| Full, 16K, sixteen high-fidelity images | 20.86 GiB | 19.21 GiB | 19.98 | 21.06 |

Flash's 24K two-image peak fell by about 32%; the sixteen-image peak fell by about
35%. On the actual 3070 Ti, 24K with sixteen images completed in 18.96 seconds at
6.92 GiB peak allocation. PyTorch peaks exclude driver/context overhead and
unused allocator reservations. Timings are single observations and may include
shape-specific kernel warmup; they do not establish a reliable latency speedup.
The reference GPU's power setting also differs from production Full.

## Full's 24K gate and retained limit

Full completed the 24K two-image experiment with unchanged selected answers,
but probability drift reached 1.052 percentage points, exceeding the 1.0-point
gate. Its sixteen-image 24K comparison passed at 0.543 points. The cause of the
larger two-image drift has not been isolated between vision batching and language
chunking. At that initial deployment, Full's image ceiling was retained at 16,384 rather
than promoting 24K from memory fit alone. The existing long-context accuracy artifact remains
separate and unresolved.

A subsequent [Full context-quality investigation](full-context-quality.md)
isolated the vision-batching and language-chunking contributions. It reproduced
the numerical shift without changing selected answers and passed known-answer
tests through 45,056 tokens, including sixteen images. The 1-point gate alone
does not establish a quality boundary at 16K. That study did not change the
production image ceiling. The subsequent automatic-budget fix removed that
stale cap; Full now follows the computed limit, validated at 45,056 tokens.

## Deployment validation and reproduction

All three live services passed seven image API cases, including their image
ceiling and sixteen high-fidelity images. They retained the native selected
answers, respected the probability gate, returned 413 at precisely one token
over the image ceiling, and recovered short prefix-cache hits afterward. Text
regressions also passed, including the computed text ceilings and busy discovery.
All eighteen deployed file hashes matched; each service uses exactly one GPU.
The spare GPU was released after reference testing.

Enable with `CLEF_IMAGE_PREFILL=1` or `--image-prefill`; the portable default is
off. Full defaults to `CLEF_MAX_IMAGE_LENGTH=auto`; explicit numeric limits
remain available. On the 3070 Ti,
reverting to single-pass processing also requires restoring the image cap to
8,192. See [runtime controls](runtime-optimizations.md#incremental-image-prefill).

CPU semantics checks: `scripts/test_multimodal_prefill.py`. Isolated GPU
comparisons: `scripts/benchmark_multimodal_prefill.py`. Live image checks:
`scripts/test_multimodal_prefill_api.py`. These photograph benchmarks require the
private fixtures via `--images-dir`; original images are not bundled in Git.
The CPU test uses synthetic tensors and needs no photos, model weights, or GPU.

Raw measurements are kept outside this repository.

The subsequent [long-request cache update](long-request-cache.md) adds
memory-budgeted prefix and image-feature reuse to this path, removing the
previous 8K cache bypass. The Full RTX 3090 service is deployed and tested with that update.
