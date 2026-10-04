# Independent input cache benchmarks

October 3, 2026. Both HTTP builds and all three live services updated.

Each row compares the same model on the same single GPU. Two private supplied photographs, three warm requests per timing, 256/512/1024 squared pixel budgets with aspect ratio and patch rounding preserved. Full: spare RTX 3090, 250 W; Flash: RTX 3070 Ti, 290 W. The live Full service remains on its own RTX 3090 at 400 W. Timings include in-process decode/processing/hash/collation/model work, excluding HTTP upload/parsing. Small samples, not confidence intervals.

| Model | Scenario | Image budget side | Before ms | After ms | Reduction |
|---|---|---:|---:|---:|---:|
| Full | changed context and order | 256 | 1093 | 792 | 27.6% |
| Full | identical prefix | 256 | 671 | 368 | 45.2% |
| Full | changed context and order | 512 | 1551 | 1193 | 23.1% |
| Full | identical prefix | 512 | 681 | 388 | 43.1% |
| Full | changed context and order | 1024 | 3349 | 2694 | 19.6% |
| Full | identical prefix | 1024 | 758 | 435 | 42.6% |
| Flash | changed context and order | 256 | 806 | 488 | 39.5% |
| Flash | identical prefix | 256 | 579 | 324 | 44.1% |
| Flash | changed context and order | 512 | 1031 | 683 | 33.7% |
| Flash | identical prefix | 512 | 596 | 296 | 50.3% |
| Flash | changed context and order | 1024 | 2263 | 1466 | 35.2% |
| Flash | identical prefix | 1024 | 642 | 325 | 49.3% |

Changed-context/order comparisons disable prefix reuse: uncached versus independent input reuse. Identical-prefix comparisons measure additional savings over the deployed prefix cache. CPU preprocessing for the two photos dropped from approximately 274–356 ms to 4–15 ms.

88 total checks across Full/Flash, including three exact processor-equivalence checks per model and 41 native probability/selected-option comparisons per model. No selected options changed. Full maximum probability delta: 3.81 percentage points for 16 duplicates, 0.28 outside that case. Flash maximum: 0.064 points. NF4 batch-shape kernel differences can alter probabilities; future near ties may change. Input reuse is independently disableable for the original batch reference.

All three deployed services passed 14 live HTTP checks apiece, including source validation, input cap, image order, partial replacement, fidelity, pooling, duplicate images and prefix layering. CPU tests verify LRU/caps, namespaces and immutability. Each service remains assigned to exactly one GPU.

Reproduce using `scripts/benchmark_input_cache.py full --output result.json` (or `flash`) in the serving environment with exactly one GPU selected. Synthetic images are the default; supply two private local image paths with `--images` if desired. Results omit images and source text. The optional file inputs expect JPEG data, matching the supplied original photos used here. Serving itself accepts PNG/JPEG/WebP.

See [runtime controls and semantics](runtime-optimizations.md). No new queue or microbatch scheduler is deployed in this change.
