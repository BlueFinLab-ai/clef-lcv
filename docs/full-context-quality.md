# Full 27B context quality investigation

## Conclusion

The original 24K warning was a repeatable numerical difference, not an observed
wrong answer. Controlled experiments isolate contributions from both vision
batching and language chunking. The maximum combined shift was 1.052 percentage
points on a text urgency question; its selected answer remained correct.

The expanded suite produced **120/120 correct known-answer checks**
across 26 successful trials, including **102/102
with chunked processing**. Both tested chunk layouts passed through **45,056 total
encoded tokens**, including two and sixteen high-fidelity images. This supports
45,056 as a tested capacity candidate on this 3090; it does not establish a
universal accuracy guarantee for all long documents.

Production configuration was not changed during that investigation. At its end, Full
advertised a computed text budget (45,056 at the previous idle validation) and a
configured 16,384-token image ceiling. The previous 1-point probability cutoff
alone is not evidence that 16K is a quality boundary. Those tests supported increasing the image budget. The subsequent deployment
removed the stale image cap with `CLEF_MAX_IMAGE_LENGTH=auto`; Full now follows
the computed memory budget, validated at 45,056 tokens. See
[automatic image admission](runtime-optimizations.md#context-budget-discovery).

## Controlled 24K experiment

Same NF4 checkpoint, BF16 compute, efficient SDPA, fast linear attention, native
decision head, GPU and input. The two original photographs were processed at
1024 fidelity. Image features were frozen on CPU and reused verbatim to isolate
language changes. Persistent prefix and image-feature caches were disabled.

| Change relative to native single pass | Largest probability change | Selected answers |
|---|---:|---|
| Three repeated native runs | 0.000 pp | Identical |
| Frozen batched features, whole language pass | 0.000 pp | Identical |
| One-image vision batches, whole language pass | 0.415 pp | Identical |
| Identical native image features, 8K + 4K language chunks | 0.461 pp | Identical |
| One-image batches plus 8K + 4K chunks | 1.052 pp | Identical |
| Repeat of combined path | Same 1.052 pp | Identical |
| Identical native features, 4K + 4K chunks | 0.287 pp | Identical |
| Identical native features, 8K + 2K chunks | 0.492 pp | Identical |

The effects interact; their maximum differences should not be added as if they
were independent errors. For the combined path, urgency probability moved from
78.75% to 79.80%, retaining “Today.” The largest shift among the five visual
questions was only 0.175 points. Repeated batched vision features were bitwise
identical. Changing vision batch size produced feature RMS difference 0.01470;
we have not attributed that difference to a particular kernel or arithmetic
operation. Low-precision, shape-dependent numerical effects are a plausible
explanation, not a proven kernel-level diagnosis.

These results measure implementation agreement. Unchanged choices do not prove
unchanged calibration, and a close decision on another input could be more
sensitive. The 1-point gate remains useful as an investigation trigger rather
than a standalone maximum-context rule.

## Known-answer length tests

Varied synthetic operational log entries surround explicitly named facts at the
beginning, middle and end. Text-only requests ask three fact questions. Image
requests add three questions about the first photo: scene, clothing color and
presence of a cruise ship. At the maximum length a second variant rotates the
facts and reverses the photos, preventing success through a fixed answer pattern.
The sixteen-image case repeats the supplied pair. All image cases use 1024
fidelity, no pooling, and one-image vision batches.

Two chunk layouts were tested: production 8K first + 4K continuation, and 4K
throughout. The following table shows the production layout. Token counts
include context, state, images, questions, answer options and template.

| Total tokens | Input | Correct answers | Peak allocated | Time |
|---:|---|---:|---:|---:|
| 8,192 | Text only | 3/3 | 19.03 GiB | 7.78 s |
| 8,192 | 2 images | 6/6 | 19.05 GiB | 8.37 s |
| 24,576 | Text only | 3/3 | 19.92 GiB | 29.77 s |
| 24,576 | 2 images | 6/6 | 19.94 GiB | 30.28 s |
| 32,768 | Text only | 3/3 | 20.78 GiB | 42.01 s |
| 32,768 | 2 images | 6/6 | 20.80 GiB | 43.20 s |
| 45,056 | Text only | 3/3 | 22.07 GiB | 64.88 s |
| 45,056 | 2 images | 6/6 | 22.09 GiB | 65.58 s |
| 45,056 | Text only; changed facts / reversed photos | 3/3 | 22.07 GiB | 64.95 s |
| 45,056 | 2 images; changed facts / reversed photos | 6/6 | 22.09 GiB | 65.55 s |
| 45,056 | 16 images | 6/6 | 22.22 GiB | 66.34 s |

Native single-pass references were correct at 8K and 24K. Both native 32K
attempts ran out of memory; chunked requests completed. Consequently, results
above 24K demonstrate known-answer correctness and agreement between chunk
layouts, not equivalence to an unchunked reference that could not fit.

Times are single observations on the spare 3090 at 250 W. They are not a
controlled speed comparison with the production GPU at 400 W. Allocated memory
excludes CUDA context/driver overhead and unused allocator reservations.

## Using the maximum length

- Treat `/v1/models` as the current **admission budget**, including its separate
  `max_input_tokens_with_images`; the experimental maximum does not override it.
- Count the complete encoded input. Raw text token counts alone omit image tokens
  and question/option/template overhead. The server rejects over-budget input
  with 413 rather than truncating it.
- Keep memory capacity and task accuracy separate. This study found no Full 27B
  retrieval failure through the tested 45K ceiling, but broader documents and
  tasks need their own quality checks.
- The earlier wrong-answer long-context artifact was observed in **Flash** with
  text-only repeated-word inputs. This Full study does not resolve that artifact.

## Reproduction and evidence

Run `scripts/benchmark_full_precision_ablation.py` and
`scripts/benchmark_full_context_quality.py` on an otherwise free single 24 GB
GPU. Both accept `--data-dir`, `--images-dir`, `--gpu` and `--output`; they use the
Full profile by default. The source JPEGs are private fixtures and are not
included in the project or source archive. Their names match the earlier image
benchmark. Runtime source was commit `45d17cb`; this study changes no runtime
implementation or model weights.

Raw observations are kept outside this repository.
