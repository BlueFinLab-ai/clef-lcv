# Decision service comparison

`scripts/benchmark_providers.py` sends the public
[100-email synthetic sample](../email-sample/README.md) to a local Clef service,
the hosted TypeSafe JEV API, or another local SystemOne-compatible decision
service, and records timing and agreement with the reference labels, which are
GPT Sol 6.1's categorizations of the same emails. Reference labels are never sent.

## Running it

Install the optional client dependency, then run from the project root:

```sh
python3 -m pip install httpx

# Local Clef, Flash or Full: use the model ID from /v1/models.
python3 scripts/benchmark_providers.py --provider clef \
  --url http://localhost:8080 --model clef-flash --trials 1,4,4 \
  --output local/benchmarks/flash.json

# Hosted JEV. Set JEV_API_KEY in your environment first.
python3 scripts/benchmark_providers.py --provider jev \
  --model jev-1.13.0 --trials 1,4,4 \
  --output local/benchmarks/jev.json

# Another local SystemOne-compatible service.
python3 scripts/benchmark_providers.py --provider systemone \
  --url http://localhost:8080 --model laya --trials 1,4,4 \
  --output local/benchmarks/laya.json
```

- **Trials:** each number in `--trials` is the client concurrency for one full
  100-email pass, run in order. `--limit 2 --trials 1` is a quick smoke test.
- **Keys:** the JEV key comes from `JEV_API_KEY`, or from an owner-only file passed
  with `--key-file`. Keep it outside the repository. Requests that carry a key go
  only to the official JEV endpoint, with TLS verified and redirects refused.
- **Output:** results are saved after each pass and never overwrite an existing
  file. They omit authorization headers and raw error bodies.
- **Errors:** there are no automatic retries. Errors are counted, and the command
  exits non-zero if any occurred. `completed` means all trials finished, not that
  every request succeeded.

CPU-only self-tests:

```sh
python3 scripts/test_provider_benchmark.py
python3 scripts/test_email_benchmark.py
```

## What is measured

- **Same task for every service:** each email is a separate request with the same
  category guide, category definitions and email fields. Clef receives the guide
  as reusable `context` with its cache options; JEV and generic services receive
  it in `state`. Every service returns a choice and a probability distribution,
  which the runner validates.
- **Client time:** includes network transport, queueing, inference and response.
  Hosted results include internet latency; local results include the LAN and the
  service's queue. Clef also reports its own processing and queue time.
- **Concurrency:** means outstanding client requests. Local services process them
  one at a time; this is not GPU batching.
- **Caching:** Clef's caches stay enabled and are not cleared between passes.
  Smoke tests ran before the measured passes, so results reflect a warm service,
  not cold starts. Other services control their own caching.
- **Agreement:** is measured against GPT Sol 6.1's labels for the sample, the
  quality baseline, not independently adjudicated ground truth. Two of the twelve categories
  (`work`, `other`) have no examples.
- **Rounding:** JEV returns probabilities rounded to two decimals. The runner
  accepts the resulting small sum error and keeps the reported values unchanged.

## Results: October 4, 2026

Each service completed three 100-email passes (concurrency 1, then 4, then 4)
with zero errors and no truncated inputs. Category choices did not change
between passes for any service. The four-call figure is the mean of the two
concurrent passes.

**Read these as observations, not a ranking.** The services run on different
GPUs, runtimes and settings, and some third-party models were run outside their
defaults (see the notes under the table). Clef Flash was measured on three GPUs;
see [Clef Flash on three GPUs](#clef-flash-on-three-gpus).

| Service | Hardware | Serial set | Four-call set | Matches GPT Sol 6.1 |
|---|---|---:|---:|---:|
| JEV 1.13.0 | Hosted | 18.16 s | 4.25 s | 100/100 |
| Clef Flash 9B | RTX 3070 Ti, NF4/FP16 | 59.98 s | 59.70 s | 96/100 |
| Clef Flash 9B | RTX 2080 Ti (300 W), NF4/FP16 | 65.22 s | 64.85 s | 96/100 |
| Clef Flash 9B | RTX 3090 (400 W limit), NF4/FP16 | 43.41 s | 42.49 s | 96/100 |
| Clef Full 27B | RTX 3090 (400 W limit), NF4/BF16 | 126.37 s | 126.20 s | 100/100 |
| Laya ¹ | RTX 2080 Ti (300 W) | 14.01 s | 13.26 s | 42/100 |
| Laya Typed Decisions ¹ | RTX 2080 Ti (300 W) | 14.07 s | 13.30 s | 36/100 |
| Decider 2B v11 ² | RTX 2080 Ti (300 W) | 48.34 s | 47.09 s | 94/100 |
| Kev-4B Q8_0 ³ | RTX 2080 Ti (300 W) | 129.18 s | 123.72 s | 96/100 |
| Kev-9B Q4_K_M, community ³ | RTX 2080 Ti (300 W) | 179.38 s | 170.26 s | 100/100 |

Notes on the third-party local models:

1. **Laya was run with longer inputs than it was trained for.** To pass every email
   in full, it used 8,192 state tokens and 4,096 head tokens, and the SDK's
   48-token-per-option cap was removed. These lengths exceed its published
   training and default lengths, so this is not representative of Laya at its
   default settings. Both Laya models shared one loaded service.
2. Decider used its pinned FP16 eager runtime with efficient SDPA and no CUDA graphs.
3. Kev used native llama.cpp decision binaries: all layers on the GPU, 8,192-token
   context, one slot, FP16 KV cache, flash attention off and context shifting
   disabled.

For all three, a benchmark-only wrapper queued overlapping requests instead of
rejecting them; it adds no batching. The runtimes are not equally optimized, so
timings across different models aren't directly comparable, even on the same GPU.

The Laya, Decider and Kev rows come from a rerun after extra cooling, which peaked at 75 °C
with no recorded thermal slowdown. An earlier run reached 87–90 °C and
throttled. Its choices were identical, but Kev was slower (139.27 s and
202.75 s serial). That run is kept in
[results-expanded-initial-2026-10-04.json](results-expanded-initial-2026-10-04.json).

## Clef Flash on three GPUs

The same Flash model (NF4 weights, FP16 compute) ran the same 100 emails on three
GPUs, using the same Docker image contents. Each run completed three passes
(concurrency 1, 4, 4) with zero errors and no truncated inputs.

| GPU | Power limit | Serial set | Four-call set | Matches GPT Sol 6.1 | Peak temperature |
|---|---|---:|---:|---:|---:|
| RTX 3090 | 400 W | 43.41 s | 42.49 s | 96/100 | 72 °C |
| RTX 3070 Ti | Default | 59.98 s | 59.70 s | 96/100 | No slowdown recorded |
| RTX 2080 Ti | 300 W | 65.22 s | 64.85 s | 96/100 | 70 °C |

- **Same answers on every GPU:** every request body was identical across the
  three runs, and all 100 category choices matched in every pass. Probabilities
  differed by at most 0.18 percentage points.
- **Speed:** the 3090 finished the serial set about 1.4 times faster than the
  3070 Ti; the 2080 Ti was about 8% slower than the 3070 Ti. The 2080 Ti uses
  adaptive kernels (optimized up to 512 tokens, native PyTorch above that).
- **2080 Ti cooling:** the first 2080 Ti run reached 88 °C with nine
  thermal-slowdown samples and took 68.78 s serial (66.52 s four-call). After
  cooling was restored it peaked at 70 °C with no slowdown samples. Choices were
  identical; caches stayed warm between runs, which may also have helped.
  The first run is kept in
  [results-flash-2080ti-initial-2026-10-04.json](results-flash-2080ti-initial-2026-10-04.json).
- **3090 telemetry:** one software thermal-slowdown flag was sampled at 68 °C,
  but the driver's follow-up report showed no active flag and zero accumulated
  slowdown. It's recorded as a telemetry discrepancy, not overheating.
- **Warm service:** caches stayed enabled and smoke checks ran before timing,
  as in the other runs. Concurrency means queued requests, not GPU batching.

## Files

| File | Contents |
|---|---|
| [results-2026-10-04.json](results-2026-10-04.json) | JEV, Clef Flash and Clef Full: per-pass timings, latency, Clef processing and queue time, pairwise agreement, GPU telemetry samples |
| [results-expanded-2026-10-04.json](results-expanded-2026-10-04.json) | The same plus the five 2080 Ti models after cooling |
| [results-expanded-initial-2026-10-04.json](results-expanded-initial-2026-10-04.json) | The earlier, thermally throttled 2080 Ti run |
| [results-flash-2080ti-2026-10-04.json](results-flash-2080ti-2026-10-04.json) | Adds Clef Flash on the RTX 2080 Ti after cooling, with the 3070 Ti comparison |
| [results-flash-2080ti-initial-2026-10-04.json](results-flash-2080ti-initial-2026-10-04.json) | The earlier, thermally throttled Flash run on the 2080 Ti |
| [results-flash-3090-2026-10-04.json](results-flash-3090-2026-10-04.json) | Adds Clef Flash on the RTX 3090, with comparisons against both other Flash runs |
| [test-list.json](test-list.json) | Services, pinned checkpoints and hardware |

The `raw_results` names in these files refer to per-request output files, which
are not included; the summaries above are derived from them.
