# Scheduling and batching experiments

Tested October 3, 2026. Every service and experimental process used one GPU.
No model was split across devices. These measurements describe the original
experimental queue/batching prototype. A bounded FIFO queue is now deployed in
both builds (October 4); dynamic GPU batching remains experimental. See
[production queue behavior](runtime-optimizations.md#request-queue).

The workload was the existing frozen 100-email dataset, complete email content,
unchanged category descriptions and classification guide. Each email remained
an independent record in Clef's native joint decision head. Reference agreement
means agreement with frozen assistant labels, not human-verified accuracy; the
guide was developed using this sample.

## Recommended settings from this workload

| Service | Single-client baseline | Recommended prototype | 100-email runtime | Reduction |
| --- | --- | --- | --- | --- |
| Full, RTX 3090 at 250 W | 321.0 s | Shared-guide prefix reuse, GPU batch 1 | 184.7 s | 42.5% |
| Flash, RTX 3070 Ti at 290 W | 198.8 s | Shared-guide reuse, batch at most 2, four outstanding clients | 105.6 s | 46.9% |

The Full benchmark used the spare 3090, not the live Full card at 400 W. Flash
was tested directly on its deployment's 8 GB 3070 Ti. Its normal service was
restored after the isolated benchmark. The recommended settings are for this
text workload; image batching has not been validated.

Both models preserved all 100 choices from their own single-record baselines.
Full matched 100/100 reference labels and Flash 95/100 throughout these checks.
Probabilities can change slightly; outputs are not bit-identical.

## What the queue prototype does

One GPU worker owns model execution. CPU encoding can run while that worker is
busy. A bounded queue holds at most 64 ready requests; a 5 ms collection window
allows small GPU batches. The oldest request anchors each batch, with nearby
lengths selected from the ready queue. A padded-token budget bounds
`batch_count * maximum_record_length`.

Full used a 12,000-token batch budget. Flash used 6,000, so this dataset could
never form more than two Flash records per GPU forward, even when the nominal
maximum was four. Longer records ran individually. Measured Flash batches
averaged 1.33 records with four clients and 1.41 with eight clients.

The design follows [NVIDIA Triton's dynamic batching guidance](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html),
but the prototype executes the existing native Clef model rather than deploying
Triton. Multiple Uvicorn model workers were not used.

## Shared-guide prefix reuse

The dataset has an exact common prefix of 1,568 tokens. The prototype prefills
that prefix once and retains its hidden states plus hybrid attention cache.
Every email's remaining tokens and native decision head are recomputed.
There is no answer-result cache.

Before each branch, the cache is cloned. Batches require independent copies of
attention keys and values, convolution state, and recurrent state. A helper
that only repeats KV tensors is insufficient for this hybrid backbone.
Exact token identity is checked before using the shared prefix.

The measured retained cache was 189 MiB for Full and 86.75 MiB for Flash. Its
one-time build took 1.67 and 1.33 seconds respectively, excluded from the warm
hit timings above. Adding that cost still gives about 42% and 46% less total
time for a cold 100-email run with the recommended configurations.

At the time of this benchmark the deployed cache retained only a complete exact
image/state prefix, so different emails missed it. The newer branching cache
implements exact-prefix checkpoint selection, explicit reusable context and
bounded eviction; see [runtime optimizations](runtime-optimizations.md). The
timings in this document remain results of the original queue prototype, not
measurements of the newer cache.

## Why higher concurrency is not always faster

Full's cached batch-1 run took 184.7 seconds with one client and 185.0 seconds
with two. Four clients and batch 2 took 210.0 seconds. Sixteen clients with a
nominal batch maximum of four took 184.1 seconds, essentially tied with batch 1,
but p95 request latency increased from 3.43 to 49.26 seconds.

Full queued batching without prefix reuse took 340.9 seconds with eight clients
and a batch maximum of four, versus the 321.0-second single-client control.
Its small-batch scheduling did not improve throughput on this workload.

For Flash, prefix reuse alone took 126.2 seconds. Four clients with batch 2
reduced that to 105.6 seconds. Eight clients took 103.6 seconds, only about 2%
less time, while p95 latency increased from 5.75 to 10.99 seconds. Queued
batching without prefix reuse took 185.5 seconds, versus the 198.8-second
baseline: the shared guide provides most of the combined improvement.

An offline Full sweep also demonstrated padding cost: batch 2 in arrival order
took 395.2 seconds versus a 340.5-second serial control; globally sorting by
length reduced it to 319.5 seconds. That sort can see the entire workload and
is not equivalent to an online scheduler. The later warmed HTTP serial control
also took about 321 seconds, so the small offline sorting speedup should not be
treated as an established improvement over warmed single-record execution.

## Scope and rollout

HTTP trials used server loopback and continuously maintained the stated client
concurrency. Latency includes server queueing, preprocessing and the response;
it excludes waiting for a client slot. Kernel/model warmups were excluded.
These were sustained runs without cooling pauses between requests. GPU busy
percentage, clocks, power, temperatures and memory were recorded for the HTTP
trials. Busy percentage does not measure SM occupancy.

The GPU batching prototype accepts text only. Integrating it into both builds
still requires preserving image, pooling and prefix-cache behavior and validating
mixed image/text batches. The deployed FIFO queue now supplies pending-call
cancellation, wait deadlines and bounded wire-payload admission; it preprocesses
one request at a time rather than holding multiple decoded image batches.
The source email dataset and private benchmark inputs remain outside the saved
project and are not included in its source archive.


## Production integration, October 5

[Opportunistic CUDA queue batching](runtime-optimizations.md#opportunistic-cuda-text-batching)
now implements the backlog-only text path with no collection timer, bounded
padding/memory admission and independent hybrid state. The default cap is two
on validated SM86 builds; four remains an explicit experiment. Images and long
prompts stay single-request operations. The preceding October 3 private-email
measurements are historical. The new [public synthetic 100-email results](../benchmarks/nvidia-batching/summary-2026-10-05.json)
use warmed HTTP passes and fixed per-card settings.
