# Bounded text queue matching

`--batch-matching` enables an optional scheduler for independent text requests.
The oldest waiting request is always the anchor. While a backlog exists, the
single GPU worker can inspect up to eight compatible waiting requests and select
nearby lengths with reusable exact cache ancestry. There is no collection timer;
a lone request runs immediately. `--no-batch-matching` retains consecutive FIFO
pairing. Matching does not change model weights, source context, option order,
prefix identity, or answer selection.

```sh
python -m clef_service --model flash --batch-size 2 --batch-matching
```

Selection only prepares CPU inputs and reads cache metadata. It never restores
or evicts a prefix, allocates GPU tensors, or runs the model. The selected group
still passes the normal token, workspace, VRAM and exact-ancestry admission checks.
By default, an individual hit that reuses more tokens than the shared branch
remains a single request. Independent cached-history batching below optionally
groups those eligible hits, including CPU snapshots.

An image request, different model/cache controls, or a request that has already
been bypassed twice stops the search. A skipped request remains in its original
position and anchors a later dispatch. Deadline, cancellation, payload-byte and
shutdown limits remain active during selection. Wire admission is retained while
the CPU selector inspects a cancelled request's body; it is released after the
probe completes. Invalid neighbours and selector failures do not fail the anchor.

Prepared text inputs are reused when a skipped request reaches the worker. The
default conservative 128 MiB preparation estimate is divided across all allowed
queued/active requests, and also respects currently available host memory. Large
or unknown inputs decline this preparation path and use normal single processing.
This is an admission estimate, not an OS memory reservation.

| Option | Environment | Default |
|---|---|---|
| `--batch-matching` / `--no-batch-matching` | `CLEF_BATCH_MATCHING` | `0` |
| `--batch-lookahead` | `CLEF_BATCH_LOOKAHEAD` | `8`, valid 1–16 |
| `--batch-max-bypasses` | `CLEF_BATCH_MAX_BYPASSES` | `2`, valid 0–8 |
| Preparation estimate cap | `CLEF_BATCH_MATCH_PREPARE_MIB` | `128` |

Batch size remains independently controlled by `--batch-size` (1, 2, or 4).
Four requests must still fit the padded-token budget; raising the cap does not
guarantee a four-row GPU forward or a throughput improvement. Matching is opt-in
pending workload qualification. CPU preprocessing overlap is a separate option.

`/health.queue.matching` exposes bounded aggregate scan, selection, bypass and
failure counters. Response usage and metadata logs include `scheduler_matching_ms`,
`scheduler_candidates`, `scheduler_bypasses`, and `worker_handoff_ms`. Matching time
is attributed to the anchor once. GPU-window timing is shared by members of a GPU
batch; summing those request windows double-counts batch execution. Use elapsed
workload time and client latency percentiles for throughput comparisons.

The October 11 qualification replays 45 frozen Oida item-qualification HTTP bodies
with original context, state, questions, choice order and cache flags. Bodies and
raw responses remain in private development work. Aggregate results belong in
`benchmarks/nvidia-batching`; numerical consistency is distinct from model accuracy.


## Independent cached histories

`--independent-cache-batching` (or `CLEF_INDEPENDENT_CACHE_BATCHING=1`) extends
bounded queue matching to different retained text histories on CUDA Flash FP16.
It is opt-in; use `--batch-matching --batch-size 4` to group queued work. An idle
queue still dispatches one request immediately. GPU selection remains one GPU
per service. Full BF16, HIP, images, long/streamed requests and pending checkpoint
creation retain their existing paths.

Each row keeps its exact token ancestry, logical rotary positions, convolution
state, recurrent state and decision-head inputs. KV histories are left padded and
masked; query suffixes are right padded. Cached sources are leased, remain charged
to their GPU/CPU banks, and cannot be evicted during the batch. Packed final states
are discarded because the native recurrent kernel advances through padding; they
are never published as reusable checkpoints. The planner preserves every native
checkpoint cut. A request needing a new checkpoint falls back to the existing
shared or individual path.

Matching limits history padding to a 0.65 ratio and suffix padding to the configured
batch length ratio (default 0.75). Physical padded history plus suffix counts toward
`CLEF_BATCH_PADDED_TOKENS`. Workspace admission includes the working cache, hidden
vectors, attention mask and temporaries; retained leased sources are not counted
as reclaimable capacity. Aggregate failure releases leases and retries through the
existing smaller/individual paths without lowering single-request context limits.

`usage.prefill_mode=independent_cache_batch` identifies this path. Each response
reports its own `reused_prefix_tokens` and `prefix_cache_tier`. `cache_assembly_ms`
includes packing and any CPU-to-GPU copies; it is not a pure GPU-kernel measurement.
`gpu_time_ms`, `language_ms` and `head_ms` are shared batch windows, so do not sum
those per-member values as if they represented distinct work. `/health` reports
independent batch/request counts under `text_batching.independent_cache`.
