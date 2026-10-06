# RX 580 text batch-size experiment

This is an isolated backbone/decision-head prototype, **not a production queue
or dynamic batching implementation**. It uses eight public synthetic emails,
a prebuilt exact shared prefix, right padding, per-row hybrid recurrent/KV state,
and the existing decision head. Reference labels are recorded for checking but
are not passed to the model. Model load, CPU encoding, prefix construction and
shape warm-ups are excluded from measured totals.

The measured settings are native GDN block 64, FP16 compact NF4, math SDPA,
SDMA off and the pinned offline rocBLAS choices. Prefix construction uses
512-token chunks in both experiments. The changed parameter is the language
**tail chunk size**, independent of the GDN inner block.

Run only with exclusive GPU access, stopping and later restoring the normal
service. The commands below assume a prepared Flash checkpoint, the validated
community HIP image, and host video/render groups 44/991; adjust groups for the
host. Mount `benchmarks/email-sample/` read-only at `/bench`, this directory at
`/experiment`, and a writable output directory at `/results`:

```sh
docker run --rm --init --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 991 -e HIP_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=4 \
  -v /path/to/model-nf4-compact:/checkpoint:ro \
  -v /path/to/Clef/benchmarks/email-sample:/bench:ro \
  -v /path/to/Clef/experimental/rocm:/experiment:ro \
  -v /path/to/results:/results --entrypoint python \
  clef:lcv-rocm-startup-20261004 /experiment/batch-sizes.py \
  --chunk-tokens 512 --output /results/batch512.json
```

The 512-token trial failed during the first batch-four warm-up. Its previously
completed serial and batch-two rows remain valid; the planned return controls
were not reached. An OOM returns nonzero and preserves completed measurements.
The isolated test process exits, releasing its allocations before the normal
service is restored. No memory-overflow retry or batch splitting is implemented
by this script.

The follow-up uses the saved serial reference and smaller tail chunks:

```sh
# Same Docker devices, environment and mounts as above.
python /experiment/batch-sizes.py --chunk-tokens 256 \
  --batch-sizes 2 4 4 2 --serial-reference /results/batch512.json \
  --output /results/batch256.json
```

Both runs use the same eight-record selection and grouping order. The first
experiment's controls are single measurements; the second brackets two
batch-four repeats with batch-two controls. Compare batch four both against
its matching chunk-size control and against the fastest batch-two configuration.
Mean group completion is a latency observation, not HTTP timing or queue wait.
Torch peak allocation excludes external HIP/BLAS/driver memory, so it alone
cannot predict whether a batch fits the physical card.

[512-token raw results](batch4-results-2026-10-05.json).
[256-token paired results](batch4-chunk256-results-2026-10-05.json).

Measured totals for eight emails: serial 124.33 seconds; batch two with 512-token
chunks 108.63 seconds; batch two with 256-token chunks averaged 164.34 seconds;
batch four with 256-token chunks averaged 134.28 seconds. Batch four was 18.3%
faster than its matching smaller-chunk control, but 23.6% slower than the best
batch-two configuration. Both batch-four runs encountered allocator allocation
failures and recovered. Their mean group completion was 67.14 seconds, compared
with 27.16 seconds for the faster batch-two path. All completed categories agreed
with serial; the maximum probability difference was 0.0018 (0.18 percentage points).
Peak Torch allocation was 5.44 / 5.90 GiB for serial / batch two at 512 tokens,
and 5.68 / 6.38 GiB for batch two / four at 256 tokens. External HIP/BLAS memory
is excluded. This is not a basis for enabling batch four universally on an 8GB card.
The serving app remains a single-GPU FIFO worker with production batching off.

[Restored portal and allocator-warning checks](batch4-validation-2026-10-05.json).
