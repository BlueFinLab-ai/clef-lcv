# Cold-input throughput benchmark

The portal's **Cold-pass benchmark** checkbox runs the current input once without
reusing CPU preprocessing, text tokens, image features or language-prefix state.
It keeps the model loaded and does not flush shared caches or restart the service.
A new input may still pay first-use kernel/shape setup. Use the CLI's untimed
warmup and repeated passes for GPU comparisons.

The API payload stays compatible with `/v1/systemone`. Send the optional header:

```http
X-Clef-Benchmark: cold
```

The service forces `prefix_cache:false` and `input_cache:false` for this request.
It uses the existing authenticated, bounded queue, with a GPU batch size of one
and no short-request interleaving inside its backbone pass. Other callers can
still be queued before or after it. This is **cold input**, not cold model startup.
Request workspace pressure can cause the normal cache eviction/offload policy;
the benchmark never issues a global cache clear.

`usage.cold_benchmark` reports:

- Actual processed language tokens and language-forward count, including chunks.
- `language_window_ms` and `language_tokens_per_second`: device-stream windows
  around the language-model forward calls, excluding vision and the decision head.
- `vision_window_ms` and `head_window_ms` when those modules are called directly.
  A CPU-prepared head may use its methods without a module call; in that case
  `head_timing_complete:false` and the head-window value is absent/null.
- `gpu_window_tokens_per_second`: input tokens divided by the existing total
  inference window, which also includes vision, head and other in-model work.
- `valid` and an explicit reason when reuse, a retry or mismatched processed-token
  accounting makes the rate unsuitable. An invalid rate is never reported as zero.

Device events include launch/transfer delays; these are not measurements of pure
GPU-busy time or core occupancy. Event hooks are installed only for benchmark
requests and removed in `finally`. Phase events are read after the existing total
inference timer has synchronized; no extra phase-boundary synchronization is added
to ordinary requests. CUDA and HIP use the existing PyTorch device-event API.

The portal also reports end-to-end input tokens/s using its browser timer, including
image preparation, upload, queue wait and reply transit. Text tokens, image tokens,
model size, precision and residency strategy affect throughput. Record them and
compare like workloads; a larger model's rate is not a GPU-only comparison.

## Compare services from the command line

No extra Python dependencies are required:

```sh
python scripts/benchmark_throughput.py \
  --url http://192.168.1.181:8070 \
  --url http://192.168.1.181:8071 \
  --words 1024 4096 --warmup 1 --repeats 3 \
  --output ./cold-throughput.json
```

`--words` sets synthetic text sizes in words, not an assumed token count. Rates
always use the service's actual `input_tokens`. For representative data, use
`--request-file`: one standard request, an array of requests, or a saved fixture
with `rows[].request`. The CLI discovers the model separately for each endpoint;
`--model` can override it. Input content/question/option ordering is preserved.

Use `CLEF_API_KEY` or `--api-key-file` for protected services. Credentials, prompts,
images and answers are omitted from the JSON report. The report includes GPU/model
metadata, workload hashes, actual tokens, timing samples, queue wait, strategy and
peak GPU allocation. It reuses an HTTP connection, does an untimed uncached shape
warmup, and runs serially. Each pass recomputes every token, even when content is
identical. Shared caches are not cleared.

Aggregate rate is **sum of actual tokens / sum of the corresponding durations**,
not the arithmetic mean of per-request rates. Queue time is included in end-to-end
throughput and excluded from language-forward throughput. Failures/unsupported
servers are recorded and give a nonzero exit status; they are not silently dropped.
