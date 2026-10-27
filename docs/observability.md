# Optional observability and request logging

Metrics and request logging default **off**. They use only Python's bundled
standard library; no exporter SDK, compiler or new package is required.

```sh
python -m clef_service --metrics \
  --request-log-mode metadata --request-log-dir /data/logs/flash-gpu3
```

`GET /metrics` exposes Prometheus text format and uses the same authentication
as other API endpoints. With an API key configured, provide its Bearer header.
When metrics are off the endpoint returns 404. `/health.observability` shows the
active mode, queued reservations, rejected queue offers, dropped captures,
write errors and the last error's type. It never exposes log contents or paths.

## Answering questions takes priority

The GPU worker performs no new observations. A small allowlisted event is offered
**after the HTTP response is sent**. The producer uses a nonblocking lock; a busy
or full queue rejects the event rather than waiting. Logging errors do not change
a response or retry inference. Metrics aggregation and storage have independent
background threads and queues, so a slow disk cannot stall collection or inference.

Collection adds no CUDA/NVML calls, GPU synchronization, tensor copies, cache
inventories or tensor-storage walks. Gauges read existing CPU integers; retained
VRAM bytes are the most recently computed prefix-plus-image-feature total.
Scrapes return an immutable snapshot prepared in the background, refreshed about
twice a second. Metrics are best effort and may be stale or incomplete; inspect
snapshot timestamps and dropped-event counters.

Background work still consumes CPU, RAM and storage bandwidth. This is not a
zero-overhead or lossless audit guarantee. Defaults remain off for deployments
that have not qualified their workload. Use `--no-metrics` to override an enabled
environment setting. Large payload capture is separately opt-in and was not
included in the latency qualification for ordinary metrics/metadata.

## Metrics

The request counter includes 4xx/5xx errors and client cancellations. Histograms
cover HTTP duration, queue wait, processing, inference event window, estimated
CPU elapsed time, prefix restoration and offload. Missing backend timing fields
are omitted rather than treated as zero. Token/reuse counters, cache outcomes,
queue gauges, last batch size, last peak allocation and CPU/GPU cache occupancy
are available. Labels are restricted to service profile, status, cache outcome
and tier; request IDs, prompts, image hashes and user identifiers are not labels.

`inference_window_seconds` uses existing CUDA event timing and can include CPU
launch gaps; it is not exclusive kernel time. `cpu_estimated_seconds` is the
existing elapsed estimate, not CPU utilization or summed core time.

## Metadata logs

`--request-log-mode metadata` writes `requests.jsonl`, with UTC Unix timestamps,
random request IDs, optional cache-trace IDs, status, error category, existing
usage fields and timing data. It excludes input text, questions, images, results,
headers, cookies, credentials and client addresses.

Segments target 8 MiB with four backups. `--request-log-rotate-mib`,
`--request-log-backups` and `--request-log-retention-days` customize limits.
Rotated segments expire on subsequent writes; retention is segment-based and
an active segment can contain older records until rotation. Newly created log
files have owner-only permissions. Use a separate directory per service instance
and a persistent Docker volume when logs must survive container replacement.

## Optional text/result/image capture

```sh
python -m clef_service --metrics \
  --request-log-mode payload --request-log-dir /data/logs/flash-gpu3 \
  --request-log-images --request-log-archive-mib 1024
```

Payload mode also writes `payloads.sqlite3`, containing validated requests
(defaults included), returned decisions/probabilities/usage and error results.
There is no archive-reading HTTP endpoint. Inputs rejected before authentication
or schema validation are not captured. Cancelled requests may have a cancellation
status without a returned answer.

Images are omitted unless `--request-log-images` is explicitly enabled. Received
JPEG, PNG and WebP bytes are saved without resizing or re-encoding, deduplicated
by SHA-256. References retain original image order, including duplicates.
Authentication headers and cookies are never captured in either mode.

The archive removes oldest request records under its logical payload budget;
expired records are pruned on subsequent captures. Shared images remain while
any retained request references them. SQLite page limits also cap the main
database file; indexes and transient rollback journals add storage overhead.
A capture failure falls back to metadata and does not fail inference.

The storage thread serializes payloads and decodes base64 image bytes. These
operations can compete for CPU and I/O, so full capture needs its own workload
qualification. Capture uses an additional reservation of eight times received
wire length for parsed objects; this is an estimate, not an exact measurement of
Python heap usage. The default log queue budget is 64 MiB. An oversized capture
is dropped while a metadata fallback is attempted. Configure
`--request-log-queue-mib` for larger captures; request admission limits still apply.

## Environment settings

| Variable | Default | Purpose |
|---|---:|---|
| `CLEF_METRICS` | `0` | Optional metrics; `--metrics` / `--no-metrics` override |
| `CLEF_REQUEST_LOG_MODE` | `off` | `off`, `metadata` or `payload` |
| `CLEF_REQUEST_LOG_DIR` | unset | Required for logging |
| `CLEF_REQUEST_LOG_IMAGES` | `0` | Requires payload mode |
| `CLEF_OBSERVATION_QUEUE_ITEMS` | `4096` | Maximum items in each background queue |
| `CLEF_REQUEST_LOG_QUEUE_MIB` | `64` | Reserved bytes for queued log records |
| `CLEF_REQUEST_LOG_ROTATE_MIB` | `8` | Metadata segment target |
| `CLEF_REQUEST_LOG_BACKUPS` | `4` | Retained rotated segments |
| `CLEF_REQUEST_LOG_RETENTION_DAYS` | `7` | Segment/archive age limit on writes |
| `CLEF_REQUEST_LOG_ARCHIVE_MIB` | `1024` | Optional payload archive budget |

Metadata queues carry only bounded scalar fields. Credentials belong in secret
files/environment settings, not request bodies; payload mode intentionally saves
user-supplied content verbatim. No logs are transmitted to an external service.
