# Unified service and container

`clef_service/app.py` is the single serving implementation for Flash and Full.
Legacy `builds/*/clef_app.py` imports resolve to this module. One profile is
selected per process; each service uses one GPU and one FIFO inference worker.
Both profiles retain the GUI, images, preprocessing and feature caches, elastic
branching prefix cache, context discovery, incremental prefill and optional pooling.

## Startup

```sh
docker build -t clef:local .
docker run -d --name clef --gpus device=GPU-YOUR-UUID -p 8080:8080 \
  -v clef-cache:/data clef:local
docker logs -f clef
```

Default startup selects Flash, masks CUDA to a single visible GPU before Torch
initialization, checks capability and model capacity, downloads the pinned release,
prepares an NF4 checkpoint, then starts Uvicorn. An already prepared managed
checkpoint skips network and quantization. Full uses the same command with `--full`.
`--model full` and `CLEF_PROFILE=full` are equivalent model selectors.

To pre-download Full without starting a second inference service, run
`docker run --rm -v clef-cache:/data clef:local download --full`. The `download`
action does not require a GPU; `prepare --full` also creates the NF4 checkpoint
and needs a compatible GPU. Serving later with `--full` uses those cached files.

Model selection changes what this service loads; it does not run both models on
one GPU. Multiple GPUs can each run a container with its own explicit UUID and
port. Containers may share the model volume; preparation is protected by a
per-profile filesystem lock. Use a local volume/filesystem supporting Linux flock
and atomic rename. In-flight inference caches remain local to each service.

To inspect detection without downloading/loading weights:

```sh
docker run --rm --gpus device=0 clef:local inspect
docker run --rm --gpus device=0 clef:local inspect --full
```

The default wheel has SM75, SM80, SM86, SM89, SM90 and SM120 CUDA code. Turing SM75 uses adaptive FLA
prefill at up to 512 tokens, native PyTorch larger chunks, and optimized
recurrence/convolution/normalization. Ampere SM80/86, Ada SM89, Hopper SM90
and consumer Blackwell SM120 use FLA. Flash uses FP16;
Full uses BF16 and requires at least 22 GiB physical VRAM with BF16 capability.
Architectures absent from the installed wheel manifest use native linear-attention/conv/norm and conservative
uncalibrated context limits. They require a compatible Torch/driver and have not
been benchmarked here. CUDA devices below SM75 and Flash cards below 8 GB are
rejected. The launcher never selects `device_map=auto` or shards onto another GPU.

An explicitly supplied `--gpu` is a CUDA-visible index or UUID. In a container
exposing only a host GPU, its local CUDA index is 0; prefer selecting the host UUID
using Docker's `--gpus device=...` and leaving the app's GPU flag unset.
The service defaults to local index 0 when no mask is supplied. A multi-device
`CUDA_VISIBLE_DEVICES` mask is rejected.

`/health.startup_strategy` records detection and policy defaults. The adjacent
`linear_attention` object records actual selected kernels and explicit overrides.
Use `--linear-prefill-backend torch|fla|adaptive|auto` for experiments.
`--allow-slow-kernels` permits the native fallback on a covered architecture if
optional kernels are absent. Unsupported cubins are never enabled automatically.

See [GPU support](gpu-support.md) for the build matrix and validation boundaries.
A smaller image can be built with `--build-arg CLEF_CUDA_ARCHES=89` (4090),
`120` (5090), `80` (A100), or `90` (H100). Comma-separated subsets also work.
The build checks native cubin coverage and ships `clef_service/kernel-build.json`;
startup reports `compiled_kernel_arches` and uses that manifest for kernel selection.

## Persistent storage and first preparation

```text
/data/
  huggingface/                    Hub metadata/cache
  flash/
    .prepare.lock
    source/                      Pinned Flash release, about 18 GB
    model-nf4-compact/            Prepared Flash, about 4.9 GB
      clef-checkpoint.json        Profile/revision/version/file-size manifest
    cache/sm75-float16/           Architecture/dtype-specific compiler caches
    cache/sm86-float16/
  full/
    source/                      Pinned Full release, about 55 GB
    model-nf4/                    Prepared Full, about 18.2 GB
    cache/sm86-bfloat16/
```

Allow at least 30 GB free for Flash or 85 GB for Full, plus the Docker image/build
cache. Preparation stages its output under a hidden `.preparing` directory and
publishes it with atomic rename only after validation. Interrupted staging output
is discarded and rebuilt on the next startup; Hugging Face retains/resumes source
downloads. A malformed published checkpoint fails clearly rather than being
overwritten. Use a new cache directory to rebuild after deliberate format changes.

Preparation constructs a meta model to identify the exact Linear modules, reads
one source tensor at a time, and quantizes that tensor on `cuda:0`. All serialized
linear weights use NF4 double quantization. Flash input/output embedding matrices
use NF4 row lookup with FP32 scales. Full retains dense BF16 embeddings. Vision
and the native decision head are retained. Output safetensors are sharded to
limit CPU staging; no dense model is loaded onto the GPU and Flash's old dense
intermediate checkpoint is no longer stored. Host RAM is still needed for file
mapping and staging, especially Full's large dense embedding tensors.

The model revisions and decision wrapper are pinned. Managed checkpoint validation
checks profile, revision, format version, required files and recorded sizes;
it is not a cryptographic verification of every weight byte. Full and Flash
checkpoints are never substituted for each other. Model licenses stay with weights.

`--offline` disables downloads. It can prepare from an already downloaded pinned
source, or serve an existing checkpoint. Without either it fails. `prepare` does
the same preparation and exits; `download` fetches source without requiring a GPU.
Optional `HF_TOKEN` is read by Hugging Face for authenticated downloads. It is
not stored in the image or checkpoint manifest. Both currently pinned releases
can be downloaded without authentication.

## Reuse existing prepared weights

Mount an existing checkpoint read-only and use a writable compiler/cache volume:

```sh
docker run -d --name clef --gpus device=0 -p 8080:8080 \
  --user "$(id -u):$(id -g)" \
  -v /path/to/model-nf4-compact:/checkpoint:ro \
  -v /path/to/writable-cache:/data \
  clef:local --checkpoint /checkpoint --offline
```

For Full mount `model-nf4` and add `--full`. Explicit checkpoints can predate the
new manifest but must match the selected profile's quantization/dtype and complete
file layout. They are never modified. Bind-mounted files/cache must be accessible
to the container user; its default UID/GID are 1000. Named volumes work by default.
`--data-dir` remains an exact per-profile directory for legacy installations;
`--cache-dir`/`CLEF_CACHE_DIR` names the new parent cache.

## Container operation

Use Linux x86_64 and NVIDIA Container Toolkit with a driver compatible with
CUDA 12.8 Torch and the CUDA 12.9-built extension. Build stages download pinned
Python dependencies and checksum-verified causal-conv1d source. The final image
contains no model weights, CUDA compiler, private test inputs or access credentials.
A small host C compiler is retained because Triton builds its CUDA driver stub
at runtime. CUDA kernels and their JIT caches are supported without the full
CUDA development toolkit.
It runs as a non-root user. `/data` stores weights and compiler caches. Model
downloads/preparation happen before the HTTP service is ready, so the healthcheck
allows a 30-minute startup period. Longer downloads remain visible in logs.

Compose exposes a single configurable GPU, not all GPUs:

```sh
CLEF_GPU=GPU-YOUR-UUID CLEF_PORT=8080 docker compose up --build -d
CLEF_GPU=GPU-YOUR-UUID CLEF_PROFILE=full docker compose up -d
```

After startup check `/health`, then `/v1/models` for the current computed context
budget before sending requests. Oversized encoded input returns 413. Queued work
waits for the single GPU worker; admission overload returns 429. Unexpected GPU
memory exhaustion returns 503. Model preparation does not fix the recorded Flash
long-context retrieval quality artifact.

## Validation

The CPU regression script covers tested/unknown GPU policies, undersized model
rejection, single-device masking, interrupted preparation, cache reuse, profile
mismatch and truncated checkpoint rejection. The API smoke script exercises the
same shared app for text and vision, exact reuse, optional pooling, sixteen images,
8K continuation, discovery, input/invalid-model errors, and four concurrent calls.

Validated on October 4, 2026:

| Case | Result |
|---|---|
| Full 27B on RTX 3090, Docker | All API smoke checks passed; computed text/image budget 45,056 |
| Flash on RTX 3070 Ti, Docker | Fresh preparation, all API checks, cached restart passed; computed text/image budget 24,576 |
| Flash on RTX 2080 Ti, same container contents | All API checks passed; adaptive 512-token strategy selected; computed text budget 65,536 and image budget 24,576 |
| Native fallback simulated on RTX 3090 | Text, vision, pooling, cache, queue and rejection checks passed; conservative 8,192-token cap; >8K continuation intentionally not tested |
| SM86 optimized kernels | FP16 and BF16 output, recurrent-state, split-continuation, single-token recurrence and convolution checks passed |
| SM75 optimized kernels | FP16 output/state, continuation, recurrence and convolution checks passed |

One-tensor-at-a-time preparation produced the same tensor values as the existing
working checkpoints: all 2,548 Flash core tensors, both compact embedding sets,
and all 4,214 Full tensors matched. Flash preparation used 2,485.6 MiB peak GPU
allocation and took 35.78 seconds on the 3090 / 41.58 seconds on the 3070 Ti.
Full preparation took 209.84 seconds with 224.5 MiB peak GPU allocation;
its large dense embeddings are staged on CPU. Preparation timings depend on disk
and host load. Previously downloaded sources were used for the 3070 Ti and Full preparation;
a fresh pinned Flash source download was also completed separately.

These are compatibility and correctness checks, not controlled throughput
benchmarks. Cold starts still load weights, initialize temporary CPU embeddings
for compact restoration, and may compile new Triton kernel shapes. Later starts
reuse downloaded/prepared weights and compiler caches; in-memory request caches
reset at service restart. Other GPU families remain untested on physical hardware.
The existing Flash long-context retrieval artifact remains recorded and unresolved.

