# Experimental RX 580 / community ROCm

**Current status:** the optimized Flash service was validated on an RX 580 8GB Linux host.
GPU prefix and independent image-feature caches are enabled on the validated
ROCm math-attention path. Saved rocBLAS choices replay without live tuning.
Prefill uses 1024-token chunks, or 512 above 4096 input tokens. Admission is
8192 tokens for text and 4096 for requests with images, plus an 8192 unmerged
vision-patch cap per image. Larger-photo testing produced a GPU VM fault;
this range is rejected before GPU work. Long-term stability remains experimental.

See [the results](../../docs/amd-rx580.md).

The user also reported replacing the previous RX 580 with a better card.
The precise swap timing is unverified, so the successful retest cannot establish
that disabling SDMA caused the improvement. The original card may have been faulty.

SDMA was also tested enabled on the replacement card. All small tests passed,
but the matched off/on/off comparison found no inference speedup. The running
instance and Dockerfile retain `HSA_ENABLE_SDMA=0`. To repeat the enabled trial,
override it at runtime with `-e HSA_ENABLE_SDMA=1`; do not change other settings.

This isolated build targets **Clef Flash compact NF4 on Linux, gfx803, 8GB**.
It keeps the Clef LCV API, portal, queue, processor and trained vision weights.
It does not install the NVIDIA dependencies or optimized CUDA convolution
wheel. Full and other AMD architectures are rejected by the startup policy.

The base is [Schaka's community gfx803 stack](https://github.com/Schaka/rocm-gfx803).
This is an unsupported hardware experiment, not official AMD support. Its
patched Torch/Triton wheels are protected by the base image's constraints.
bitsandbytes is built from the pinned 0.50.2 commit for gfx803, with a narrow
[wave64 architecture patch](bitsandbytes-gfx803-wave64.patch). Upstream defaults
to wave32 for gfx803; the patch extends its existing wave64 quantization paths.
The existing NF4 checkpoint is mounted read-only, without conversion or requantization.

Build from the project root on x86_64 Linux:

```sh
docker build -f experimental/rocm/Dockerfile -t clef:lcv-rocm-gfx803 .
```

The Dockerfile pins the immutable base digest used for the hardware gate.
The shared startup policy sets `HSA_ENABLE_SDMA=0` before importing Torch,
using compute kernels for copies instead of SDMA. The Dockerfile supplies only
the community runtime, experimental opt-in, and matching offline tuning table.
To select another community build, use
`--build-arg ROCM_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder@sha256:...`.
Changing the base requires rerunning the hardware gate; the upstream versioned
tag is mutable and is not used as the shipped build default.

## Hardware gate

Confirm `amdgpu`, `/dev/kfd`, the render node, and 8GB physical VRAM. Match the
host's video/render group IDs (these vary by distribution). Use one GPU per
service. On the October test host the IDs are 44 and 991:

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  -e HIP_VISIBLE_DEVICES=0 --entrypoint python clef:lcv-rocm-gfx803 /app/probe.py
```

The gate compares FP16 GEMM, layer normalization, causal/chunked attention,
depthwise convolution, NF4 dequantization and nested NF4 linear output against
CPU reference computations. A successful import alone is insufficient.

## First service test

Replace `/path/to/checkpoint` with the complete existing Flash checkpoint.
The writable cache directory is separate from its read-only weights:

```sh
docker run -d --init --name clef-rocm-rx580 --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 991 -e HIP_VISIBLE_DEVICES=0 \
  -p 8085:8080 -v /path/to/checkpoint:/checkpoint:ro \
  -v clef-rocm-cache:/data clef:lcv-rocm-gfx803 \
  --checkpoint /checkpoint --offline
```

Startup automatically selects the measured gfx803 Flash options. Use `inspect`
in place of the serving arguments to review the effective policy without loading
weights. CLI/environment overrides are reported and the tested admission ceilings
cannot be increased. These are fixed tested admission caps, not model maxima. Check `/v1/models`
for `max_input_tokens`, `max_input_tokens_with_images`, and
`max_vision_patch_tokens_per_image`. Excess requests return 413 without
truncation. The per-image bound is checked before optional pooling.

| Experimental setting | Default | Purpose |
|---|---|---|
| `CLEF_ROCM_GPU_CACHE` | `1` | Prefix and independent image-feature reuse with math SDPA |
| `CLEF_CACHE_PROMOTE_REPEATED_BOUNDARY` | `1` | Save a repeated complete prefix, even after a short tail |
| `CLEF_PREFILL_INITIAL_CHUNK_TOKENS`, `CLEF_PREFILL_CHUNK_TOKENS` | `1024` | Reduce launch overhead for shorter requests |
| `CLEF_ROCM_LONG_PROMPT_THRESHOLD` | `4096` | Switch larger text to smaller chunks |
| `CLEF_ROCM_LONG_PREFILL_CHUNK_TOKENS` | `512` | Keep 8K text attention workspace within VRAM |
| `CLEF_ROCM_VISION_TILING` | `1` | Every query tile attends all image keys |
| `CLEF_VISION_QUERY_CHUNK_TOKENS` | `256` | Bound math vision attention workspace |
| `CLEF_MAX_VISION_PATCH_TOKENS_PER_IMAGE` | `8192` | Reject the unvalidated larger-vision workload before GPU work |
| `CLEF_ROCM_TUNABLEOP_FILE` | `/app/tuning/gfx803.csv` | Replay 111 offline rocBLAS choices; no live tuning |

The numeric [tuning table](tunable-rx580-2026-10-04.csv) validates its architecture
and Torch/HIP/BLAS versions at startup. It is specific to the pinned stack and
was measured on this RX 580. Set `-e CLEF_ROCM_TUNABLEOP_FILE=` to disable replay
when validating another stack; regenerate matching results before enabling it.
`PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0` avoids the failing hipBLASLt path.
The [raw optimization suite](optimization-results-2026-10-04.json) preserves both
successes and failures. Fused FLA remains disabled.


Host firmware, clocks, kernel and drivers are left unchanged. Preserve the
NVIDIA image/container and prepared weights for rollback after swapping cards.
Record actual results and failures in [the AMD experiment notes](../../docs/amd-rx580.md).


## Lean runtime packaging

The final image now comes from a fresh runtime base, with compiler/SDK executables
confined to producer stages. Unused Triton and ONNX Runtime/MIGraphX packages are
removed; HIPRTC/COMGR/LLVM shared libraries remain required by the HIP stack.
[Runtime-only binary builds and validation](../../docs/runtime-packaging.md).

## Email benchmark — October 6, 2026

The RX 580 ran the same [100-email benchmark](../../benchmarks/email-sample/README.md)
as the NVIDIA cards, one request at a time with GPU batching off. All 100
requests succeeded in 2,261 s (median 16.6 s per email), and 96/100 matched the
GPT Sol 6.1 reference labels, the same as Flash on NVIDIA. That is about 35 times
slower than Flash on an RTX 3070 Ti. It confirms the path works end to end; it
isn't fast enough for interactive use.
[Results](email-100-results-2026-10-06.json).
