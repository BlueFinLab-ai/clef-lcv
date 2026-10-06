# RX 580 8GB / Linux experiment

The active route is **community ROCm with PyTorch**, using the same Clef LCV
service and existing compact NF4 Flash checkpoint. The NVIDIA Docker image and
dependency lock remain separate. The alternative [direct Vulkan operator map](../experimental/vulkan/README.md)
is retained as research; it has no implemented Clef runtime.

**Current status:** the optimized Flash service was validated on an RX 580 8GB Linux host.
GPU prefix and independent image-feature caches are enabled on the validated
ROCm math-attention path. Saved rocBLAS choices replay without live tuning.
Prefill uses 1024-token chunks, or 512 above 4096 input tokens. Admission is
8192 tokens for text and 4096 for requests with images, plus an 8192 unmerged
vision-patch cap per image. Larger-photo testing produced a GPU VM fault;
this range is rejected before GPU work. Long-term stability remains experimental.

## Startup strategy

The shared `python -m clef_service` entry point reads the installed Torch build's
HIP/CUDA metadata before importing Torch, selects one device in the corresponding
visibility namespace, then checks the actual GPU architecture, VRAM and model.
ROCm still requires `--experimental-rocm` or `CLEF_EXPERIMENTAL_ROCM=1`; the
community image supplies that opt-in. Flash is the default. Full, other AMD
architectures and cards with less than 8GB are rejected before loading weights.
The CUDA image and HIP image remain separate dependency builds.

The `gfx803-flash-native` startup policy now supplies the measured defaults
rather than relying on duplicated Docker environment settings: FP16 NF4,
native GDN block 64, math SDPA, outer chunks 1024 and 512 above 4096 tokens,
GPU prefix/image-feature caches, repeated-prefix promotion, query tiling 256,
SDMA off and allocator expansion off. Idle cache headroom is 256 MiB and active
prefill margin is 512 MiB. Pooling and the observed-workspace experiment stay off.
The experimental dense GEMM/GEMV adapters and text batching prototype are not
installed into serving. The existing FIFO worker remains one GPU per service.

Text admission stays at or below 8192 tokens, image-request language admission
at or below 4096, and raw patches per image at or below 8192. Startup rejects
attempts to raise these bounds or use automatic CUDA memory calibration on HIP.
CLI and environment overrides can lower limits or disable caches; experimental
chunk, native-block and SDMA overrides remain explicit choices for retesting.

If the image packages `/app/tuning/gfx803.csv`, startup selects it automatically.
An installation without that file uses ordinary rocBLAS until a matching table
is supplied with `CLEF_ROCM_TUNABLEOP_FILE`. An explicit empty value disables
replay. Torch validates the table against runtime versions before use; a missing
explicit file or incompatible table fails startup. Live tuning and hipBLASLt
are disabled. Changing runtime versions requires a new validated tuning table.

`inspect` works without downloading or loading model weights:

```sh
python -m clef_service inspect --experimental-rocm --gpu 0
```

It and `/health.startup_strategy` expose `strategy_id`, `experimental`, the
selected GPU/model/dtype, `runtime_options`, and overrides relative to the
hardware defaults. `/health` also reports the actual loaded attention/tuning
backends. The policy does not install drivers or turn a CUDA Torch build into
HIP; use the matching image first.

## Current scope

### Native GDN block and prefill admission controls

`CLEF_TORCH_GDN_BLOCK_TOKENS=16|32|64` selects the internal block size of the
upstream native PyTorch gated-delta algorithm. The default is 64. This requires
`CLEF_LINEAR_PREFILL_BACKEND=torch`; it does not replace the outer 1024/512-token
prefill chunks, change model weights, or discard context. FP32 recurrent state is
retained. Changing the arithmetic partition can change rounding, so validate
decisions and cached continuation on each runtime before enabling it elsewhere.
`python scripts/test_native_linear_attention.py --device cuda` exercises native
partitions and continuation state through the PyTorch CUDA/HIP device API.

`CLEF_ROCM_OBSERVED_PREFILL_WORKSPACE=1` uses the largest measured chunked-prefill
workspace when admitting another chunked ROCm request. Fixed-limit deployments
do not have a calibrated token-based projection; using only a short-request
measurement can retain too many cache entries and cause an expensive retry.
Admission evicts before inference and releases unused allocator segments when
needed. Idle cache headroom remains 256 MiB; active prefill uses the existing
512 MiB margin plus its observed workspace. This is a measurement-based guard,
not a guarantee that an unseen input shape cannot exhaust memory. The existing
bounded retry still applies. Other runtime policies keep their existing defaults.

The follow-up email trial reduced wall time by 17.4% with that reservation flag,
but subsequent 8192-token and larger-photo repeats in the same process after
the kernel experiments lost GPU prefix retention.
The flag therefore remains disabled by default and on the restored service.
It needs estimates that account for the reusable prefix and remaining work
and a fresh native-only large-context comparison before it can be recommended
across request sizes. The measured 64-token native
GDN block also remains the default; smaller blocks slowed individual requests.

[Existing gfx803 kernel experiments](../experimental/rocm/gfx803-kernels/README.md)
record numerical checks and timing for the community prefill GEMM, two weight
layout paths, and native-layout GEMV-M. Faster raw multiplication did not produce
a net gain for this NF4 prefill deployment. A separate text batching prototype
reduced an eight-email run by 11.1%; production FIFO behavior is unchanged.

[Experimental build and hardware gate](../experimental/rocm/README.md).
The startup policy explicitly requires gfx803, at least 8GB VRAM, Flash and
`CLEF_EXPERIMENTAL_ROCM=1`. It selects FP16, native PyTorch gated-delta kernels,
math SDPA and native gated-delta attention. The shared startup policy selects
1024-token chunks and 512 above 4096 tokens. Full is rejected for this experiment.
Torch HIP continues to use the `torch.cuda` API and `cuda:0` device spelling;
that does not mean it is executing on NVIDIA hardware.

The base is [Schaka's patched gfx803 ROCm image](https://github.com/Schaka/rocm-gfx803).
Its Torch/Triton wheels are retained. bitsandbytes is built from the pinned
0.50.2 source commit for gfx803 rather than installing the CUDA wheel. Existing
NF4 weights are mounted read-only and are neither converted nor requantized.

## Physical host

On October 4, 2026, server <HOST_NAME> was found with the RX 580 already
installed: MSI Ellesmere PCI 1002:67df / subsystem 1462:8a92, amdgpu driver,
8,589,934,592 bytes of physical VRAM, Ubuntu 26.04 and kernel 7.0.0-38-generic.
The host has an i7-6700, 12GB system RAM and 15GB swap. `/dev/kfd` and the
render device are present. Host drivers, firmware and clocks are unchanged.

The previous NVIDIA container was stopped after the card swap; its image,
configuration and compact NF4 checkpoint were kept. AMD testing used a separate
directory, container and port rather than replacing that configuration.

## Physical results, October 4

[Machine-readable results](../experimental/rocm/results-2026-10-04.json).
The pinned community image is
`ghcr.io/schaka/rocm-migraphx-ort-torch-builder@sha256:b65af203cf7ebdf1751127ee79333438acfc5226086e889391f675e176d5e7f4`.
Its provenance records source revision `50614047a5171f2ab4c4ab5e7a43d23906b8df69`.
Torch is `2.14.0+gitc299f65` with HIP `7.15.26333` inside the ROCm 10 distribution.
The upstream `latest-gfx803` tag resolved to the same digest at test time.

The hardware gate passed FP16 GEMM, layer normalization, causal/chunked math
SDPA, depthwise convolution, nested NF4 dequantization, batched and single-row
linear output, and compact embedding lookup. Dequantization and embedding
lookup matched CPU output exactly; relative maximum linear errors were below
0.00051. NF4 round-trip RMS loss was about 9.2%, distinct from GPU-versus-CPU
kernel error.

bitsandbytes required one narrow source patch: its architecture constants
assumed wave32 for gfx803. [The patch](../experimental/rocm/bitsandbytes-gfx803-wave64.patch)
selects wave64, extending the existing quantization load/store and reduction
paths. No weights were changed.

Flash loaded all 358 NF4 layers on the single RX 580. Idle Torch allocation was
4966.8 MiB; device memory including runtime overhead was approximately 5.1 GiB.
The initial service used a fixed 2048-token smoke-test cap and math SDPA with
256-token prefill chunks. This is not a measured maximum context.

| HTTP request | Tokens | Client time | Result |
|---|---:|---:|---|
| Outage text, first | 229 | 18.50 s | Outage, urgent, severe |
| Same text, repeat | 229 | 7.06 s | Same decisions |
| Red-circle image, first | 176 | 8.86 s | Red |
| Same image, repeat | 176 | 6.27 s | Red |

These are small smoke cases, not the 100-email benchmark. First calls include
kernel initialization. GPU prefix and image-feature caching were disabled;
CPU preprocessing caching was enabled. Repeat probability differences on the
text case were approximately 0.0001 and did not change decisions.

The user subsequently reported replacing the earlier RX 580 with a better
RX 580. The exact swap time relative to the recorded trials is not established.
A faulty original card is therefore a possible explanation for the failure.
The successful retest cannot isolate the effect of disabling SDMA from the
card replacement and reboot; a subsequent off/on/off comparison on the replacement
card is recorded below.

## Initial failure and SDMA-disabled retest

The first SDMA-enabled two-photo request stalled. At 00:27 UTC on October 5 the
kernel reported `ring sdma1 timeout`, `device lost from bus`, and
`GPU Recovery Failed: -19`. The container was stopped (exit 137, not an OOM kill),
preserving its logs and configuration. The underlying cause is unresolved;
these symptoms alone do not distinguish hardware, firmware, host driver or runtime.

The user power-cycled the host. The second build sets `HSA_ENABLE_SDMA=0`, which
[AMD documents](https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/debugging.html)
as selecting compute-shader copy kernels. The GPU gate passed again, followed
by the same text/single-image/two-photo sequence. This build also includes the
CPU bitsandbytes library for GPU-free build validation and removes unused vLLM
packages from the PyTorch service image.

| SDMA-disabled HTTP request | Tokens | First / repeat client time | Result |
|---|---:|---:|---|
| Outage text | 229 | 10.54 / 4.41 s | Outage, urgent, severe |
| Red-circle image | 176 | 4.85 / 3.90 s | Red |
| Two supplied photographs | 268 | 11.75 / 8.98 s | Two images; fish detected |

Four further two-photo calls took 8.99, 8.98, 9.09 and 9.00 seconds and all
selected two images. Fish scores stayed around 0.68 and image-count confidence
around 0.94. The photographs fit within 256x256 with aspect ratio preserved.
Peak Torch allocation on the two-photo case was 5149.5 MiB.

Two concurrent text calls completed in 4.38 and 8.76 seconds. The latter spent
4.37 seconds waiting in the FIFO queue; both returned the expected decisions.
The health endpoint recorded 12 completed requests and zero failures. No new
timeout, lost-bus, GPU reset or VM fault was found after the reboot/retest.
The running container is `clef-rocm-rx580-sdma-off`, image
`clef:lcv-rocm-gfx803-sdma-off-20261004`.

The first-call differences include initialization and other run-to-run effects;
these samples do not isolate a throughput gain caused by disabling SDMA.
Cache parity, longer contexts, sustained loads and the 100-email benchmark
remain untested on this AMD build. The 2048 cap is not a measured maximum.

CPU opt-in/policy tests, existing NVIDIA hardware/checkpoint checks and project
validation passed. The default NVIDIA Dockerfile and dependency pins remain
unchanged. The experiment uses its own image, cache directory and port 8085;
the original NVIDIA container, image and checkpoint remain preserved.

## Replacement-card SDMA comparison

A controlled off → on → off comparison was then run on the replacement RX 580.
The same container image and checkpoint were used, overriding only
`HSA_ENABLE_SDMA` (0 / 1). The CPU thread count, 2048-token cap, chunk settings
and payloads stayed unchanged. The enabled hardware gate and all 12 enabled
HTTP cases passed. No new GPU timeout, lost-bus, reset or VM fault appeared.

| Warm measurement | SDMA off, before | SDMA on | SDMA off, after |
|---|---:|---:|---:|
| Two-photo client time, mean of four calls | 8.98 s | 9.46 s | 9.46 s |
| Text inference time, mean of two queued calls | 4.38 s | 4.55 s | 4.56 s |

The first photo call after the final restart was excluded as warmup. Queue
waiting is excluded from the text inference measurement. SDMA did not improve
inference in this workload. The final disabled run was close to the enabled
run (0.1% difference), so the roughly 5% change from the initial
baseline cannot reliably be attributed to SDMA. This small serial comparison
does not establish a universal transfer-performance result. SDMA affects copies, while most of this request
is model computation. SDMA-disabled operation is restored on port 8085;
the enabled container is preserved, stopped, for future experiments.

Passing SDMA-enabled tests on the replacement card makes an original-card
fault plausible, but it does not establish the root cause of the earlier crash.


## Optimization follow-up

[Optimization measurements and failures](../experimental/rocm/optimization-results-2026-10-04.json)
record chunk ablations, rocBLAS tuning, prefix promotion, independent feature
reuse, real-photo dense/tiled parity, and admission boundaries.

~2K uncached text fell from 43.87 to 22.42 seconds with larger chunks and saved
rocBLAS choices. Repeated 3,628-token photo input fell from 84.28 seconds cold
to 5.68 seconds on its first repeat and 3.26 seconds after complete-prefix
promotion. 8K text required 512-token chunks: 139.00 seconds cold, 3.76 cached,
with matching decisions and probabilities at response precision.

The subsequent two-photo 2048-pixel case failed with a GPU VM memory-access
fault and exit 134, without an OOM kill. The faulting operator is unresolved.
A new GPU process recovered without reboot. The per-image vision-patch guard
blocks that workload before GPU work, even with pooling. The accepted 8192-patch
single-image test produced 2218 language-input tokens, beyond the old cap.

The preceding 2048-token configuration and disabled-cache descriptions are
historical. Current limits are fixed measured admission settings; the NVIDIA
context-memory calibration is not applied to this ROCm math-attention route.


## Startup integration validation, October 4

The measured options are centralized in the shared launcher and the experimental
Dockerfile no longer duplicates them. Physical `inspect` on the RX 580 selected
`gfx803-flash-native` after clearing the previous image's ROCm option environment,
with no overrides and without loading weights. The updated test image
was built as a source-only layer over the tested HIP image, retaining its
patched dependency stack. The prior optimized container is preserved, stopped,
for rollback; a full clean dependency-image rebuild was not repeated in this step.

[Startup/API regression results](../experimental/rocm/startup-api-validation-2026-10-04.json)
record two-photo calls at 11.87 / 3.39 / 3.37 seconds, correct image counts,
two successful queued text calls, token/vision 413 guards (including pooling),
358 NF4 layers, and 111 offline rocBLAS entries with live tuning disabled.
These are smoke checks, not a new throughput benchmark. The queue returned to
zero active/waiting requests. CPU tests passed pre-import runtime dispatch,
HIP isolation, explicit opt-in, caps, CLI precedence, and the existing NVIDIA
hardware/checkpoint matrix. No new physical NVIDIA benchmark was run.


## Batch-four follow-up, October 5

[Raw 512-token trial](../experimental/rocm/batch4-results-2026-10-05.json) and
[256-token paired trials](../experimental/rocm/batch4-chunk256-results-2026-10-05.json)
use the same eight synthetic emails as the earlier text batching prototype.
Fresh serial and batch-two controls took 124.33 and 108.63 seconds with 512-token
tail chunks. Batch four failed during warm-up in language math SDPA with physical
VRAM exhausted; the successful controls were preserved, but planned return
controls were not reached.

With 256-token tails, batch two averaged 164.34 seconds and batch four averaged
134.28 seconds over two runs each, in 2/4/4/2 order. Batch four gained 18.3% against
that matching smaller-chunk control but was 23.6% slower than the best batch-two
configuration. Both batch-four runs recovered from allocator allocation failures.
Torch peaks were 5.68 and 6.38 GiB, excluding external HIP/BLAS/driver allocations.
Mean batch completion was 67.14 seconds for batch four, versus 27.16 seconds for
the faster batch-two path. All completed category choices matched serial, with
maximum probability difference 0.0018. The shared prefix was prebuilt and shape
warm-ups excluded; these are isolated GPU-group timings, not HTTP queue latency
or a full 100-email provider benchmark. The normal portal was restored afterward.

[Reproduction instructions](../experimental/rocm/batch-sizes.md).
Production FIFO behavior remains unchanged. Batch two is still the stronger
candidate for future memory-aware queue integration on this workload; batch four
is not a safe universal default for the RX 580 8GB.
