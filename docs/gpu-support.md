# GPU support

The shared entry point detects the installed CUDA/HIP Torch build before GPU
initialization, then selects by architecture, model and capacity. Run `inspect`
to review the hardware policy, effective options and explicit overrides. A CUDA
image cannot run AMD cards; use the separate community HIP build.

An [RX 580 8GB / Linux community ROCm experiment](amd-rx580.md) passed Flash,
text, vision, prefix/feature caching and queue checks with SDMA disabled.
Its separate image admits 8192 text tokens, 4096 tokens with images and 8192
unmerged vision patches per image. The larger-photo GPU VM fault remains
unresolved and is guarded before inference. This is experimental support;
see the measured workload limits and historical failures before use.

The unified Linux x86_64 container includes native convolution code for six CUDA
targets. Selection uses device compute capability, available model capacity and
the installed wheel manifest, rather than GPU name matching. Flash remains the
default; `--full` selects Clef 27B NF4. Each service uses exactly one GPU.

| GPU | Target | Flash | Full 27B NF4 | Physical validation |
|---|---|---|---|---|
| RTX 2080 Ti | SM75 | FP16; adaptive prefill | Requires BF16 and more VRAM | Flash tested |
| RTX 3070 Ti | SM86 | FP16; FLA | Requires more VRAM | Flash tested |
| RTX 3090 | SM86 | FP16; FLA | BF16; FLA | Both tested |
| A100 40 GB / 80 GB | SM80 | FP16; FLA | BF16; FLA | Hardware unavailable |
| RTX 4090 | SM89 | FP16; FLA | BF16; FLA | Hardware unavailable |
| H100 | SM90 | FP16; FLA | BF16; FLA | Hardware unavailable |
| RTX 5090 | SM120 | FP16; FLA | BF16; FLA | Hardware unavailable |

Targets come from [NVIDIA's compute capability table](https://developer.nvidia.com/cuda/gpus).
Full needs at least 22 GiB physical VRAM and BF16 capability; Flash needs an 8 GB
card. A small MIG partition can fail those requirements even on an A100 or H100.
The service checks the visible device's capacity, so parent-card VRAM is not used
to admit an oversized model. Memory available to requests is still computed at
runtime; check `/v1/models`. Support does not establish long-context quality.

The pinned Torch 2.11.0+cu128 binary reports SM75/80/86/90/100/120. Its SM86 code
can execute on SM89 via [NVIDIA's Ampere-to-Ada binary compatibility](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html).
The custom convolution wheel includes an explicit SM89 cubin too. CUDA 12.9.1
NVCC supports all six requested targets. The CUDA 12.8 bitsandbytes build covers
SM80/89/90/120 ([installation matrix](https://huggingface.co/docs/bitsandbytes/main/en/installation)).
FLA 0.5.2 recognizes both datacenter SM100 and consumer SM120 Blackwell, and
registers its scratch allocator on SM120 ([pinned device policy](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/utils/_device.py)).
Triton 3.6.0 compiles these kernels at runtime. Use an NVIDIA driver compatible
with the CUDA 12.9 extension and the particular GPU, plus NVIDIA Container Toolkit.
Existing dependency pins are retained.

## Builds and inspection

```sh
# Default portable image: all six targets.
docker build -t clef:local .

# Optional smaller build for one family or a comma-separated subset.
docker build --build-arg CLEF_CUDA_ARCHES=120 -t clef:5090 .
docker build --build-arg CLEF_CUDA_ARCHES=80,90 -t clef:datacenter .

docker run --rm --gpus device=0 clef:local inspect
docker run --rm --gpus device=0 clef:local inspect --full

# Native build: portable Python 3.11 environment, with the pinned dependencies.
.venv/unified/bin/python scripts/build_causal_conv.py --arch all
```

The builder verifies the source checksum, builds unmodified upstream kernel code,
inspects ELF cubins with `cuobjdump`, and records target coverage plus wheel SHA256.
Docker copies that generated manifest into the runtime image. Native installation
writes the same manifest in the selected environment's `causal_conv1d` package,
so different Full/Flash virtual environments cannot overwrite each other's coverage. `inspect` and
`/health.startup_strategy` expose `compiled_kernel_arches`, `kernel_arch_supported`
and `tested_family`. The last field indicates architecture-family hardware testing
(SM75/86 for CUDA, gfx803 for experimental HIP); it does not claim every SKU
in that family was tested.

A reduced build selects native fallback on an uncovered architecture; forcing
FLA on it fails before model loading. Unknown architectures are not automatically
treated as covered. Rebuild rather than copying a manifest from another wheel.
Native fallback additionally requires a Torch build that supports that GPU.

## Validation without the target cards

CPU tests exercise both model profiles on each requested family, 40/80 GB A100
variants, missing targets, single-GPU masking and rejection of undersized devices.
The container build checks actual native cubin coverage, not just compiler flags.
`scripts/test_gpu_compile.py` additionally cross-compiles representative FLA
prefill, recurrence, normalization and continuation-shaped kernels for each new
target in FP16 and BF16. It suppresses GPU launches and does not benchmark or
compare numerical outputs. A visible supported GPU supplies small tensor buffers;
the compilation target is explicitly overridden. Run it in a CUDA environment:

```sh
python scripts/test_gpu_compile.py --arch 120 --output /tmp/sm120-compile.json
```

The A100, 4090, H100 and 5090 still need physical text/vision correctness,
concurrent-request, memory-pressure and throughput validation. Their FLA defaults
are compatibility choices; the 512-token Turing crossover was measured only on
the 2080 Ti. Results from new hardware are welcome.

Completed October 4, 2026: six-target convolution build and cubin inspection,
120 FLA cross-compiled specializations across the four new targets, CPU policy
matrix, and physical 3090 FP16/BF16 plus Full/Flash API regressions all passed.
The image can be rebuilt from this repository with the same dependency pins;
bit-for-bit reproducibility is not claimed.
