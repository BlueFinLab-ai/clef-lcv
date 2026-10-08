# Optimized linear-attention kernels

Full and Flash can use Flash Linear Attention 0.5.2 (`fla-core` 0.5.2),
causal-conv1d 1.7.0, Torch 2.11.0+cu128 and Triton 3.6.0. Full uses BF16
compute and 48 linear-attention layers; Flash uses FP16 and 24. Optimized
GatedDeltaNet chunk/recurrent kernels, causal convolution and gated normalization
replace the native PyTorch fallback. Full-attention layers still use
memory-efficient PyTorch SDPA. Weight quantization, compact Flash embeddings,
image handling, decision heads, queues and prefix caching are unchanged.

## Build

Use Linux x86_64, a portable Python 3.11.17 distribution (for example uv), and
Docker usable by your account. The interpreter and its base distribution must be
readable at their existing host paths from Docker bind mounts. Set `PYTHON` when
creating the environment. A system Python is not supported by this container
recipe. Build and validate in a separate candidate environment before replacing a
running service.

```sh
make env-full PYTHON=/absolute/path/to/portable/python3.11
make kernels-full

# Or, after creating the base Flash environment:
make kernels-flash CUDA_ARCH=86 # RTX 3070 Ti / 3090
# Use CUDA_ARCH=75 for RTX 2080 Ti.
```

Flash additions are pinned in `requirements/flash-kernels.txt` and installed with
`--no-deps` to preserve the base Torch/Transformers versions. The base Flash
requirements retain the fallback option. Full requirements include FLA already.

The successful build uses `nvidia/cuda:12.9.1-devel-ubuntu24.04`. The original host
GCC 14 / CUDA 12.8 / glibc combination could not compile causal-conv1d. The build
changes only CUDA architecture flags and NVCC compiler threads (four to two);
CUDA kernel source is unchanged. Source URL and checksum are pinned in
`profiles/causal-conv1d-source.json`. The default native recipe now builds SM75/80/86/89/90/120 together.
Use `--arch 75|80|86|89|90|120` for a single target or `--arch all` for all six.
A reduced wheel must match the intended GPU. Build output records targets and
wheel SHA256, verifies cubin coverage, and installs `clef-kernel-build.json` in that environment's `causal_conv1d`
package. Full and Flash virtual environments can contain different wheels.
For manually installed wheels, copy the matching build's `clef-kernel-build.json`
to the installed `causal_conv1d` package directory too. Without a manifest the launcher retains the legacy SM75/86
policy. See [GPU support](gpu-support.md) for hardware validation status.

To build a wheel without changing the interpreter environment:

```sh
.venv/full/bin/python scripts/build_causal_conv.py --arch 75 --build-only --output build/kernels-sm75
```

The build mounts the virtual environment read-only. A normal build installs the
resulting wheel into that environment and verifies Transformers' fast-path flag.
Use distinct output directories for architecture-specific builds.

## Validate and serve

```sh
CUDA_VISIBLE_DEVICES=0 .venv/full/bin/python scripts/kernel_smoke.py
CUDA_VISIBLE_DEVICES=0 .venv/flash/bin/python scripts/kernel_smoke.py --dtype float16
CLEF_REQUIRE_FAST_LINEAR_ATTENTION=1 make serve-flash GPU=0
```

Full's launcher requires fast kernels by default. Flash's require flag is opt-in.
With it enabled, startup fails unless the fast path and all expected FLA chunk
kernels are available before any explicit prefill override. Check `/health.linear_attention`: expect
`fast_path_available: true`, 48 layers for Full or 24 for Flash, and chunk-kernel
names beginning with `fla.`. Flash also reports `optimized_prefill`,
`optimized_recurrence`, `recurrent_kernels` and `prefill_override`. Availability
alone does not identify the selected prefill implementation.

Flash accepts `CLEF_LINEAR_PREFILL_BACKEND=auto|fla|torch|adaptive` (default `auto`). The
explicit `torch` option retains native chunk prefill while leaving available
optimized recurrence, convolution and gated normalization enabled. This permits a
mixed path when FLA chunk prefill is slower on a particular GPU. An explicit
`fla` request fails if its chunk kernels are unavailable. The require flag can be
combined with `torch`: it requires the optimized library path, then intentionally
selects native chunks. The `adaptive` option uses FLA chunks through
`CLEF_FLA_MAX_CHUNK_TOKENS` (positive integer, default 512) and native chunks above
that size. It requires FLA availability and keeps optimized recurrence,
convolution and normalization. The deployed 2080 Ti uses this option after its
FP16 kernel crossover tests; the 3070 Ti uses FLA throughout. This is an explicit
hardware-tuning choice, not automatic architecture detection. `optimized_prefill`
means every chunk uses FLA; it is false for `torch` and `adaptive`. Inspect
`prefill_override` and `fla_max_chunk_tokens` for the actual policy. All these
choices leave full-attention SDPA unchanged.

```sh
CLEF_REQUIRE_FAST_LINEAR_ATTENTION=1 CLEF_LINEAR_PREFILL_BACKEND=adaptive \
  CLEF_FLA_MAX_CHUNK_TOKENS=512 make serve-flash GPU=0
```

To reproduce the operation-level head geometry and kernel crossover test:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/flash/bin/python scripts/kernel_smoke.py --dtype float16 \
  --heads 32 --lengths 64,128,256,512,1024,4096,8192 --benchmark
```

It reports median kernel time over ten measured iterations after two warm-ups.
These operation timings do not include weights, full attention or HTTP.

Kernel smoke checks cover outputs, final recurrent state, split-prefix
continuation, single-token recurrence, and convolution. They passed in FP16 on
SM86 and SM75, and in BF16 on SM86. Model comparisons must also exercise text,
images, pooling, cached continuations, and the intended largest context. The
portable comparison script is `scripts/benchmark_flash_kernels.py`; private email
fixtures and photos are optional external inputs, never packaged with source.
For example, compare native, fully optimized and mixed prefill in alternating
order, excluding warm-up, on selected new-prefix inputs:

```sh
.venv/flash/bin/python scripts/benchmark_flash_kernels.py --gpu 0 \
  --data-dir /absolute/path/to/models/flash --output build/flash-interleaved.json \
  --include-native-prefill --include-adaptive-prefill --fla-max-chunk-tokens 512 \
  --interleaved-rounds 3 \
  --cases short,text-8192,text-24576
```

Omit interleaving to include uncached/build/warm comparisons. Use
`--allow-capacity-misses` for large prefixes on small GPUs: it permits budget
retention declines, while still rejecting memory-fallback outcomes.

The first request for a new kernel shape can spend time compiling. Persist
`TRITON_CACHE_DIR` and `PYTORCH_KERNEL_CACHE_PATH` in writable storage and exclude
compilation/warm-up from steady-state timings. The launcher sets persistent paths
under its data directory. See [validation](validation.md) for measured results.

These optional kernels do not increase context limits or change scheduling.
Retained prefixes can still be declined or evicted under memory pressure. Keep
the original environment and adapter backup for rollback. A clean model download
and quantization cycle from the packaged repository remains a separate unvalidated
step; the kernel build and existing-model comparisons were exercised on the hosts.
