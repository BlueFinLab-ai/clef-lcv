# Compiled kernels and lean serving images

The serving image and kernel producer are separate. The producer can contain
CUDA/HIP compilers, CMake and Ninja; the final serving image excludes those
general compiler/build executables. This is a reduction in build-tool footprint,
not a claim that every form of runtime GPU compilation has been removed.

## NVIDIA

The convolution extension already contains native cubins for SM75, SM80, SM86,
SM89, SM90 and SM120. Flash and Full share that verified extension. The pinned
Torch and bitsandbytes wheels provide their GPU binaries too.

The image now includes a small [Triton host-helper pack](../artifacts/triton-host/manifest.json)
for CPython 3.11 / Linux x86_64 / Triton 3.6.0 / Torch 2.11.0+cu128 / FLA 0.5.2.
These Python/CUDA driver and launcher modules are independent of GPU SM. The
pack is checked for ABI, package versions, path containment and binary SHA256
before installing Triton's existing `knobs.build.impl` hook. On a cold cache it
loads the matching packaged module instead of invoking GCC. Missing signatures
fail explicitly and must be generated in a builder; no runtime C compiler
fallback is enabled. Dependency changes require refreshing the pack.

Triton's GPU compiler backend, libdevice and PTX assembler remain. FLA still
specializes GPU code for architecture, dtype, layouts and runtime shapes.
A finite warm cache is not universal ahead-of-time coverage for arbitrary
requests or future package versions. Removing that backend would require a
separate AOT kernel/dispatch design or native fallback with different performance.
The optimized inference strategy remains enabled in this packaging change.

Build the producer once, then let users build the runtime from its binaries:

```sh
# Maintainer build; SDK/compiler confined to the producer stages.
docker build --target cuda-binaries -t clef:cuda-binaries .

# Customer build: no kernel compilation or compiler installation.
docker build -f Dockerfile.runtime \
  --build-arg CLEF_BINARIES_IMAGE=clef:cuda-binaries -t clef:runtime .
```

The normal `Dockerfile` also produces a lean final runtime. Published binary
images should be pinned by digest when distributed. Local binary images exist
on the validation host; no registry publication was performed by this work.
The binary producer does not contain model weights or user inputs. Model
selection, download/preparation, offline checkpoints and single-GPU isolation
retain the existing interface.

## RX 580 / community ROCm

The [ROCm Dockerfile](../experimental/rocm/Dockerfile) now uses a builder,
runtime-payload collector and fresh Ubuntu runtime rather than serving from the
entire community SDK builder. The collector keeps the patched Torch and
bitsandbytes binaries, required shared libraries, code objects, BLAS solution
indexes, MIOpen data and runtime-required headers. SDK executables, GCC/G++, Git,
CMake, Make and Ninja are excluded. Unused Triton and ONNX Runtime/MIGraphX Python
packages are removed from the native serving path. The embedded Python runtime
is relocated into `/opt/python`; ROCm aliases are explicit rather than relying
on the builder's `/etc/alternatives` configuration.

HIPRTC, COMGR, LLVM/Clang shared libraries and runtime-required headers remain:
Torch's HIP libraries link them, and MIOpen can compile kernels for new input
shapes. AMD documents [MIOpen's precompiled caches and runtime compilation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/install/install.html).
These libraries are distinct from standalone compiler executables. Removing them
breaks the retained runtime rather than simply eliminating unused build tools.
ROCm support remains experimental, gfx803 / 8GB / Flash only.

```sh
# Maintainer producer build.
docker build -f experimental/rocm/Dockerfile --target rocm-binaries \
  -t clef:rocm-binaries .

# Runtime-only build using the prepared binary image.
docker build -f experimental/rocm/Dockerfile.runtime \
  --build-arg CLEF_BINARIES_IMAGE=clef:rocm-binaries -t clef:rocm-runtime .
```

The tested RX 580 image retains 8192 text tokens, 4096 image-request language
tokens, the raw-patch guard, native block 64, saved rocBLAS replay, prefix/image
caches and the single FIFO worker. Production batching remains off.

## Validation and boundaries

On October 5, 2026, Flash and Full on an RTX 3090 passed the existing text,
vision, cache, pooling, 8K continuation, discovery, error and concurrent-request
API regressions with an empty Triton cache, optimized FLA enabled and no general
compiler/build executable available. CPU checks cover artifact tampering,
version/ABI mismatch and path escape. Other CUDA families retain their native
cubin coverage but were not physically retested during this packaging change.

The reconstructed RX 580 image passed FP16 GEMM, normalization, math attention,
convolution, NF4 quantization/dequantization, compact embedding and linear-output
checks, then live vision/cache/queue/discovery/413 regressions. The lean runtime
is running at port 8085, with the previous container preserved for rollback.
Its uncompressed image size fell from 15.96 GB to 7.40 GB (about 54%). Image size
is not a vulnerability count; retained libraries still need maintenance and scans.

`scripts/assert_runtime_tools.py` checks the absence of general compiler/build
executables. It deliberately does not claim absence of embedded GPU compilation.
