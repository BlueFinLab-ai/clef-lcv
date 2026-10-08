# syntax=docker/dockerfile:1
ARG ROCM_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder@sha256:b65af203cf7ebdf1751127ee79333438acfc5226086e889391f675e176d5e7f4
FROM ${ROCM_IMAGE} AS build
ARG BNB_REVISION=08a9956b939857ae2c4fbd56753960cc370fa7cf
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends cmake make g++ ninja-build \
    && rm -rf /var/lib/apt/lists/*
COPY experimental/rocm/bitsandbytes-gfx803-wave64.patch /build/gfx803-wave64.patch
RUN pip install --no-cache-dir scikit-build-core==1.1.1 trove-classifiers==2026.9.21.13 wheel==0.48.0 \
    && git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git bitsandbytes \
    && cd bitsandbytes && git checkout --detach ${BNB_REVISION} \
    && git apply --check /build/gfx803-wave64.patch && git apply /build/gfx803-wave64.patch \
    && cmake -S . -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx803 \
    && cmake --build build --parallel 2 \
    && cmake -S . -B build-cpu -DCOMPUTE_BACKEND=cpu \
    && cmake --build build-cpu --parallel 2 \
    && BNB_SKIP_CMAKE=1 pip install --no-cache-dir --no-deps --no-build-isolation .
WORKDIR /app
COPY experimental/rocm/requirements.txt /tmp/clef-rocm-requirements.txt
RUN pip uninstall -y vllm mistral-common \
    && pip install --no-cache-dir -r /tmp/clef-rocm-requirements.txt \
    && python -c "import torch,bitsandbytes,transformers; assert torch.version.hip; print(torch.__version__,torch.version.hip,bitsandbytes.__version__,transformers.__version__)"
COPY clef_service/ clef_service/
COPY runtime/ runtime/
COPY profiles/ profiles/
COPY vendor/ vendor/
COPY web/ web/
COPY experimental/rocm/probe.py /app/probe.py
COPY experimental/rocm/tunable-rx580-2026-10-04.csv /app/tuning/gfx803.csv
# The shared startup policy selects the measured gfx803 defaults. The image
# supplies the community HIP dependencies and matching offline tuning table.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CLEF_EXPERIMENTAL_ROCM=1 CLEF_CACHE_DIR=/data
# Explicit hardware devices and host video/render GIDs are supplied at run time.
EXPOSE 8080
ENTRYPOINT ["python", "-m", "clef_service"]

FROM build AS collect
RUN python -m pip uninstall -y triton onnxruntime-migraphx ninja scikit-build-core trove-classifiers wheel opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
COPY experimental/rocm/collect_runtime.py /tmp/collect_runtime.py
RUN python /tmp/collect_runtime.py

FROM scratch AS rocm-binaries
COPY --from=collect /runtime-root/ /

FROM ubuntu:26.04@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 libgcc-s1 libgomp1 libnuma1 libdrm2 libdrm-amdgpu1 \
    libelf1 libzstd1 zlib1g libopenblas0-pthread libgfortran5 libssl3t64 libffi8 \
    libsqlite3-0 libbz2-1.0 liblzma5 libcrypt1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=rocm-binaries / /
WORKDIR /app
COPY clef_service/ clef_service/
COPY runtime/ runtime/
COPY profiles/ profiles/
COPY vendor/ vendor/
COPY web/ web/
COPY experimental/rocm/probe.py /app/probe.py
COPY experimental/rocm/tunable-rx580-2026-10-04.csv /app/tuning/gfx803.csv
COPY scripts/assert_runtime_tools.py scripts/assert_runtime_tools.py
COPY scripts/strip_runtime_installers.py /tmp/strip_runtime_installers.py
ENV PATH=/opt/venv/bin:/opt/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CLEF_EXPERIMENTAL_ROCM=1 CLEF_CACHE_DIR=/data \
    ROCM_PATH=/opt/rocm HSA_OVERRIDE_GFX_VERSION=8.0.3 TORCH_BLAS_PREFER_HIPBLASLT=0 \
    LD_PRELOAD=/opt/rocm/lib/libgfx803_sgemm_shim.so \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib:/opt/rocm/lib/host-math/lib:/opt/rocm/lib/rocm_sysdeps/lib
RUN printf '/opt/rocm/lib\n/opt/rocm/lib/llvm/lib\n/opt/rocm/lib/host-math/lib\n/opt/rocm/lib/rocm_sysdeps/lib\n' > /etc/ld.so.conf.d/clef-rocm.conf \
    && ldconfig && python /tmp/strip_runtime_installers.py && rm /tmp/strip_runtime_installers.py \
    && python scripts/assert_runtime_tools.py \
    && python -c "import torch,bitsandbytes; assert torch.version.hip"
EXPOSE 8080
ENTRYPOINT ["python","-m","clef_service"]
