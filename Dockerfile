# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11.16-slim-bookworm
FROM ${PYTHON_IMAGE} AS python

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build
COPY --from=python /usr/local/ /usr/local/
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3t64 libffi8 libsqlite3-0 libbz2-1.0 liblzma5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
RUN python3 -m venv /opt/venv
WORKDIR /src
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/unified.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
COPY profiles/causal-conv1d-source.json profiles/causal-conv1d-source.json
COPY scripts/build_container_kernel.py scripts/build_container_kernel.py
COPY clef_service/__init__.py clef_service/kernel_arches.py clef_service/
ARG CLEF_CUDA_ARCHES=75,80,86,89,90,120
ENV CLEF_CUDA_ARCHES=${CLEF_CUDA_ARCHES}
RUN python scripts/build_container_kernel.py \
    && pip install --no-cache-dir --no-deps /wheels/causal_conv1d-*.whl
RUN pip uninstall -y ninja && rm -rf /opt/venv/lib/python3.11/site-packages/torch/include
COPY artifacts/triton-host/ /artifacts/triton-host/

# Export this target once for runtime-only customer builds.
FROM scratch AS cuda-binaries
COPY --from=build /opt/venv /opt/venv
COPY --from=build /wheels/clef-kernel-build.json /kernel-build.json
COPY --from=build /artifacts/ /artifacts/

FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS runtime
COPY --from=python /usr/local/ /usr/local/
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3t64 libffi8 libsqlite3-0 libbz2-1.0 liblzma5 ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd ubuntu >/dev/null; then userdel ubuntu; fi \
    && if getent group ubuntu >/dev/null; then groupdel ubuntu; fi \
    && groupadd --gid 1000 clef && useradd --uid 1000 --gid 1000 --create-home clef \
    && mkdir -p /data && chown clef:clef /data
COPY --from=cuda-binaries /opt/venv /opt/venv
COPY --from=cuda-binaries /artifacts /app/artifacts
RUN rm -rf /usr/local/include
WORKDIR /app
COPY clef_service/ clef_service/
COPY --from=build /wheels/clef-kernel-build.json clef_service/kernel-build.json
COPY runtime/ runtime/
COPY profiles/ profiles/
COPY vendor/ vendor/
COPY web/ web/
COPY scripts/kernel_smoke.py scripts/kernel_smoke.py
COPY scripts/assert_runtime_tools.py scripts/assert_runtime_tools.py
COPY THIRD_PARTY.md README.md LICENSE ./
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CLEF_PREBUILT_LAUNCHERS_DIR=/app/artifacts/triton-host \
    CLEF_CACHE_DIR=/data NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python scripts/assert_runtime_tools.py
USER clef
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=4)"
ENTRYPOINT ["python", "-m", "clef_service"]
