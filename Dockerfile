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

FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS runtime
COPY --from=python /usr/local/ /usr/local/
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3t64 libffi8 libsqlite3-0 libbz2-1.0 liblzma5 ca-certificates libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd ubuntu >/dev/null; then userdel ubuntu; fi \
    && if getent group ubuntu >/dev/null; then groupdel ubuntu; fi \
    && groupadd --gid 1000 clef && useradd --uid 1000 --gid 1000 --create-home clef \
    && mkdir -p /data && chown clef:clef /data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY clef_service/ clef_service/
COPY --from=build /wheels/clef-kernel-build.json clef_service/kernel-build.json
COPY runtime/ runtime/
COPY profiles/ profiles/
COPY vendor/ vendor/
COPY web/ web/
COPY scripts/kernel_smoke.py scripts/kernel_smoke.py
COPY THIRD_PARTY.md README.md LICENSE ./
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CLEF_CACHE_DIR=/data NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER clef
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"
ENTRYPOINT ["python", "-m", "clef_service"]
