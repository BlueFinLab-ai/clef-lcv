PYTHON ?= python3.11
GPU ?= 0
CUDA_ARCH ?= all
HOST ?= 127.0.0.1
MODEL ?= flash
UNIFIED_PY = .venv/unified/bin/python
FULL_PY = .venv/full/bin/python
FLASH_PY = .venv/flash/bin/python

.PHONY: env kernels serve docker-build docker-up env-full env-flash download-full download-flash prepare-full prepare-flash kernels-full kernels-flash serve-full serve-flash check

env:
	$(PYTHON) -m venv .venv/unified
	$(UNIFIED_PY) -m pip install -r requirements/unified.txt --extra-index-url https://download.pytorch.org/whl/cu128

kernels:
	$(UNIFIED_PY) scripts/build_causal_conv.py --arch $(CUDA_ARCH)

serve:
	$(UNIFIED_PY) -m clef_service --model $(MODEL) --gpu $(GPU) --host $(HOST)

docker-build:
	docker build -t clef:local .

docker-up:
	CLEF_GPU=$(GPU) CLEF_PROFILE=$(MODEL) docker compose up --build -d

env-full:
	$(PYTHON) -m venv .venv/full
	$(FULL_PY) -m pip install -r requirements/full.txt --extra-index-url https://download.pytorch.org/whl/cu128

env-flash:
	$(PYTHON) -m venv .venv/flash
	$(FLASH_PY) -m pip install -r requirements/flash.txt --extra-index-url https://download.pytorch.org/whl/cu128

download-full:
	$(FULL_PY) scripts/clef.py download full

download-flash:
	$(FLASH_PY) scripts/clef.py download flash

prepare-full:
	$(FULL_PY) scripts/clef.py prepare full --gpu $(GPU)

prepare-flash:
	$(FLASH_PY) scripts/clef.py prepare flash --gpu $(GPU)

kernels-full:
	$(FULL_PY) scripts/build_causal_conv.py --arch $(CUDA_ARCH)

kernels-flash:
	$(FLASH_PY) -m pip install --no-deps -r requirements/flash-kernels.txt
	$(FLASH_PY) scripts/build_causal_conv.py --arch $(CUDA_ARCH)

serve-full:
	$(FULL_PY) scripts/clef.py serve full --gpu $(GPU) --host $(HOST)

serve-flash:
	$(FLASH_PY) scripts/clef.py serve flash --gpu $(GPU) --host $(HOST)

check:
	$(PYTHON) scripts/check_project.py
