"""CPU coverage of the experimental guard and single-device HIP selection."""
import os
import json
import io
import contextlib
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clef_service.hardware import choose_rocm_strategy, configure_strategy, select_gpu, prepare_runtime, runtime_backend_from_build
from clef_service.__main__ import main

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected rejection")

rejects(lambda: choose_rocm_strategy("RX 580", "gfx803", 8192, "flash"))
for arch, memory, profile in [("gfx1100", 8192, "flash"), ("gfx803", 4096, "flash"), ("gfx803", 8192, "full")]:
    rejects(lambda: choose_rocm_strategy("unsupported", arch, memory, profile, enabled=True))
strategy = choose_rocm_strategy("RX 580", "gfx803:sramecc-", 8192, "flash", enabled=True)
assert strategy.runtime_backend == "rocm" and strategy.compute_dtype == "float16"
with patch.dict(os.environ, {"CLEF_EXPERIMENTAL_ROCM": "1"}, clear=True):
    prepare_runtime("rocm")
    select_gpu(runtime_backend="rocm")
    assert os.environ["HIP_VISIBLE_DEVICES"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    select_gpu("1", runtime_backend="rocm")
    assert os.environ["HIP_VISIBLE_DEVICES"] == "1"
    rejects(lambda: select_gpu("0,1", runtime_backend="rocm"))
    configure_strategy(strategy)
    assert os.environ["CLEF_ATTENTION_BACKEND"] == "math"
    assert os.environ["CLEF_DISABLE_FUSED_KERNELS"] == "1"
    assert os.environ["CLEF_PREFILL_CHUNK_TOKENS"] == "1024"
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    rejects(lambda: select_gpu(runtime_backend="rocm"))
with patch.dict(os.environ, {"CLEF_ATTENTION_BACKEND": "efficient"}, clear=True):
    rejects(lambda: configure_strategy(strategy))
print("PASS: ROCm opt-in, architecture/profile/memory guard, HIP isolation, safe defaults")

# Generated version metadata must be read without importing Torch.
with TemporaryDirectory() as folder:
    package = Path(folder)
    origin = package / "__init__.py"
    origin.touch()
    for declaration, expected in [('hip: str | None = "7.15"', 'rocm'), ('hip = None', 'cuda')]:
        (package / "version.py").write_text(declaration)
        with patch('clef_service.hardware.find_spec', return_value=SimpleNamespace(origin=str(origin))):
            imported_before = "torch" in sys.modules
            assert runtime_backend_from_build() == expected
            assert ("torch" in sys.modules) == imported_before
with patch.dict(os.environ, {"CLEF_EXPERIMENTAL_ROCM": "1"}, clear=True):
    prepare_runtime('rocm')
    effective = configure_strategy(strategy)
    options = effective['runtime_options']
    assert options['HSA_ENABLE_SDMA'] == '0'
    assert options['PYTORCH_ALLOC_CONF'] == 'expandable_segments:False'
    assert options['CLEF_ROCM_LONG_PREFILL_CHUNK_TOKENS'] == '512'
    assert options['CLEF_MAX_LENGTH'] == '8192' and options['CLEF_MAX_IMAGE_LENGTH'] == '4096'
    assert options['CLEF_MAX_VISION_PATCH_TOKENS_PER_IMAGE'] == '8192'
    assert options['CLEF_ROCM_GPU_CACHE'] == options['CLEF_ROCM_VISION_TILING'] == '1'
    assert options['CLEF_ROCM_OBSERVED_PREFILL_WORKSPACE'] == '0'
    assert options['CLEF_TORCH_GDN_BLOCK_TOKENS'] == '64'
    assert json.loads(os.environ['CLEF_STARTUP_STRATEGY']) == json.loads(json.dumps(effective))
for invalid in [{'CLEF_LINEAR_PREFILL_BACKEND': 'fla'}, {'CLEF_MAX_LENGTH': '8193'},
                {'CLEF_MAX_IMAGE_LENGTH': 'auto'}, {'CLEF_MAX_IMAGE_LENGTH': '4097'},
                {'CLEF_MAX_VISION_PATCH_TOKENS_PER_IMAGE': '8193'}, {'CLEF_CONTEXT_LIMIT_MODE': 'auto'}]:
    with patch.dict(os.environ, invalid, clear=True):
        rejects(lambda: configure_strategy(strategy))
with patch.dict(os.environ, {}, clear=True):
    rejects(lambda: prepare_runtime('rocm'))
with patch.dict(os.environ, {'CLEF_EXPERIMENTAL_ROCM': '1'}, clear=True):
    rejects(lambda: prepare_runtime('cuda'))
with patch.dict(os.environ, {'CLEF_EXPERIMENTAL_ROCM': '1', 'HSA_ENABLE_SDMA': '1',
                             'CLEF_ROCM_TUNABLEOP_FILE': ''}, clear=True):
    prepare_runtime('rocm')
    assert os.environ['HSA_ENABLE_SDMA'] == '1'
    assert configure_strategy(strategy)['runtime_options']['CLEF_ROCM_TUNABLEOP_FILE'] == ''
# Actual CLI ordering: overrides applied before hardware defaults; inspect skips weights.
with patch.dict(os.environ, {}, clear=True), \
        patch('clef_service.__main__.runtime_backend_from_build', return_value='rocm'), \
        patch('clef_service.__main__.detect_strategy', return_value=strategy), \
        patch('clef_service.__main__.ensure_checkpoint') as checkpoint, \
        contextlib.redirect_stdout(io.StringIO()) as out:
    main(['inspect', '--experimental-rocm', '--gpu', '2', '--max-length', '4096',
          '--prefill-chunk-tokens', '512', '--allow-slow-kernels', '--no-prefix-cache'])
    report = json.loads(out.getvalue())
    assert report['runtime_options']['CLEF_MAX_IMAGE_LENGTH'] == '4096'
    assert report['runtime_options']['CLEF_LINEAR_PREFILL_BACKEND'] == 'torch'
    assert report['overrides']['CLEF_MAX_LENGTH'] == '4096'
    assert report['overrides']['CLEF_PREFILL_CHUNK_TOKENS'] == '512'
    assert os.environ['HIP_VISIBLE_DEVICES'] == '2' and os.environ['CLEF_PREFIX_CACHE'] == '0'
    checkpoint.assert_not_called()
print("PASS: pre-import runtime dispatch, measured defaults, admission guards, CLI precedence")
