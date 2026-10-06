"""CPU checks of GPU selection, strategy policies, and checkpoint lifecycle."""
import json
import multiprocessing
import time
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clef_service.hardware import choose_strategy, configure_strategy, select_gpu, environment_kernel_arches
from clef_service.kernel_arches import SUPPORTED_ARCHES, compiler_flags, installed_arches, parse_arches
from clef_service import models


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError("Expected rejection")


for name, major, minor, memory, profile, backend, dtype in [
    ("2080 Ti", 7, 5, 11264, "flash", "adaptive", "float16"),
    ("3070 Ti", 8, 6, 8192, "flash", "fla", "float16"),
    ("3090", 8, 6, 24576, "flash", "fla", "float16"),
    ("3090", 8, 6, 24576, "full", "fla", "bfloat16"),
    *[(name, major, minor, memory, profile, "fla", dtype)
      for name, major, minor, memory in [("4090", 8, 9, 24576), ("5090", 12, 0, 32768),
                                       ("A100 40GB", 8, 0, 40960), ("A100 80GB", 8, 0, 81920),
                                       ("H100", 9, 0, 81920)]
      for profile, dtype in [("flash", "float16"), ("full", "bfloat16")]],
]:
    strategy = choose_strategy(name, major, minor, memory, profile, kernel_arches=SUPPORTED_ARCHES)
    assert strategy.linear_prefill_backend == backend and strategy.compute_dtype == dtype
    assert strategy.kernel_arch_supported
    assert strategy.tested_family == ((major, minor) in {(7, 5), (8, 6)})
    with patch.dict(os.environ, {}, clear=True):
        configure_strategy(strategy)
        assert os.environ["CLEF_BATCH_MAX_SIZE"] == ("2" if (major,minor)==(8,6) else "1")
        assert os.environ["CLEF_LINEAR_PREFILL_BACKEND"] == backend
        assert os.environ["CLEF_ACTIVE_CONTEXT_OFFLOAD"] == "auto"
        assert os.environ["CLEF_PREFIX_SHARED_HOST_BLOCKS"] == "1"
        assert (os.environ.get("CLEF_DISABLE_FUSED_KERNELS") == "1") == (not strategy.kernel_arch_supported)

# A reduced/custom build must not advertise optimized kernels it doesn't contain.
for major, minor in [(8, 0), (8, 9), (9, 0), (12, 0), (12, 1)]:
    strategy = choose_strategy("uncovered", major, minor, 32768, "flash", kernel_arches=("75", "86"))
    assert not strategy.kernel_arch_supported and strategy.linear_prefill_backend == "torch"
    with patch.dict(os.environ, {}, clear=True):
        configure_strategy(strategy)
        assert os.environ["CLEF_DISABLE_FUSED_KERNELS"] == "1"
    with patch.dict(os.environ, {"CLEF_LINEAR_PREFILL_BACKEND": "fla"}, clear=True):
        rejects(lambda: configure_strategy(strategy))
assert parse_arches("75,120,75") == ("75", "120")
assert choose_strategy("2080 Ti", 7, 5, 11264, "flash", kernel_arches=("89",)).linear_prefill_backend == "torch"
rejects(lambda: parse_arches("75,999"))
assert compiler_flags(("89", "120")) == ["-gencode", "arch=compute_89,code=sm_89", "-gencode", "arch=compute_120,code=sm_120"]
with TemporaryDirectory() as folder:
    path = Path(folder) / "kernel-build.json"
    with patch("clef_service.kernel_arches.MANIFEST", path):
        assert installed_arches() == ("75", "86")
        path.write_text(json.dumps({"architectures": list(SUPPORTED_ARCHES)}))
        assert installed_arches() == SUPPORTED_ARCHES
    with patch("clef_service.hardware.metadata.distribution") as distribution:
        distribution.return_value.locate_file.return_value = path
        path.write_text(json.dumps({"architectures": ["120"]}))
        assert environment_kernel_arches() == ("120",)
        assert choose_strategy("5090", 12, 0, 32768, "full").kernel_arch_supported
        assert choose_strategy("3090", 8, 6, 24576, "full").linear_prefill_backend == "torch"
for major, minor, memory, profile in [(6, 1, 24000, "flash"), (7, 5, 11264, "full"),
                                     (8, 6, 8192, "full"), (8, 6, 6000, "flash")]:
    rejects(lambda: choose_strategy("test", major, minor, memory, profile))
with patch.dict(os.environ, {}, clear=True):
    select_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    select_gpu("GPU-example")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-example"
    rejects(lambda: select_gpu("0,1"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    rejects(lambda: select_gpu())

# Fake tiny checkpoints exercise actual validation and atomic lifecycle logic.
def prepared(source, folder, profile):
    config = models.profile_config(profile)
    for name in ["config.json", "joint_head.safetensors", "joint_head_config.json", "tokenizer.json",
                 "processor_config.json", "joint_schema_model.py", "LICENSE", "model.safetensors",
                 "compact_quantization.json", "input-embedding-nf4.safetensors", "output-embedding-nf4.safetensors"]:
        (folder / name).write_text("test")
    (folder / "joint_schema_model.py").write_bytes((models.ROOT / "vendor/cloudflare/joint_schema_model.py").read_bytes())
    (folder / "config.json").write_text(json.dumps({"quantization_config": {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16"}}))
    files = models.checkpoint_files(folder, True)
    (folder / "clef-checkpoint.json").write_text(json.dumps({"format_version": models.FORMAT_VERSION,
        "profile": profile, "revision": config["revision"],
        "files": {name: (folder / name).stat().st_size for name in files}}))


with TemporaryDirectory() as tmp:
    data = Path(tmp)
    partial = data / ".model-nf4-compact.preparing"
    partial.mkdir()
    (partial / "interrupted.txt").write_text("partial work")
    with patch.object(models, "download_source", return_value=data / "source") as download, \
            patch.object(models, "prepare_checkpoint", side_effect=prepared) as prepare:
        checkpoint = models.ensure_checkpoint(data, "flash")
        assert not partial.exists() and not (checkpoint / "interrupted.txt").exists()
        assert download.call_count == prepare.call_count == 1
        assert models.ensure_checkpoint(data, "flash", offline=True) == checkpoint
        assert download.call_count == prepare.call_count == 1  # no network/quantization on cached startup
        rejects(lambda: models.validate_checkpoint(checkpoint, "full"))
        (checkpoint / "model.safetensors").write_text("truncated")
        rejects(lambda: models.ensure_checkpoint(data, "flash"))
    # Explicit old checkpoints need no managed manifest, but must match profile.
    fresh = data / "external"
    fresh.mkdir()
    prepared(None, fresh, "flash")
    (fresh / "clef-checkpoint.json").unlink()
    assert models.ensure_checkpoint(data, "flash", external=fresh) == fresh.resolve()
    rejects(lambda: models.validate_checkpoint(fresh, "flash"))

# Two startup processes sharing a volume must publish exactly one checkpoint.
with TemporaryDirectory() as tmp:
    data = Path(tmp)
    def counted_prepare(source, folder, profile):
        with (data / "preparation-count").open("a") as out:
            out.write("prepared\n")
        time.sleep(0.15)
        prepared(source, folder, profile)
    def startup():
        with patch.object(models, "download_source", return_value=data / "source"), \
                patch.object(models, "prepare_checkpoint", side_effect=counted_prepare):
            models.ensure_checkpoint(data, "flash")
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=startup) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert (data / "preparation-count").read_text() == "prepared\n"

print("PASS: hardware matrix, single GPU isolation, cache reuse, interrupted work, provenance and truncation rejection")
