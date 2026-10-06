"""Hardware policy, independent of Torch so device selection happens first."""
from dataclasses import asdict, dataclass
import os
import ast
from pathlib import Path
from importlib.util import find_spec
import json
from importlib import metadata
from .kernel_arches import installed_arches, parse_arches


def environment_kernel_arches():
    # Native Full/Flash virtual environments may contain different wheels even
    # when they share serving source. Prefer the selected environment's manifest.
    try:
        manifest = metadata.distribution("causal-conv1d").locate_file("causal_conv1d/clef-kernel-build.json")
    except metadata.PackageNotFoundError:
        return installed_arches()
    if manifest.exists():
        return parse_arches(",".join(json.loads(manifest.read_text())["architectures"]))
    return installed_arches()


@dataclass(frozen=True)
class Strategy:
    gpu_name: str
    capability: str
    total_mib: int
    profile: str
    compute_dtype: str
    linear_prefill_backend: str
    fla_max_chunk_tokens: int
    tested_family: bool
    kernel_arch_supported: bool
    compiled_kernel_arches: tuple
    runtime_backend: str = "cuda"

    def metadata(self):
        return asdict(self)


def choose_strategy(name, major, minor, total_mib, profile, kernel_arches=None):
    if profile not in {"flash", "full"}:
        raise ValueError("Model must be flash or full")
    if (major, minor) < (7, 5):
        raise ValueError("This CUDA build requires NVIDIA SM75 or newer")
    if profile == "full" and (major < 8 or total_mib < 22000):
        raise ValueError("Full NF4 requires a BF16-capable GPU with at least 22 GiB VRAM; use Flash on this GPU")
    if profile == "flash" and total_mib < 7800:
        raise ValueError("Flash compact NF4 requires at least 8 GB VRAM")
    tested = (major, minor) in {(7, 5), (8, 6)}
    arches = tuple(kernel_arches) if kernel_arches is not None else environment_kernel_arches()
    # Select the optimized path only when this installation contains its cubin.
    supported = f"{major}{minor}" in arches
    backend = ("adaptive" if (major, minor) == (7, 5) else "fla") if supported else "torch"
    return Strategy(name, f"{major}.{minor}", total_mib, profile,
                    "float16" if profile == "flash" else "bfloat16",
                    backend, 512, tested, supported, arches)


def runtime_backend_from_build():
    """Read Torch's generated build metadata without importing/initializing Torch."""
    spec = find_spec("torch")
    if spec is None or not spec.origin:
        raise RuntimeError("Install the CUDA or community ROCm Torch runtime before startup")
    version_file = Path(spec.origin).parent / "version.py"
    for node in ast.parse(version_file.read_text()).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "hip" for target in targets):
            return "rocm" if ast.literal_eval(value) else "cuda"
    raise RuntimeError("Torch build metadata does not identify its HIP runtime")


def prepare_runtime(runtime_backend):
    """Set driver/allocator policy before importing Torch and initializing a GPU."""
    if runtime_backend == "rocm":
        if os.environ.get("CLEF_EXPERIMENTAL_ROCM") != "1":
            raise ValueError("ROCm is experimental; use --experimental-rocm or CLEF_EXPERIMENTAL_ROCM=1")
        os.environ.setdefault("HSA_ENABLE_SDMA", "0")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")
        os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
        os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
        os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
        os.environ["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] = "0"
    else:
        if os.environ.get("CLEF_EXPERIMENTAL_ROCM") == "1":
            raise ValueError("ROCm was requested, but the installed Torch build is not HIP; use the ROCm image")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def select_gpu(selector=None, *, runtime_backend=None):
    """Mask before importing Torch; preserve UUID masks from the container host."""
    if (runtime_backend or runtime_backend_from_build()) == "rocm":
        # HIP uses Torch's cuda API, but has a separate visibility namespace.
        mask = selector if selector is not None else os.environ.get("HIP_VISIBLE_DEVICES", "0")
        if not mask or "," in mask:
            raise ValueError("HIP_VISIBLE_DEVICES must select exactly one GPU")
        if os.environ.get("ROCR_VISIBLE_DEVICES"):
            raise ValueError("Use HIP_VISIBLE_DEVICES only; combining HIP and ROCR masks can remap GPU indices")
        os.environ["HIP_VISIBLE_DEVICES"] = mask
        return
    if selector is not None:
        if not selector:
            raise ValueError("Select exactly one GPU index or UUID")
        if "," in selector:
            raise ValueError("Select exactly one GPU index or UUID")
        os.environ["CUDA_VISIBLE_DEVICES"] = selector
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        # All-visible containers are also confined to one GPU, not sharded.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif not os.environ["CUDA_VISIBLE_DEVICES"] or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly one GPU")


def detect_strategy(profile):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one GPU must be visible; use --gpu and expose the matching runtime devices")
    prop = torch.cuda.get_device_properties(0)
    if torch.version.hip:
        return choose_rocm_strategy(prop.name, prop.gcnArchName, prop.total_memory // 2**20, profile,
                                    enabled=os.environ.get("CLEF_EXPERIMENTAL_ROCM") == "1")
    return choose_strategy(prop.name, prop.major, prop.minor, prop.total_memory // 2**20, profile)


def choose_rocm_strategy(name, architecture, total_mib, profile, *, enabled=False):
    """Narrow opt-in policy; no claim of general AMD or Full support."""
    arch = architecture.split(":", 1)[0]
    if not enabled:
        raise ValueError("ROCm is experimental; explicitly set CLEF_EXPERIMENTAL_ROCM=1")
    if arch != "gfx803" or profile != "flash" or total_mib < 7800:
        raise ValueError("Experimental ROCm currently targets gfx803, 8GB, Clef Flash only")
    return Strategy(name, arch, total_mib, profile, "float16", "torch", 512,
                    True, False, (), runtime_backend="rocm")


# Measured RX 580 profile. Keep experiment choices out of the serving defaults.
ROCM_DEFAULTS = {
    "CLEF_ATTENTION_BACKEND": "math",
    "CLEF_LINEAR_PREFILL_BACKEND": "torch",
    "CLEF_REQUIRE_FAST_LINEAR_ATTENTION": "0",
    "CLEF_DISABLE_FUSED_KERNELS": "1",
    "CLEF_PREFILL_INITIAL_CHUNK_TOKENS": "1024",
    "CLEF_PREFILL_CHUNK_TOKENS": "1024",
    "CLEF_ROCM_LONG_PROMPT_THRESHOLD": "4096",
    "CLEF_ROCM_LONG_PREFILL_CHUNK_TOKENS": "512",
    "CLEF_CONTEXT_LIMIT_MODE": "fixed",
    "CLEF_MAX_LENGTH": "8192",
    "CLEF_MAX_IMAGE_LENGTH": "4096",
    "CLEF_MAX_VISION_PATCH_TOKENS_PER_IMAGE": "8192",
    "CLEF_ROCM_GPU_CACHE": "1",
    "CLEF_ROCM_VISION_TILING": "1",
    "CLEF_VISION_QUERY_CHUNK_TOKENS": "256",
    "CLEF_CACHE_PROMOTE_REPEATED_BOUNDARY": "1",
    "CLEF_IMAGE_PREFILL": "1",
    "CLEF_TORCH_GDN_BLOCK_TOKENS": "64",
    "CLEF_ROCM_OBSERVED_PREFILL_WORKSPACE": "0",
    "CLEF_IMAGE_POOLING": "0",
    "CLEF_PREFIX_CACHE_RESERVE_MIB": "256",
    "CLEF_PREFIX_CACHE_PREFILL_RESERVE_MIB": "512",
}


def configure_strategy(strategy):
    defaults = {
        "CLEF_LINEAR_PREFILL_BACKEND": strategy.linear_prefill_backend,
        "CLEF_FLA_MAX_CHUNK_TOKENS": str(strategy.fla_max_chunk_tokens),
        "CLEF_REQUIRE_FAST_LINEAR_ATTENTION": "1" if strategy.kernel_arch_supported else "0",
        "CLEF_ATTENTION_BACKEND": "efficient",
        "CLEF_IMAGE_PREFILL": "1",
        "CLEF_PREFILL_INITIAL_CHUNK_TOKENS": "8192",
        "CLEF_PREFILL_CHUNK_TOKENS": "4096",
        "CLEF_BATCH_MAX_SIZE": "2" if strategy.runtime_backend == "cuda" and strategy.capability == "8.6" and strategy.kernel_arch_supported else "1",
        "CLEF_BATCH_PADDED_TOKENS": "6000" if strategy.total_mib < 16000 or strategy.profile == "full" else "12000",
        "CLEF_BATCH_RECORD_TOKENS": "4096",
        "CLEF_PREFIX_CACHE": "1",
        "CLEF_INPUT_CACHE": "1",
        "CLEF_IMAGE_POOLING": "0",
        "CLEF_MAX_IMAGE_LENGTH": "auto",
        "CLEF_ACTIVE_CONTEXT_OFFLOAD": "auto" if strategy.runtime_backend == "cuda" else "none",
        "CLEF_PREFIX_SHARED_HOST_BLOCKS": "1" if strategy.runtime_backend == "cuda" else "0",
    }
    if strategy.runtime_backend == "rocm":
        if os.environ.get("CLEF_BATCH_MAX_SIZE", "1") != "1":
            raise ValueError("Queue batching is currently CUDA-only; RX 580 stays at batch one")
        defaults.update(ROCM_DEFAULTS)
        tuning = Path(__file__).resolve().parents[1] / "tuning" / "gfx803.csv"
        defaults["CLEF_ROCM_TUNABLEOP_FILE"] = str(tuning) if tuning.is_file() else ""
        if os.environ.get("CLEF_ATTENTION_BACKEND", "math") != "math":
            raise ValueError("Experimental gfx803 requires CLEF_ATTENTION_BACKEND=math")
        # `auto` is a CLI compatibility choice; on gfx803 it must resolve to native.
        if os.environ.get("CLEF_LINEAR_PREFILL_BACKEND") == "auto":
            os.environ["CLEF_LINEAR_PREFILL_BACKEND"] = "torch"
        for key, ceiling in [("CLEF_MAX_LENGTH", 8192), ("CLEF_MAX_IMAGE_LENGTH", 4096),
                             ("CLEF_MAX_VISION_PATCH_TOKENS_PER_IMAGE", 8192)]:
            value = os.environ.get(key, defaults[key])
            if not value.isdigit() or not 1 <= int(value) <= ceiling:
                raise ValueError(f"Experimental gfx803 requires {key} between 1 and {ceiling}")
        if os.environ.get("CLEF_CONTEXT_LIMIT_MODE", "fixed") != "fixed":
            raise ValueError("Experimental gfx803 uses fixed tested admission limits")
    overrides = {key: os.environ[key] for key, value in defaults.items()
                 if key in os.environ and os.environ[key] != value}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if not strategy.kernel_arch_supported:
        if os.environ["CLEF_LINEAR_PREFILL_BACKEND"] in {"fla", "adaptive"}:
            raise ValueError("This GPU requires a matching convolution wheel before enabling FLA")
        if os.environ["CLEF_REQUIRE_FAST_LINEAR_ATTENTION"] == "1":
            raise ValueError(f"The installed optimized wheel targets {strategy.compiled_kernel_arches}; rebuild for this GPU")
        os.environ["CLEF_DISABLE_FUSED_KERNELS"] = "1"
    effective = strategy.metadata()
    effective.update(
        strategy_id="gfx803-flash-native" if strategy.runtime_backend == "rocm" else f"cuda-{strategy.linear_prefill_backend}-{strategy.profile}",
        experimental=strategy.runtime_backend == "rocm",
        runtime_options={key: os.environ[key] for key in defaults},
        overrides=overrides,
    )
    for key in ["HSA_ENABLE_SDMA", "PYTORCH_ALLOC_CONF", "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"]:
        if key in os.environ:
            effective["runtime_options"][key] = os.environ[key]
    if os.environ.get("CLEF_PREBUILT_LAUNCHERS_INFO"):
        effective["prebuilt_host_launchers"] = json.loads(os.environ["CLEF_PREBUILT_LAUNCHERS_INFO"])
    os.environ["CLEF_STARTUP_STRATEGY"] = json.dumps(effective)
    return effective
