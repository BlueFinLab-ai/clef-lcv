"""Hardware policy, independent of Torch so device selection happens first."""
from dataclasses import asdict, dataclass
import os
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


def select_gpu(selector=None):
    """Mask before importing Torch; preserve UUID masks from the container host."""
    if selector:
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
        raise RuntimeError("Exactly one CUDA GPU must be visible; use --gpu or docker --gpus device=UUID")
    prop = torch.cuda.get_device_properties(0)
    return choose_strategy(prop.name, prop.major, prop.minor, prop.total_memory // 2**20, profile)


def configure_strategy(strategy):
    os.environ["CLEF_STARTUP_STRATEGY"] = __import__("json").dumps(strategy.metadata())
    os.environ.setdefault("CLEF_LINEAR_PREFILL_BACKEND", strategy.linear_prefill_backend)
    os.environ.setdefault("CLEF_FLA_MAX_CHUNK_TOKENS", str(strategy.fla_max_chunk_tokens))
    os.environ.setdefault("CLEF_REQUIRE_FAST_LINEAR_ATTENTION", "1" if strategy.kernel_arch_supported else "0")
    if not strategy.kernel_arch_supported:
        if os.environ["CLEF_LINEAR_PREFILL_BACKEND"] in {"fla", "adaptive"}:
            raise ValueError("This GPU requires a matching convolution wheel before enabling FLA")
        if os.environ["CLEF_REQUIRE_FAST_LINEAR_ATTENTION"] == "1":
            raise ValueError(f"The installed optimized wheel targets {strategy.compiled_kernel_arches}; rebuild for this GPU")
        os.environ["CLEF_DISABLE_FUSED_KERNELS"] = "1"
