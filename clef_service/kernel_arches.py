"""Shared CUDA targets and installed convolution-wheel coverage (no Torch import)."""
import json
from pathlib import Path

SUPPORTED_ARCHES = ("75", "80", "86", "89", "90", "120")
MANIFEST = Path(__file__).with_name("kernel-build.json")


def parse_arches(value):
    arches = tuple(dict.fromkeys(value.split(",")))
    if not arches or any(arch not in SUPPORTED_ARCHES for arch in arches):
        raise ValueError(f"CUDA targets must be comma-separated members of {SUPPORTED_ARCHES}")
    return arches


def installed_arches():
    if not MANIFEST.exists():
        # Historical native installations have the original SM75/SM86 wheels.
        # New builds publish a manifest; merely adding policy does not add cubins.
        return ("75", "86")
    info = json.loads(MANIFEST.read_text())
    return parse_arches(",".join(info["architectures"]))


def compiler_flags(arches):
    return [flag for arch in arches
            for flag in ("-gencode", f"arch=compute_{arch},code=sm_{arch}")]
