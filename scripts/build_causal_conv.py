"""Build the saved causal-conv1d wheel in a CUDA development container.

Run with either profile's Python 3.11 environment, installed from a portable
Python distribution (e.g. uv). Kernel implementation remains unmodified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
import re
import zipfile
from importlib import metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clef_service.kernel_arches import SUPPORTED_ARCHES, compiler_flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "kernels")
    parser.add_argument("--arch", choices=["all", *SUPPORTED_ARCHES], default="all",
                        help="Default: all supported targets. 75=2080 Ti, 80=A100, 86=30-series, 89=4090, 90=H100, 120=5090.")
    parser.add_argument("--build-only", action="store_true", help="Produce a wheel without changing this environment.")
    args = parser.parse_args()
    arches = SUPPORTED_ARCHES if args.arch == "all" else (args.arch,)
    if sys.version_info[:2] != (3, 11) or sys.platform != "linux":
        parser.error("The saved wheel recipe targets Linux x86_64 / Python 3.11.")
    if sys.prefix == sys.base_prefix or Path(sys.base_prefix).resolve() in {Path("/usr"), Path("/usr/local")}:
        parser.error("Use a virtual environment backed by portable Python 3.11 (see docs).")
    out = args.output.resolve()
    wheels = out / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    info = json.loads((ROOT / "profiles" / "causal-conv1d-source.json").read_text())
    archive = out / "causal_conv1d-1.7.0.tar.gz"
    urllib.request.urlretrieve(info["source_url"], archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != info["sha256"]:
        raise RuntimeError("Source archive checksum mismatch")
    source = out / "source"
    source.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(source, filter="data")
    base = source / "causal_conv1d-1.7.0"
    setup = base / "setup.py"
    original = setup.read_text()
    marker = "    # HACK: The compiler flag"
    if original.count(marker) != 1:
        raise RuntimeError("Unexpected upstream build script")
    setup.write_text(original.replace(marker, f"    cc_flag = {compiler_flags(arches)!r}\n" + marker)
                    .replace('["--threads", "4"]', '["--threads", "2"]'))
    venv = Path(sys.prefix).absolute()
    portable = Path(sys.base_prefix).parent.absolute()
    python = str(venv / "bin" / "python")
    image = "nvidia/cuda:12.9.1-devel-ubuntu24.04"
    cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "HOME=/tmp", "-e", "CUDA_HOME=/usr/local/cuda", "-e", "MAX_JOBS=2",
           "-e", "CAUSAL_CONV1D_FORCE_BUILD=TRUE",
           "-e", f"PATH={venv / 'bin'}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
           "-v", f"{venv}:{venv}:ro", "-v", f"{portable}:{portable}:ro", "-v", f"{out}:{out}",
           image, python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
           "--wheel-dir", str(wheels), str(base)]
    subprocess.run(cmd, check=True)
    wheel = next(wheels.glob("causal_conv1d-1.7.0-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        extension = next(name for name in archive.namelist() if name.endswith(".so"))
        binary = out / "convolution.so"
        binary.write_bytes(archive.read(extension))
    coverage = subprocess.check_output(["docker", "run", "--rm", "-v", f"{out}:{out}:ro",
                                       image, "cuobjdump", "--list-elf", str(binary)], text=True)
    if set(re.findall(r"sm_(\d+)", coverage)) != set(arches):
        raise RuntimeError("Built convolution wheel does not contain all requested CUDA targets")
    manifest = {"architectures": list(arches), "version": info["version"],
                "source_sha256": info["sha256"], "coverage_verified": True,
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    (out / "clef-kernel-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.build_only:
        subprocess.run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
        package_manifest = metadata.distribution("causal-conv1d").locate_file("causal_conv1d/clef-kernel-build.json")
        package_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run([python, "-c", "from transformers.models.qwen3_5 import modeling_qwen3_5 as m; assert m.is_fast_path_available; print('Fast linear-attention path available')"], check=True)
    (out / "build-info.json").write_text(json.dumps({**info, "image": image,
        "architectures": list(arches), "installed": not args.build_only,
        "changes": f"CUDA architecture flags restricted to {arches}; nvcc compilation threads reduced from four to two. Kernel source unchanged.",
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
