"""Build and verify the checksum-pinned multi-architecture convolution wheel."""
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

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from clef_service.kernel_arches import SUPPORTED_ARCHES, compiler_flags, parse_arches
arches = parse_arches(os.environ.get("CLEF_CUDA_ARCHES", ",".join(SUPPORTED_ARCHES)))
info = json.loads((root / "profiles/causal-conv1d-source.json").read_text())
work = Path("/tmp/clef-kernel-build")
work.mkdir(exist_ok=True)
archive = work / "source.tar.gz"
urllib.request.urlretrieve(info["source_url"], archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["sha256"], "Kernel source checksum mismatch"
with tarfile.open(archive) as tar:
    tar.extractall(work, filter="data")
source = work / f"causal_conv1d-{info['version']}"
setup = source / "setup.py"
text = setup.read_text()
marker = "    # HACK: The compiler flag"
assert text.count(marker) == 1, "Unexpected upstream build script"
text = text.replace(marker, f"    cc_flag = {compiler_flags(arches)!r}\n" + marker)
setup.write_text(text.replace('["--threads", "4"]', '["--threads", "2"]'))
os.environ.update(CAUSAL_CONV1D_FORCE_BUILD="TRUE", MAX_JOBS="2", CUDA_HOME="/usr/local/cuda")
subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                "--wheel-dir", "/wheels", str(source)], check=True)
wheel = next(Path("/wheels").glob("causal_conv1d-*.whl"))
with zipfile.ZipFile(wheel) as archive:
    extension = next(name for name in archive.namelist() if name.endswith(".so"))
    binary = work / "convolution.so"
    binary.write_bytes(archive.read(extension))
coverage = subprocess.check_output(["cuobjdump", "--list-elf", str(binary)], text=True)
compiled = set(re.findall(r"sm_(\d+)", coverage))
if compiled != set(arches):
    raise RuntimeError(f"Convolution cubin coverage {compiled} differs from requested {arches}")
Path("/wheels/clef-kernel-build.json").write_text(json.dumps({
    "architectures": list(arches), "version": info["version"],
    "source_sha256": info["sha256"], "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "coverage_verified": True,
}, indent=2) + "\n")
print(f"Verified native cubins for {arches}", flush=True)
