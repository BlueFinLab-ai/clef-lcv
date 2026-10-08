# Container vulnerability scans

Scan the final CUDA and experimental ROCm runtime images separately. Flash and
Full share the CUDA image. Model weights, runtime caches and host GPU drivers are
outside the image scan. Scanner results describe known dependency vulnerabilities;
they do not establish whether a particular API request can exploit them.

## Run locally

Build the image using the current Dockerfile, then:

```sh
python3 scripts/security_scan.py --image clef:local --output-dir local/security-cuda
python3 scripts/security_scan.py --image clef:rocm-runtime --output-dir local/security-rocm
```

The script exports an image archive, pulls digest-pinned Trivy 0.75.0 and Grype
0.120.0 scanner images, initializes their databases, and scans the active final
filesystem. Scanner containers run without privileges or added capabilities as
the calling user. They receive a read-only image archive, private scratch space
and a database cache. No Docker socket, model volume, API keys or GPUs are mounted.
The scanners are developer tools and are not included in the application image.

Results include both raw JSON reports, stderr logs, a Trivy version/database
record, Grype database metadata and a summary. Exit codes:

- `0`: neither scanner reports high or critical findings.
- `1`: either scanner reports at least one high or critical finding.
- `2`: a scanner, database, archive or report failed validation.

Unfixed findings are included. There is no vulnerability ignore list, severity
reclassification or `only-fixed` filtering. Counts are package/advisory matches,
so one CVE can appear against several installed GnuPG packages or vendored copies.
The two scanners use different advisory sources and may report different counts.
Keep both reports rather than treating their counts as interchangeable.

For an existing export or a controlled before/after comparison:

```sh
python3 scripts/security_scan.py --archive /path/to/image.tar \
  --output-dir local/security-comparison --cache-dir local/security-cuda/.cache \
  --skip-db-update
```

The cache must contain completed databases and be readable/writable by the calling
user. Normal runs update databases first. Snapshot reuse avoids attributing a
newly published advisory to a package change during the comparison. Archive and
scratch files may require tens of GB; choose a disk-backed output directory.

## Release check

The `Container vulnerability scan` GitHub workflow accepts a public image
reference or digest through manual dispatch, runs both scanners and saves reports
even when the gate fails. Run it on each final image before publishing a release.
Its actions and scanner images are pinned to immutable commits/digests. It does
not build GPU kernels or need a GPU runner.

Trivy's March 2026 supply-chain incident affected specific releases and action
tags. The pinned scanner is a later release; see the
[maintainer advisory](https://github.com/aquasecurity/trivy/security/advisories/GHSA-69fq-xp46-6x23).
Review scanner updates and their provenance before changing pins.

## October 6, 2026 remediation

The CUDA runtime baseline reported 0 critical / 15 high matches in Trivy and
0 critical / 16 high in Grype. The fixes are:

- Python 3.11.17, retaining the 3.11 ABI for the existing GPU extensions. This
  fixes [CVE-2026-82049 and other interpreter security issues](https://www.python.org/downloads/release/python-31117/).
- Setuptools 84.0.0 in both the global interpreter and service venv, including
  updated vendored jaraco.context and wheel. Updating just a top-level standalone
  wheel would leave setuptools' vendored copies vulnerable.
- Ubuntu package upgrades in both final CUDA build routes and removal of unused
  GnuPG client/agent packages. The retained `gpgv` verifier is updated by Ubuntu.
- Runtime-only builds also update old precompiled payloads' setuptools rather
  than assuming that an older binaries image already contains the fix.

The RX 580 baseline reported 0 critical / 4 high matches in Trivy and
0 critical / 17 high in Grype. Grype found bundled FFmpeg libraries in the unused
OpenCV package, which both ROCm Docker routes now remove. The four Trivy findings
came from pip's vendored dependency inventory in **both** interpreter prefixes:
msgpack 1.1.2, setuptools 70.3.0 and urllib3 2.7.0. These are separate copies from
the top-level installed packages; updating top-level urllib3/setuptools does not
repair pip's vendor tree. The latest available pip (26.2.1) still included these
copies at review time.

Final images remove pip and ensurepip entirely, including their vendored code,
metadata, bundled installer wheels and console commands. Build stages keep the
installation tools. Model downloading/preparation uses the preinstalled Hugging
Face/PyTorch dependencies and remains available. Rebuild the image for dependency
changes rather than installing packages into the running service. No SBOM-only
deletion, version relabeling or suppression is used. The ROCm interpreter was
already Python 3.12.15 and remains on that ABI.

The CUDA compiled kernels and GPU runtime wheels are preserved. No compiler
packages or scanner dependencies are added to either application runtime.

## Final scan and deployment results

Both final runtime images passed both tools with **0 critical and 0 high**
findings on the October 6 database snapshots. Remaining medium counts were:
CUDA Trivy 9 / Grype 168; ROCm Trivy 31 / Grype 101. Low and negligible findings
also remain. Different catalogs/advisories account for scanner disagreement;
these results are not a claim of zero vulnerabilities.

Security-only images preserve the deployed application code and were installed
on the Flash RTX 3070 Ti, Full 400W RTX 3090 and Flash RX 580 services. Ports and
configuration were preserved, with stopped pre-update containers kept for rollback.
CUDA Flash/Full passed warmup, typed text/image requests and exact-image cache
reuse; all three live services passed a synthetic PNG and typed-answer request
after installer removal. Runtime imports confirmed pip/ensurepip unavailable and
OpenCV absent from ROCm. Restarting clears the process-local input/prefix caches.

GPU kernels and preinstalled libraries are preserved; no new compiler packages
were added. General compiler checks and prebuilt-launcher ABI checks passed.
The default full compiler stage was updated in source but was not rebuilt during
this audit; the runtime-only CUDA route was built with existing compiled kernels.
Other GPU families were not available for hardware retesting.

## CUDA runtime trimming

The default final CUDA image removes `cuda-compat-12-9` and five leftover GnuPG
support packages: `libksba8`, `libldap2`, `libsasl2-2`,
`libsasl2-modules-db` and `pinentry-curses`. The server uses host driver
610.57.04; its injected `libcuda.so` was confirmed loaded instead of the bundled
575.57.08 compatibility driver. Grype's 102 compatibility-package matches were
indirect Ubuntu source-package matches with unknown version constraints and a
`wont-fix` status; they were not 102 demonstrated vulnerabilities in the service.

The documented host requirement remains driver R575 or newer with CUDA 12.9+
support. Older-driver forward compatibility is not enabled by default. To retain
the package for an explicitly validated deployment, build with
`--build-arg CLEF_INCLUDE_CUDA_COMPAT=1`. This changes image construction, not
runtime GPU selection. Verify NVIDIA's supported GPU/driver combinations before
using that opt-in. A host driver exposing required PTX/JIT capabilities remains
necessary even when the PyTorch CUDA runtime is bundled.

Removal uses an explicit list. General `apt autoremove` is avoided because apt
does not know the native dependencies of the copied Python interpreter: it would
remove SQLite/readline here. SQLite, readline and compression/SSL libraries are
preserved. `tar` is required by `dpkg`; `gpgv`, `libassuan0` and `libnpth0t64`
are required by the retained package manager/verifier and are preserved too.
No force-removal or package-status editing is used.

The post-trim CUDA comparison used the same October 6 database snapshots.
Trivy stayed at 0 critical / 0 high / 9 medium. Grype went from
0 critical / 0 high / 168 medium to 0 critical / 0 high / 66 medium: all
102 compatibility-package matches disappeared. Flash and Full passed startup,
text/image, all typed heads and image-prefix cache checks on an isolated RTX 3090.
Python native-module imports passed with SSL/SQLite/readline/compression intact.
The active runtime filesystem loses approximately 300 MiB of compatibility
driver files; inherited image layers can retain their old bytes, so this is not
a claim of a 300 MiB download-size reduction.
