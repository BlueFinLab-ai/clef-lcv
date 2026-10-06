# Local development assets

Use `.dev/` for scratch work, third-party checkouts, local toolchains, intermediate
objects, kernel captures, logs and private testing inputs. The entire directory
is excluded from both Git and Docker build contexts. Create it as needed; empty
folders and its local README are intentionally not committed.

Generated Triton launcher C captures and their raw FP16/BF16 operator-check outputs
have been moved there. The distributable `artifacts/triton-host/` directory keeps
only the runtime binary modules, manifest and license. The public manifest retains
source hashes for provenance; the original capture manifest is kept locally.

Runtime source, reproducible build recipes, dependency pins, tests, licenses,
documentation and the sanitized sample email dataset remain in the public tree.
Compiler SDKs are still confined to Docker producer stages; ignoring a local
folder does not replace the runtime-image separation.

For new Triton captures, use the builder capture tool with a local destination:

```sh
python scripts/capture_triton_helpers.py --output-dir .dev/triton-host \
  --script scripts/kernel_smoke.py -- --dtype float16 --heads 32 \
  --output .dev/triton-host/checks/fp16.json
```

The capture command requires the pinned CUDA builder environment and GPU. When
preparing a distributable helper pack, copy only its checked runtime binaries,
license and manifest to `artifacts/triton-host/`; omit raw checks and generated
source captures from that pack.

`.gitignore` applies to untracked files. Check `git ls-files .dev` before publishing
if files were ever committed there. Do not force-add the directory. Local captures
need a separate backup if they must be retained across machines.
