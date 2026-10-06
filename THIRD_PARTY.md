# Third-party attribution

`vendor/cloudflare/joint_schema_model.py` is an unmodified copy of the native Clef
release wrapper from Cloudflare's Hugging Face model repositories. Both pinned
revisions ship the same file (SHA256
`0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`).

- Full: https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c
- Flash: https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6
- Upstream Apache License 2.0 is retained in `vendor/cloudflare/LICENSE`, including
  its copyright notice. Model downloads retain the release license as well.

Torch, Transformers, bitsandbytes, Flash Linear Attention, causal-conv1d, and the
other dependencies retain their own licenses. Dependency source/binaries and
model weights are not redistributed in this repository. The kernel build downloads
upstream source and records its checksum plus the build-only modifications.

The project's original code is licensed under the Apache License 2.0; see
[LICENSE](LICENSE). That license covers this repository's code only. Model weights
are downloaded from Cloudflare at runtime and remain under their release license.


The prebuilt Triton host-helper modules in `artifacts/triton-host/` are generated
from Triton 3.6.0 driver/launcher code (MIT). Its license is included beside the
manifest. Generated C sources and raw checks are retained in the Git-ignored `.dev/`
workspace; binary and source hashes remain in the distributable manifest. Additional previously exercised helper variants were extracted from
the validated service cache. The manifest binds Python ABI and dependency versions.
