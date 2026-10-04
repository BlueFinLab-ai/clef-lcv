# Full and Flash model profiles

Both profiles use the unified service described in
[Docker and startup](unified-service.md). This page covers native installs,
existing checkpoints and environment variables. The legacy `scripts/clef.py`
commands still work for older installations.

The model repository and revision are recorded in `profiles/full.json` and
`profiles/flash.json`. `download` obtains the complete pinned release, including
vision, tokenizer, processor, and decision-head files. It checks that the release
wrapper matches the vendored copy. `prepare` quantizes linear layers using NF4
with double quantization, leaving the lexical output matrix suitable for Clef's
decision head. Flash then replaces input/output embedding lookup with compact NF4
row lookup. The original decision head and vision processor are retained.

## Existing checkpoints

You can reuse a prepared checkpoint without downloading or requantizing:

```sh
python -m clef_service --gpu 0 --checkpoint /absolute/path/to/model-nf4-compact --offline
python -m clef_service --gpu 0 --full --checkpoint /absolute/path/to/model-nf4 --offline
```

Older installations that keep a whole data directory per profile can use the
legacy launcher:

```sh
.venv/full/bin/python scripts/clef.py serve full \
  --data-dir /absolute/path/to/existing/full-deployment --gpu 0

.venv/flash/bin/python scripts/clef.py serve flash \
  --data-dir /absolute/path/to/existing/flash-deployment --gpu 0 \
  --max-length 24576
```

Full expects `model-nf4/`. Flash expects `model-nf4-compact/`, including
`compact_quantization.json` and both embedding safetensors files. To select a
different checkpoint, set `CLEF_MODEL_DIR` to its absolute path or a name beneath
the data directory. Checkpoints need `joint_schema_model.py`, `joint_head.safetensors`,
`joint_head_config.json`, backbone weights/config, and processor/tokenizer files.

## Fresh preparation

`python -m clef_service download [--full]` downloads the pinned source; it needs no GPU.
`python -m clef_service prepare [--full] --gpu INDEX` creates the checkpoint and exits.
The legacy equivalents are `scripts/clef.py download|prepare PROFILE --data-dir PATH`.
Preparation validates and reuses a complete managed checkpoint. It quantizes
one source tensor at a time and publishes only a complete result with a profile,
revision and file-size manifest. Interrupted hidden staging output is rebuilt;
a malformed published checkpoint is rejected without overwriting it. Explicit
`--checkpoint PATH` can reuse complete older prepared weights without a manifest.
The legacy serve launcher detects older checkpoint directories automatically.

Both adapters keep all parameters on the selected GPU. Flash compact restoration
occurs on CPU first to avoid allocating temporary dense embeddings on an 8 GB GPU.
Full uses BF16 compute and dense embeddings; Flash uses FP16 compute. Avoid
substituting generic Qwen checkpoints: Clef also needs its trained native decision head.

## Environment controls

| Variable | Purpose |
|---|---|
| `CUDA_VISIBLE_DEVICES` or `--gpu` | GPU index or UUID; becomes `cuda:0` inside the process |
| `CLEF_DATA_DIR` or `--data-dir` | Source, checkpoint, and runtime cache directory |
| `CLEF_MODEL_DIR` | Checkpoint path or name beneath the data directory |
| `CLEF_MAX_LENGTH` or `--max-length` | Combined token limit; no silent truncation |
| `CLEF_MAX_IMAGE_PIXELS` | Maximum legacy `image_fidelity` server-resize budget; default 1,048,576. Native `media_kwargs` uses processor options. |
| `CLEF_ATTENTION_BACKEND` | `efficient` by default; `auto` allows automatic SDPA selection |
| `CLEF_REQUIRE_FAST_LINEAR_ATTENTION` | Architectures present in the installed kernel manifest default to `1`; explicit fallback sets `0` |
| `CLEF_PREFIX_CACHE`, `CLEF_PREFIX_CACHE_MIB`, `CLEF_PREFIX_CACHE_ENTRIES` | Branching prefix reuse; defaults 1, auto-sized VRAM, 32 checkpoints |
| `CLEF_IMAGE_POOLING` | Experimental 2×2 pooling; default 0 |
| `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` | Persistent kernel caches |

The launcher prepares missing weights automatically, then selects offline serving and one Uvicorn worker. A second worker would
load another model. To run Full and Flash at the same time, give each its own
GPU and its own `--port`; both default to 8080.

## systemd

`deployment/clef@.service` is a template for `clef@flash` and `clef@full`. It runs
`python -m clef_service --model %i` and assumes:

- the project and a Python environment at `/opt/Clef` and `/opt/Clef/.venv`,
- a service account named `clef`,
- a writable cache at `/var/lib/clef`, with one subdirectory per model.

Copy [profile.env.example](../deployment/profile.env.example) to
`/etc/clef/flash.env` or `/etc/clef/full.env` to choose the GPU and port. Give
each service on the same host its own GPU and port.

```sh
sudo cp deployment/clef@.service /etc/systemd/system/
sudo mkdir -p /etc/clef /var/lib/clef && sudo chown clef:clef /var/lib/clef
sudo cp deployment/profile.env.example /etc/clef/flash.env
sudo systemctl daemon-reload
sudo systemctl enable --now clef@flash
journalctl -u clef@flash -f
```

The first start downloads and prepares the model before the service answers.
The service listens on localhost; change `--host` in the unit for LAN access.
