"""Pinned downloads and atomic, layer-wise NF4 preparation on one GPU.

Quantize one source tensor at a time; never load the dense model on the GPU.
Prepared checkpoints are ordinary Transformers/bitsandbytes checkpoints.
"""
import json
import logging
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
log = logging.getLogger("clef.models")


def profile_config(profile):
    return json.loads((ROOT / "profiles" / f"{profile}.json").read_text())


def checkpoint_files(path, compact):
    required = {"config.json", "joint_head.safetensors", "joint_head_config.json",
                "tokenizer.json", "processor_config.json", "joint_schema_model.py", "LICENSE"}
    if compact:
        required |= {"compact_quantization.json", "input-embedding-nf4.safetensors",
                     "output-embedding-nf4.safetensors"}
    index = path / "model.safetensors.index.json"
    if index.is_file():
        required.add(index.name)
        required.update(json.loads(index.read_text())["weight_map"].values())
    else:
        required.add("model.safetensors")
    missing = sorted(n for n in required if not (path / n).is_file() or (path / n).stat().st_size == 0)
    if missing:
        raise RuntimeError(f"Incomplete checkpoint {path}: missing {missing}")
    expected = ROOT / "vendor/cloudflare/joint_schema_model.py"
    if (path / expected.name).read_bytes() != expected.read_bytes():
        raise RuntimeError("Checkpoint decision wrapper differs from the pinned release")
    return sorted(required)


def validate_checkpoint(path, profile, managed=True):
    config = profile_config(profile)
    files = checkpoint_files(path, config["compact_embeddings"])
    quant = json.loads((path / "config.json").read_text()).get("quantization_config", {})
    if not quant.get("load_in_4bit", quant.get("_load_in_4bit")) or quant.get("bnb_4bit_quant_type") != "nf4":
        raise RuntimeError("Checkpoint must have NF4 quantization_config")
    if quant.get("bnb_4bit_compute_dtype") != config["compute_dtype"]:
        raise RuntimeError("Checkpoint compute dtype differs from the selected profile")
    if managed:
        manifest = json.loads((path / "clef-checkpoint.json").read_text())
        if (manifest.get("format_version"), manifest.get("profile"), manifest.get("revision")) != (
                FORMAT_VERSION, profile, config["revision"]):
            raise RuntimeError("Checkpoint provenance/version differs; use a separate cache directory")
        for name in files:
            if manifest["files"].get(name) != (path / name).stat().st_size:
                raise RuntimeError(f"Checkpoint file changed or is truncated: {name}")
    return files


def download_source(data, config, offline=False):
    from huggingface_hub import snapshot_download
    source = data / "source"
    if offline:
        if not (source / "config.json").is_file():
            raise RuntimeError("Offline mode: pinned source/checkpoint is missing")
    else:
        snapshot_download(config["repository"], revision=config["revision"], local_dir=source,
                          max_workers=2, allow_patterns=["*.json", "*.safetensors", "*.jinja", "LICENSE", "joint_schema_model.py"])
    expected = ROOT / "vendor/cloudflare/joint_schema_model.py"
    if (source / expected.name).read_bytes() != expected.read_bytes():
        raise RuntimeError("Downloaded decision wrapper differs from the saved pinned source")
    return source


def iter_source_tensors(source):
    from safetensors import safe_open
    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
        weight_map = None
    for name in shards:
        with safe_open(source / name, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if weight_map is not None and weight_map.get(key) != name:
                    raise RuntimeError(f"Unexpected source tensor {key} in {name}")
                yield key, shard.get_tensor(key)


class ShardWriter:
    """Bound CPU output staging; support Full checkpoints larger than one shard."""
    def __init__(self, folder, max_bytes=2 * 1024**3):
        self.folder, self.max_bytes = folder, max_bytes
        self.tensors, self.size, self.total, self.shards, self.weight_map = {}, 0, 0, [], {}

    def add(self, key, value):
        if self.tensors and self.size + value.numel() * value.element_size() > self.max_bytes:
            self.flush()
        self.tensors[key] = value.contiguous()
        size = value.numel() * value.element_size()
        self.size += size
        self.total += size

    def flush(self):
        if not self.tensors:
            return
        from safetensors.torch import save_file
        name = f"part-{len(self.shards)+1:05d}.safetensors"
        save_file(self.tensors, str(self.folder / name), metadata={"format": "pt"})
        self.shards.append(name)
        self.weight_map.update({key: name for key in self.tensors})
        self.tensors.clear()
        self.size = 0

    def finish(self):
        self.flush()
        if len(self.shards) == 1:
            (self.folder / self.shards[0]).rename(self.folder / "model.safetensors")
        else:
            renamed = {name: f"model-{i+1:05d}-of-{len(self.shards):05d}.safetensors"
                       for i, name in enumerate(self.shards)}
            for old, new in renamed.items():
                (self.folder / old).rename(self.folder / new)
            (self.folder / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": self.total},
                "weight_map": {k: renamed[v] for k, v in self.weight_map.items()}}, indent=2) + "\n")


def prepare_checkpoint(source, folder, profile):
    """Build the same NF4 format using bounded GPU work, including Flash rows."""
    import bitsandbytes as bnb
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    config = profile_config(profile)
    dtype = torch.float16 if profile == "flash" else torch.bfloat16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["lm_head"])
    backbone_config = AutoConfig.from_pretrained(source, local_files_only=True)
    # Meta construction identifies actual Linear modules (including vision),
    # rather than assuming all two-dimensional weights are linear matrices.
    with torch.device("meta"):
        skeleton = Qwen3_5ForConditionalGeneration(backbone_config)
    linears = {f"{name}.weight" for name, layer in skeleton.named_modules()
               if type(layer) is torch.nn.Linear and name != "lm_head"}
    expected_keys = set(skeleton.state_dict())
    del skeleton
    if len(linears) != config["nf4_linear_layers"]:
        raise RuntimeError(f"Architecture changed: {len(linears)} linear layers")
    embeddings = {"model.language_model.embed_tokens.weight": "input", "lm_head.weight": "output"}
    seen, count = set(), 0
    writer = ShardWriter(folder)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for key, value in iter_source_tensors(source):
        if key in seen or key not in expected_keys:
            raise RuntimeError(f"Unexpected or duplicate source weight {key}")
        seen.add(key)
        if value.is_floating_point():
            value = value.to(dtype=dtype)
        embedding = profile == "flash" and key in embeddings
        if key in linears or embedding:
            # GPU owns at most one dense weight plus its quantized result.
            weight = bnb.nn.Params4bit(value, requires_grad=False, quant_type="nf4",
                compress_statistics=not embedding).to("cuda:0")
            tensors = {"weight": weight.data.cpu().contiguous()}
            tensors.update({k: v.cpu().contiguous() for k, v in weight.quant_state.as_dict(packed=True).items()})
            if embedding:
                save_file(tensors, str(folder / f"{embeddings[key]}-embedding-nf4.safetensors"))
            else:
                for suffix, tensor in tensors.items():
                    writer.add(key if suffix == "weight" else f"{key}.{suffix}", tensor)
                count += 1
            del weight, tensors
        else:
            writer.add(key, value)
    if seen != expected_keys:
        raise RuntimeError(f"Missing source weights: {sorted(expected_keys-seen)}")
    writer.finish()
    for item in source.iterdir():
        if item.is_file() and item.name not in {"model.safetensors.index.json", "config.json"} and (
                item.suffix in {".json", ".jinja"} or item.name in {
                    "joint_head.safetensors", "joint_schema_model.py", "LICENSE"}):
            shutil.copy2(item, folder / item.name)
    backbone_config.quantization_config = quant.to_dict()
    backbone_config.save_pretrained(folder)
    if profile == "flash":
        (folder / "compact_quantization.json").write_text(json.dumps({
            "linear_quantization": "NF4 with double quantization",
            "embedding_quantization": "NF4, 64-weight blocks, FP32 scales",
            "embedding_shape": [248320, 4096], "missing_core_keys": sorted(embeddings),
            "vision": "retained", "decision_head": "FP16 retained"}, indent=2) + "\n")
    files = checkpoint_files(folder, profile == "flash")
    report = {"format_version": FORMAT_VERSION, "profile": profile, "revision": config["revision"],
        "nf4_linear_layers": count, "seconds": round(time.perf_counter()-started, 2),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated()/2**20, 1),
        "preparation": "one tensor at a time on cuda:0", "vision_preserved": True,
        "files": {name: (folder / name).stat().st_size for name in files}}
    (folder / "clef-checkpoint.json").write_text(json.dumps(report, indent=2) + "\n")
    log.info("Prepared checkpoint: %s", report)
    torch.cuda.empty_cache()


def ensure_checkpoint(data, profile, offline=False, external=None):
    """Serialize shared-volume writers and publish only complete checkpoints."""
    if external is not None:
        external = external.expanduser().resolve()
        validate_checkpoint(external, profile, managed=False)
        return external
    import fcntl
    config = profile_config(profile)
    data.mkdir(parents=True, exist_ok=True)
    checkpoint = data / config["checkpoint"]
    with (data / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if checkpoint.exists():
            validate_checkpoint(checkpoint, profile)
            log.info("Using cached %s checkpoint: %s", profile, checkpoint)
            return checkpoint
        source = download_source(data, config, offline=offline)
        staging = data / f".{config['checkpoint']}.preparing"
        if staging.exists():
            shutil.rmtree(staging)  # interrupted unpublished work, never a live checkpoint
        staging.mkdir()
        prepare_checkpoint(source, staging, profile)
        validate_checkpoint(staging, profile)
        staging.rename(checkpoint)
        return checkpoint
