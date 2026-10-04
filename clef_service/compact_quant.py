"""NF4 input/lexical embeddings for Clef's decision-only inference path."""
import gc
import json
from pathlib import Path

import bitsandbytes as bnb
from safetensors import safe_open
from safetensors.torch import save_file
import torch


class LexicalRows:
    def __init__(self, embedding):
        self.embedding = embedding

    def __getitem__(self, token_ids):
        return self.embedding(token_ids)


def compact_forward(self, batch):
    base = self.language_model
    media = batch.get("media") or {}
    text_model = base.model
    if not media and hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    output = text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False, return_dict=True, **media)
    return self.head(output.last_hidden_state, batch["input_ids"], batch["attention_mask"],
                     batch["records"], LexicalRows(base.get_output_embeddings()))


def install_compact_embeddings(model, checkpoint=None):
    """Only dequantize requested rows, including the head's lexical option rows."""
    from types import MethodType
    backbone = model.language_model
    for name, previous, setter in [
        ("input", backbone.get_input_embeddings(), backbone.set_input_embeddings),
        ("output", backbone.get_output_embeddings(), backbone.set_output_embeddings),
    ]:
        rows, width = previous.weight.shape
        embedding = bnb.nn.EmbeddingNF4(rows, width, dtype=torch.float16, device="meta")
        embedding.dtype = torch.float16
        if checkpoint:
            file = Path(checkpoint) / f"{name}-embedding-nf4.safetensors"
            with safe_open(file, framework="pt", device="cpu") as source:
                data = source.get_tensor("weight")
                stats = {k: source.get_tensor(k) for k in source.keys() if k != "weight"}
            embedding.weight = bnb.nn.Params4bit.from_prequantized(
                data=data, quantized_stats=stats, requires_grad=False,
                device=previous.weight.device, module=embedding)
        else:
            embedding.weight = bnb.nn.Params4bit(
                previous.weight.detach(), requires_grad=False, compress_statistics=False,
                quant_type="nf4", module=embedding)
            embedding = embedding.to(previous.weight.device)
        setter(embedding)
        if not checkpoint:
            yield name, embedding
    model.forward = MethodType(compact_forward, model)
    gc.collect()
    torch.cuda.empty_cache()


def prepare_embeddings(model, folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    for name, embedding in install_compact_embeddings(model):
        tensors = {"weight": embedding.weight.data.cpu().contiguous()}
        tensors.update({k: v.cpu().contiguous() for k, v in
                        embedding.weight.quant_state.as_dict(packed=True).items()})
        save_file(tensors, str(folder / f"{name}-embedding-nf4.safetensors"))


def load_embeddings(model, folder):
    # The installer is a generator; consume it to apply the replacements.
    list(install_compact_embeddings(model, checkpoint=folder))


def save_compact_core(original, folder):
    """Retain the existing linear NF4 tensors without re-quantizing them."""
    import shutil
    original, folder = Path(original), Path(folder)
    excluded = {"lm_head.weight", "model.language_model.embed_tokens.weight"}
    with safe_open(original / "model.safetensors", framework="pt", device="cpu") as source:
        tensors = {k: source.get_tensor(k) for k in source.keys() if k not in excluded}
        save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    del tensors
    gc.collect()
    for source in original.iterdir():
        if source.is_file() and source.name != "model.safetensors":
            shutil.copy2(source, folder / source.name)
    (folder / "compact_quantization.json").write_text(json.dumps({
        "linear_quantization": "NF4 with double quantization (unchanged)",
        "embedding_quantization": "NF4, 64-weight blocks, FP32 scales",
        "embedding_shape": [248320, 4096],
        "missing_core_keys": sorted(excluded),
        "vision": "retained", "decision_head": "FP16 retained",
    }, indent=2) + "\n")
