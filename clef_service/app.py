# Project packaging: configurable model paths and a shared web interface.
import base64
import binascii
import gc
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Literal
from contextlib import asynccontextmanager, nullcontext

import bitsandbytes as bnb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_NAME = os.environ.get("CLEF_PROFILE", "flash")
if PROFILE_NAME not in {"flash", "full"}:
    raise RuntimeError("CLEF_PROFILE must be flash or full")
PROFILE = json.loads((PROJECT_ROOT / "profiles" / f"{PROFILE_NAME}.json").read_text())
MODEL_ID = PROFILE["model_id"]
COMPUTE_DTYPE = torch.float16 if PROFILE["compute_dtype"] == "float16" else torch.bfloat16
STARTUP_STRATEGY = json.loads(os.environ.get("CLEF_STARTUP_STRATEGY", "{}"))
ROOT = Path(os.environ.get("CLEF_DATA_DIR", str(PROJECT_ROOT / "models" / PROFILE_NAME))).expanduser().resolve()
UI_DIR = PROJECT_ROOT / "web"
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "cloudflare"))
MODEL_DIR = ROOT / os.environ.get("CLEF_MODEL_DIR", PROFILE["checkpoint"])
SOURCE_REVISION = PROFILE["revision"]
MAX_LENGTH = int(os.environ.get("CLEF_MAX_LENGTH", str(PROFILE["max_input_tokens"])))
IMAGE_LENGTH_SETTING = os.environ.get("CLEF_MAX_IMAGE_LENGTH", "auto" if PROFILE_NAME == "full" else str(MAX_LENGTH)).strip().lower()
MAX_IMAGE_LENGTH = None if IMAGE_LENGTH_SETTING == "auto" else int(IMAGE_LENGTH_SETTING)
PREFILL_INITIAL_CHUNK_TOKENS = int(os.environ.get("CLEF_PREFILL_INITIAL_CHUNK_TOKENS", "8192"))
PREFILL_CHUNK_TOKENS = int(os.environ.get("CLEF_PREFILL_CHUNK_TOKENS", "4096"))
if (MAX_IMAGE_LENGTH is not None and MAX_IMAGE_LENGTH < 1) or PREFILL_INITIAL_CHUNK_TOKENS < 1 or PREFILL_CHUNK_TOKENS < 0:
    raise RuntimeError("Invalid input limit or prefill chunk size")
MAX_IMAGE_PIXELS = int(os.environ.get("CLEF_MAX_IMAGE_PIXELS", "1048576"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ATTENTION_BACKEND = os.environ.get("CLEF_ATTENTION_BACKEND", "efficient")
sys.path.insert(0, str(MODEL_DIR if MODEL_DIR.exists() else ROOT / "source"))
from joint_schema_model import load_release_model
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))
from context_budget import ContextBudget
from image_request import IMAGE_FIDELITIES, media_options
from reusable_inputs import InputCache
from request_queue import DecisionQueue, QueueAdmissionMiddleware, QueueError
from optimized_inference import InferenceEngine, encode_compact, pool_record, remap_checkpoints, answers
engine = InferenceEngine(os.environ.get("CLEF_PREFIX_CACHE_MIB", "auto"),
    int(os.environ.get("CLEF_PREFIX_CACHE_ENTRIES", "32")),
    reserve_mib=int(os.environ.get("CLEF_PREFIX_CACHE_RESERVE_MIB", "256")),
    utilization=float(os.environ.get("CLEF_PREFIX_CACHE_GPU_UTILIZATION", "1.0")),
    checkpoint_tokens=int(os.environ.get("CLEF_PREFIX_CHECKPOINT_TOKENS", "0")),
    feature_cache_mib=int(os.environ.get("CLEF_IMAGE_FEATURE_CACHE_MIB", "256")),
    feature_cache_entries=int(os.environ.get("CLEF_IMAGE_FEATURE_CACHE_ENTRIES", "64")),
    elastic=os.environ.get("CLEF_PREFIX_CACHE_ELASTIC", "1") == "1",
    prefill_reserve_mib=int(os.environ.get("CLEF_PREFIX_CACHE_PREFILL_RESERVE_MIB", "512")))
IMAGE_PREFILL_ENABLED = os.environ.get("CLEF_IMAGE_PREFILL", "0") == "1"
PREFIX_CACHE_ENABLED = os.environ.get("CLEF_PREFIX_CACHE", "1") == "1"
INPUT_CACHE_ENABLED = os.environ.get("CLEF_INPUT_CACHE", "1") == "1"
POOLING_DEFAULT = os.environ.get("CLEF_IMAGE_POOLING", "0") == "1"

log = logging.getLogger("clef")
lock = threading.Lock()
model = processor = context_budget = None
request_queue = DecisionQueue.from_env(lambda request: run_decision(request))


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    instructions: str | None = None
    criteria: dict[str, str] | list[str] | None = None


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    state: str | dict | list | int | float | bool | None
    context: str | dict | list | int | float | bool | None = Field(default=None,
        description="Optional reusable context placed before images and changing state.")
    images: list[str] = Field(default_factory=list, max_length=16,
        description="PNG/JPEG/WebP images as base64 data URLs; up to sixteen images.")
    media_kwargs: dict | None = Field(default=None,
        description="Upstream Clef processor options, e.g. images_kwargs. Omit for native defaults; use do_resize:false with patch-aligned client images.")
    image_fidelity: Literal["low", "standard", "medium", "high"] | None = Field(default=None, deprecated=True,
        description="Legacy opt-in server resizing. Prefer client resizing and media_kwargs; cannot be combined with media_kwargs.")
    image_pooling: bool = Field(default=POOLING_DEFAULT,
        description="Experimental 2x2 spatial pooling after vision encoding; may lose fine detail.")
    prefix_cache: bool = Field(default=True, description="Reuse the longest exact cached context/image/state checkpoint.")
    input_cache: bool = Field(default=True, description="Reuse exact preprocessing, text tokens and independent image features; answers are always recomputed.")
    questions: dict[str, Question] = Field(min_length=1, max_length=64)


def make_adaptive_chunk_kernel(fla_kernel, max_tokens):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    def adaptive_chunk_gated_delta_rule(*args, **kwargs):
        query = args[0] if args else kwargs["q"]
        kernel = fla_kernel if query.shape[1] <= max_tokens else torch_chunk_gated_delta_rule
        return kernel(*args, **kwargs)

    return adaptive_chunk_gated_delta_rule


def load_model():
    if ATTENTION_BACKEND == "efficient":
        # PyTorch's efficient SDPA kernel cannot consume unequal Q/KV head counts.
        # Use Transformers' existing repeat_kv path instead of enable_gqa=True.
        # This override is confined to this service process; no package files change.
        from transformers.integrations import sdpa_attention
        sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        llm_int8_skip_modules=["lm_head"],
    )
    kwargs = {"attn_implementation": "sdpa"}
    if not MODEL_DIR.exists():
        kwargs["quantization_config"] = quant
    compact_checkpoint = (MODEL_DIR / "compact_quantization.json").is_file()
    loaded, proc = load_release_model(
        MODEL_DIR if MODEL_DIR.exists() else ROOT / "source",
        device="cpu" if compact_checkpoint else "cuda:0", dtype=COMPUTE_DTYPE, **kwargs,
    )
    quantized = [m for m in loaded.language_model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    count = len(quantized)
    if count != PROFILE["nf4_linear_layers"]:
        raise RuntimeError(f"Expected {PROFILE['nf4_linear_layers']} NF4 layers for {PROFILE_NAME}, got {count}")
    if any(m.weight.quant_state.quant_type != "nf4" for m in quantized):
        raise RuntimeError("Unexpected non-NF4 quantization")
    if isinstance(loaded.language_model.get_output_embeddings(), bnb.nn.Linear4bit):
        raise RuntimeError("Clef output embeddings require row lookup; Linear4bit is unsupported")
    if compact_checkpoint:
        # Restore the saved NF4 embeddings on CPU before moving the complete
        # model to CUDA. Temporary dense missing embeddings otherwise exceed
        # the 3070 Ti's 8GB during initialization.
        from clef_service.compact_quant import load_embeddings
        from accelerate.hooks import remove_hook_from_module
        load_embeddings(loaded, MODEL_DIR)
        remove_hook_from_module(loaded.language_model, recurse=True)
        loaded.to(device="cuda:0")
        loaded.language_model.hf_device_map = {"": "cuda:0"}
        if any(p.device != torch.device("cuda:0") for p in loaded.parameters()):
            raise RuntimeError("Clef parameters did not all move to the selected GPU")
        gc.collect()
        torch.cuda.empty_cache()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if os.environ.get("CLEF_DISABLE_FUSED_KERNELS") == "1":
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
        for layer in loaded.modules():
            if isinstance(layer, qwen.Qwen3_5GatedDeltaNet):
                layer.causal_conv1d_fn = None
                layer.causal_conv1d_update = qwen.torch_causal_conv1d_update
                layer.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                layer.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
                # Native norm has the same trained weights and avoids the FLA kernel.
                norm = qwen.Qwen3_5RMSNormGated(layer.head_v_dim, eps=layer.layer_norm_epsilon)
                norm.load_state_dict(layer.norm.state_dict())
                layer.norm = norm.to(device="cuda:0", dtype=COMPUTE_DTYPE)
    if any(p.device != torch.device("cuda:0") for p in loaded.parameters()):
        raise RuntimeError("All Clef parameters must reside on the configured single GPU")
    gated = [m for m in loaded.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]
    kernel_names = sorted({m.chunk_gated_delta_rule.__module__ + "." + m.chunk_gated_delta_rule.__name__ for m in gated})
    fast = bool(qwen.is_fast_path_available)
    if os.environ.get("CLEF_REQUIRE_FAST_LINEAR_ATTENTION") == "1":
        if not fast or not gated or any(not name.startswith("fla.") for name in kernel_names):
            raise RuntimeError("Optimized linear-attention kernels are required but unavailable")
    prefill_backend = os.environ.get("CLEF_LINEAR_PREFILL_BACKEND", "auto")
    if prefill_backend not in {"auto", "torch", "fla", "adaptive"}:
        raise RuntimeError("CLEF_LINEAR_PREFILL_BACKEND must be auto, torch, fla or adaptive")
    if prefill_backend in {"fla", "adaptive"} and (not fast or not gated or any(not name.startswith("fla.") for name in kernel_names)):
        raise RuntimeError("FLA prefill kernels are requested but unavailable")
    max_fla_tokens = None
    if prefill_backend == "torch":
        for layer in gated:
            layer.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
    elif prefill_backend == "adaptive":
        max_fla_tokens = int(os.environ.get("CLEF_FLA_MAX_CHUNK_TOKENS", "512"))
        if max_fla_tokens <= 0:
            raise RuntimeError("CLEF_FLA_MAX_CHUNK_TOKENS must be positive")
        for layer in gated:
            layer.chunk_gated_delta_rule = make_adaptive_chunk_kernel(layer.chunk_gated_delta_rule, max_fla_tokens)
    kernel_names = sorted({m.chunk_gated_delta_rule.__module__ + "." + m.chunk_gated_delta_rule.__name__ for m in gated})
    recurrent_names = sorted({m.recurrent_gated_delta_rule.__module__ + "." + m.recurrent_gated_delta_rule.__name__ for m in gated})
    app.state.linear_attention = {"fast_path_available": fast, "layers": len(gated),
        "prefill_override": prefill_backend, "fla_max_chunk_tokens": max_fla_tokens, "chunk_kernels": kernel_names,
        "optimized_prefill": bool(gated) and all(name.startswith("fla.") for name in kernel_names),
        "recurrent_kernels": recurrent_names,
        "optimized_recurrence": bool(gated) and all(name.startswith("fla.") for name in recurrent_names)}
    log.info("Linear attention: %s", app.state.linear_attention)
    log.info("Loaded %s NF4 linear layers on %s", count, torch.cuda.get_device_name())
    return loaded, proc, count


@asynccontextmanager
async def lifespan(app):
    global model, processor, context_budget
    model, processor, count = load_model()
    app.state.nf4_layers = count
    engine.initialize_memory_budget()
    config = model.language_model.config.text_config
    calibrated = (ATTENTION_BACKEND == "efficient" and PREFILL_INITIAL_CHUNK_TOKENS == 8192
                  and PREFILL_CHUNK_TOKENS == 4096
                  and config.num_key_value_heads == 4 and config.head_dim == 256
                  and ((PROFILE_NAME == "flash" and config.hidden_size == 4096
                        and config.num_hidden_layers == 32
                        and (MODEL_DIR / "compact_quantization.json").is_file())
                       or (PROFILE_NAME == "full" and config.hidden_size == 5120
                           and config.num_hidden_layers == 64
                           and app.state.linear_attention["fast_path_available"]))
                  and os.environ.get("CLEF_DISABLE_FUSED_KERNELS") != "1")
    context_budget = ContextBudget(config, dtype_bytes=2, calibrated=calibrated,
        configured_limit=MAX_LENGTH, image_limit=MAX_IMAGE_LENGTH,
        reserve_mib=int(os.environ.get("CLEF_CONTEXT_RESERVE_MIB", "512")),
        mode=os.environ.get("CLEF_CONTEXT_LIMIT_MODE", "auto"),
        image_chunked=IMAGE_PREFILL_ENABLED and PREFILL_CHUNK_TOKENS > 0)
    refresh_context_budget()
    app.state.cache_stats = {"prefix_cache": engine.stats(), "input_cache": input_cache.stats()}
    await request_queue.start()
    try:
        yield
    finally:
        await request_queue.close()


app = FastAPI(title=f"Clef {PROFILE_NAME.title()} NF4", lifespan=lifespan)
app.add_middleware(QueueAdmissionMiddleware, budget=request_queue.budget)

if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

    @app.get("/", include_in_schema=False)
    def playground():
        return FileResponse(UI_DIR / "index.html")


def refresh_context_budget():
    return context_budget.snapshot(free_bytes=torch.cuda.mem_get_info()[0],
        reserved_bytes=torch.cuda.memory_reserved(), allocated_bytes=torch.cuda.memory_allocated(),
        cache_bytes=engine._bytes())


def context_metadata():
    # Never sample transient request tensors as an idle capacity estimate.
    if lock.acquire(blocking=False):
        try:
            return {**refresh_context_budget(), "context_snapshot_status": "idle"}
        finally:
            lock.release()
    return {**context_budget.last, "context_snapshot_status": "last_idle_busy"}


def runtime_cache_stats():
    # Cache dictionaries mutate on the sole inference worker. Health uses the
    # last idle sample while it is busy rather than iterating mutable entries.
    if lock.acquire(blocking=False):
        try:
            app.state.cache_stats = {"prefix_cache": engine.stats(), "input_cache": input_cache.stats()}
        finally:
            lock.release()
    return app.state.cache_stats


@app.get("/health")
def health():
    return {
        "queue": request_queue.stats(),
        "status": "ok", "model": PROFILE["repository"], "startup_strategy": STARTUP_STRATEGY, "quantization": "nf4",
        "compact_embeddings": (MODEL_DIR / "compact_quantization.json").is_file(),
        "source_revision": SOURCE_REVISION, "gpu": torch.cuda.get_device_name(),
        "compute_dtype": PROFILE["compute_dtype"], "nf4_layers": app.state.nf4_layers,
        "linear_attention": app.state.linear_attention,
        **context_metadata(),
        "input": ["text", "json", "images"],
        "image_input": {"format": "base64 data URL", "max_images": 16,
                        "max_processed_pixels_per_image": processor.image_processor.size["longest_edge"],
                        "resize_location": "client_for_gui", "processor_defaults_for_api": True,
                        "legacy_fidelity_max_pixels": MAX_IMAGE_PIXELS,
                        "default_fidelity": "standard", "fidelities": IMAGE_FIDELITIES,
                        "pooling_default": POOLING_DEFAULT, "pooling_factor": 2},
        "attention_backend": ATTENTION_BACKEND,
        "optimizations": {"compact_schema": True, "single_pass_preprocessing": True,
                          "chunked_prefill": {"enabled": PREFILL_CHUNK_TOKENS > 0, "text_only": not IMAGE_PREFILL_ENABLED, "images_enabled": IMAGE_PREFILL_ENABLED, "vision_batch_images": 1,
                                              "initial_chunk_tokens": PREFILL_INITIAL_CHUNK_TOKENS,
                                              "chunk_tokens": PREFILL_CHUNK_TOKENS},
                          "prefix_cache_enabled": PREFIX_CACHE_ENABLED and ATTENTION_BACKEND == "efficient",
                          "input_cache_enabled": INPUT_CACHE_ENABLED, **runtime_cache_stats()},
        "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "free_device_mib": round(torch.cuda.mem_get_info()[0] / 2**20, 1),
    }


def decode_image(value: str) -> Image.Image:
    try:
        header, payload = value.split(",", 1)
        if header not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
            raise ValueError("Use a PNG, JPEG, or WebP base64 data URL")
        if len(payload) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise HTTPException(413, "Image exceeds 10 MiB")
        raw = base64.b64decode(payload, validate=True)
        with Image.open(BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Unsupported image format")
            if source.width * source.height > 20_000_000:
                raise HTTPException(413, "Image exceeds 20 million source pixels")
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError,
            Image.DecompressionBombError) as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc


input_cache = InputCache(
    max_mib=int(os.environ.get("CLEF_PREPROCESS_CACHE_MIB", "256")),
    max_entries=int(os.environ.get("CLEF_PREPROCESS_CACHE_ENTRIES", "128")),
    token_mib=int(os.environ.get("CLEF_TOKEN_CACHE_MIB", "8")),
    token_entries=int(os.environ.get("CLEF_TOKEN_CACHE_ENTRIES", "1024")), decode=decode_image)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                       **context_metadata()}]}


@app.post("/v1/systemone")
async def decide(request: DecisionRequest, http_request: Request):
    try:
        return await request_queue.run(request, http_request)
    except QueueError as exc:
        return exc.response()


def run_decision(request: DecisionRequest):
    started = time.perf_counter()
    if request.model not in {MODEL_ID, "clef-27b-nf4" if PROFILE_NAME == "full" else "clef-flash-nf4", PROFILE["repository"]}:
        raise HTTPException(400, f"Select {MODEL_ID}")
    record = request.model_dump(exclude_none=True, exclude={"image_fidelity", "image_pooling", "prefix_cache", "input_cache"})
    record["state"] = request.state
    if request.images:
        try:
            options = media_options(request.media_kwargs, request.image_fidelity, MAX_IMAGE_PIXELS)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if options is not None:
            record["media_kwargs"] = options
    for question in record["questions"].values():
        kind = question["type"]
        criteria = question.get("criteria")
        if kind not in {"noul", "choice", "score"}:
            raise HTTPException(400, "Question type must be noul, choice, or score")
        if kind == "choice" and (not isinstance(criteria, dict) or not criteria):
            raise HTTPException(400, "choice requires a nonempty criteria object")
        if kind == "score" and (not isinstance(criteria, list) or not criteria):
            raise HTTPException(400, "score requires a nonempty criteria list")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise HTTPException(400, "noul criteria must be an object")
    # Only the queue worker executes inference. Metadata may briefly own the
    # same lock; wait for it instead of rejecting an already queued request.
    lock.acquire()
    try:
        preprocessing_started = time.perf_counter()
        before = input_cache.stats()
        reuse_inputs = request.input_cache and INPUT_CACHE_ENABLED
        encoded, boundary, checkpoints = encode_compact(processor, record, with_checkpoints=True,
            input_cache=input_cache, cache_enabled=reuse_inputs)
        after = input_cache.stats()
        preprocess_usage = {"preprocessing_ms": round((time.perf_counter()-preprocessing_started)*1000, 1),
            "image_preprocess_cache_hits": after["image_hits"]-before["image_hits"],
            "image_preprocess_cache_misses": after["image_misses"]-before["image_misses"],
            "token_cache_hits": after["token_hits"]-before["token_hits"],
            "token_cache_misses": after["token_misses"]-before["token_misses"]}
        original_tokens = len(encoded.input_ids)
        if request.image_pooling and request.images:
            original = encoded
            encoded, boundary = pool_record(encoded, boundary, model.language_model.config.image_token_id)
            checkpoints = remap_checkpoints(checkpoints, original, encoded)
        limits = refresh_context_budget()
        request_limit = limits["max_input_tokens_with_images" if request.images else "max_input_tokens"]
        if len(encoded.input_ids) > request_limit:
            return JSONResponse(status_code=413, content={
                "detail": f"Request exceeds {request_limit} input tokens",
                "input_tokens": len(encoded.input_ids), "max_input_tokens": request_limit,
                "context_limit_mode": limits["context_limit"]["mode"]})
        cache_preparation = engine.prepare_request(context_budget.workspace(len(encoded.input_ids))
            if context_budget.computed and len(encoded.input_ids)>PREFILL_INITIAL_CHUNK_TOKENS else None)
        torch.cuda.reset_peak_memory_stats()
        input_key = hashlib.sha256(json.dumps([request.images, record.get("media_kwargs")], separators=(",", ":")).encode()).hexdigest()
        caching = request.prefix_cache and PREFIX_CACHE_ENABLED and ATTENTION_BACKEND == "efficient"
        def infer(cache, features=True):
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if ATTENTION_BACKEND == "efficient" else nullcontext():
                return engine.forward(model, processor, encoded, boundary, input_key,
                    pooling=request.image_pooling, cache_enabled=cache, checkpoint_boundaries=checkpoints,
                    feature_cache_enabled=features and reuse_inputs and ATTENTION_BACKEND == "efficient",
                    prefill_chunk_tokens=PREFILL_CHUNK_TOKENS,
                    prefill_initial_chunk_tokens=PREFILL_INITIAL_CHUNK_TOKENS, prefill_images=IMAGE_PREFILL_ENABLED)
        try:
            logits, cache_usage = infer(caching)
        except torch.cuda.OutOfMemoryError as exc:
            # Cached states and their branch clone cost extra VRAM. Retry once
            # without caching before rejecting a request that otherwise fits.
            exc.__traceback__ = None
            engine.clear()
            gc.collect()
            torch.cuda.empty_cache()
            if not caching and not reuse_inputs:
                raise
            logits, cache_usage = infer(False, False)
            cache_usage["prefix_cache"] = "memory_fallback"
        result = {"model": request.model, "answers": answers(record, encoded, logits),
                  "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0,
                            "unpooled_input_tokens": original_tokens, "compact_schema": True,
                            "image_pooling": bool(request.image_pooling and request.images),
                            **preprocess_usage, **cache_preparation, **cache_usage}}
        if cache_usage.get("prefill_mode") == "chunked" and engine._bytes() == 0 and (not request.images or context_budget.image_computed):
            context_budget.observe(len(encoded.input_ids),
                torch.cuda.max_memory_allocated() - engine.base_bytes)
        result["usage"].update(accepted_max_input_tokens=request_limit,
                               context_limit_mode=limits["context_limit"]["mode"])
        json.dumps(result, allow_nan=False)
        torch.cuda.synchronize()
        result["usage"].update(latency_ms=round((time.perf_counter() - started) * 1000, 1),
            peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1))
        if request.images:
            grid = encoded.media.get("original_grid_thw", encoded.media["image_grid_thw"])
            patch = processor.image_processor.patch_size
            sizes = [{"width": int(w)*patch, "height": int(h)*patch} for _, h, w in grid.tolist()]
            result["usage"].update(image_count=len(request.images), image_fidelity=request.image_fidelity,
                processed_images=sizes, max_processed_pixels_per_image=max(v["width"]*v["height"] for v in sizes),
                processor_resize_enabled=record.get("media_kwargs", {}).get("images_kwargs", {}).get("do_resize", True))
        return result
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        if "encoded" in locals() and (not request.images or context_budget.image_computed):
            context_budget.record_oom(len(encoded.input_ids))
        exc.__traceback__ = None
        engine.clear()
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(503, "Insufficient GPU memory; shorten the request") from None
    finally:
        try:
            engine.finish_request()
        finally:
            lock.release()
