# Project packaging: configurable model paths and a shared web interface.
import base64
import binascii
import gc
import hashlib
from io import BytesIO
import json
import logging
import math
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
from fastapi import Security
from fastapi.security import HTTPBearer
from .access import APIKeyMiddleware, configured_api_key
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
ROCM_LONG_PROMPT_THRESHOLD = int(os.environ.get("CLEF_ROCM_LONG_PROMPT_THRESHOLD", "0"))
ROCM_LONG_CHUNK_TOKENS = int(os.environ.get("CLEF_ROCM_LONG_PREFILL_CHUNK_TOKENS", "512"))
if ROCM_LONG_PROMPT_THRESHOLD < 0 or ROCM_LONG_CHUNK_TOKENS < 1:
    raise RuntimeError("Invalid long-prompt chunk policy")
MAX_IMAGE_PIXELS = int(os.environ.get("CLEF_MAX_IMAGE_PIXELS", "1048576"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VISION_PATCH_TOKENS = int(os.environ.get("CLEF_MAX_VISION_PATCH_TOKENS_PER_IMAGE", "0"))
if MAX_VISION_PATCH_TOKENS < 0:
    raise RuntimeError("Invalid per-image vision patch limit")
ATTENTION_BACKEND = os.environ.get("CLEF_ATTENTION_BACKEND", "efficient")
# Math SDPA cache reuse is opt-in on the independently validated HIP path.
# Keep the historical CUDA policy unchanged.
GPU_CACHE_SUPPORTED = ATTENTION_BACKEND == "efficient" or (
    ATTENTION_BACKEND == "math" and STARTUP_STRATEGY.get("runtime_backend") == "rocm"
    and os.environ.get("CLEF_ROCM_GPU_CACHE", "0") == "1")
sys.path.insert(0, str(MODEL_DIR if MODEL_DIR.exists() else ROOT / "source"))
from joint_schema_model import load_release_model
sys.path.insert(0, str(PROJECT_ROOT / "runtime"))
from context_budget import ContextBudget
from image_request import IMAGE_FIDELITIES, media_options
from reusable_inputs import InputCache, cpu_bytes
from ram_cache_budget import RAMCacheBudget
from cpu_preparation import PreparedDecision, preparation_fits
from gpu_timer import GPUTimer
from media_prefix import MediaPrefix
from request_queue import DecisionQueue, QueueAdmissionMiddleware, QueueError
from text_batching import BatchUnavailable, TextBatchAdmission, shared_tokens
from optimized_inference import available_cuda_memory, InferenceEngine, encode_compact, pool_record, remap_checkpoints, answers
ram_budget = RAMCacheBudget(float(os.environ.get("CLEF_PREFIX_HOST_CACHE_RESERVE_FRACTION", "0.25")))
engine = InferenceEngine(os.environ.get("CLEF_PREFIX_CACHE_MIB", "auto"),
    int(os.environ.get("CLEF_PREFIX_CACHE_ENTRIES", "32")),
    reserve_mib=int(os.environ.get("CLEF_PREFIX_CACHE_RESERVE_MIB", "256")),
    utilization=float(os.environ.get("CLEF_PREFIX_CACHE_GPU_UTILIZATION", "1.0")),
    checkpoint_tokens=int(os.environ.get("CLEF_PREFIX_CHECKPOINT_TOKENS", "0")),
    feature_cache_mib=int(os.environ.get("CLEF_IMAGE_FEATURE_CACHE_MIB", "256")),
    feature_cache_entries=int(os.environ.get("CLEF_IMAGE_FEATURE_CACHE_ENTRIES", "64")),
    elastic=os.environ.get("CLEF_PREFIX_CACHE_ELASTIC", "1") == "1",
    prefill_reserve_mib=int(os.environ.get("CLEF_PREFIX_CACHE_PREFILL_RESERVE_MIB", "512")),
    promote_repeated_boundary=os.environ.get("CLEF_CACHE_PROMOTE_REPEATED_BOUNDARY", "0") == "1",
    host_cache_mib=os.environ.get("CLEF_PREFIX_HOST_CACHE_MIB", "auto"),
    host_cache_entries=None if os.environ.get("CLEF_PREFIX_HOST_CACHE_ENTRIES", "auto").lower() == "auto"
        else int(os.environ["CLEF_PREFIX_HOST_CACHE_ENTRIES"]),
    host_cache_reserve_fraction=ram_budget.reserve_fraction, ram_budget=ram_budget,
    prefix_index=os.environ.get("CLEF_PREFIX_INDEX", "radix"),
    async_restore=os.environ.get("CLEF_PREFIX_ASYNC_RESTORE", "0") == "1",
    restore_staging_mib=int(os.environ.get("CLEF_PREFIX_RESTORE_STAGING_MIB", "128")),
    shared_host_blocks=os.environ.get("CLEF_PREFIX_SHARED_HOST_BLOCKS", "0") == "1",
    active_context_offload=os.environ.get("CLEF_ACTIVE_CONTEXT_OFFLOAD", "none"))
IMAGE_PREFILL_ENABLED = os.environ.get("CLEF_IMAGE_PREFILL", "0") == "1"
PREFIX_CACHE_ENABLED = os.environ.get("CLEF_PREFIX_CACHE", "1") == "1"
INPUT_CACHE_ENABLED = os.environ.get("CLEF_INPUT_CACHE", "1") == "1"
POOLING_DEFAULT = os.environ.get("CLEF_IMAGE_POOLING", "0") == "1"

log = logging.getLogger("clef")
lock = threading.RLock()
gpu_profiles = threading.local()
model = processor = context_budget = None
request_queue = DecisionQueue.from_env(lambda request: run_decision(request),
    batch_handler=lambda requests, emit: run_decision_batch(requests, emit),
    batch_key=lambda request: text_batch_key(request),
    prepare_handler=lambda request: prepare_queued_decision(request))


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
    from native_linear_attention import native_chunk_kernel
    native_block_tokens = int(os.environ.get("CLEF_TORCH_GDN_BLOCK_TOKENS", "64"))
    native_prefill = native_chunk_kernel(qwen.torch_chunk_gated_delta_rule, native_block_tokens)
    if native_block_tokens != 64 and prefill_backend != "torch":
        raise RuntimeError("Native GDN block override requires CLEF_LINEAR_PREFILL_BACKEND=torch")
    if prefill_backend not in {"auto", "torch", "fla", "adaptive"}:
        raise RuntimeError("CLEF_LINEAR_PREFILL_BACKEND must be auto, torch, fla or adaptive")
    if prefill_backend in {"fla", "adaptive"} and (not fast or not gated or any(not name.startswith("fla.") for name in kernel_names)):
        raise RuntimeError("FLA prefill kernels are requested but unavailable")
    max_fla_tokens = None
    if prefill_backend == "torch":
        for layer in gated:
            layer.chunk_gated_delta_rule = native_prefill
    elif prefill_backend == "adaptive":
        max_fla_tokens = int(os.environ.get("CLEF_FLA_MAX_CHUNK_TOKENS", "512"))
        if max_fla_tokens <= 0:
            raise RuntimeError("CLEF_FLA_MAX_CHUNK_TOKENS must be positive")
        for layer in gated:
            layer.chunk_gated_delta_rule = make_adaptive_chunk_kernel(layer.chunk_gated_delta_rule, max_fla_tokens)
    kernel_names = sorted({m.chunk_gated_delta_rule.__module__ + "." + m.chunk_gated_delta_rule.__name__ for m in gated})
    recurrent_names = sorted({m.recurrent_gated_delta_rule.__module__ + "." + m.recurrent_gated_delta_rule.__name__ for m in gated})
    app.state.linear_attention = {"fast_path_available": fast, "layers": len(gated),
        "native_gdn_block_tokens": native_block_tokens if prefill_backend == "torch" else 64,
        "prefill_override": prefill_backend, "fla_max_chunk_tokens": max_fla_tokens, "chunk_kernels": kernel_names,
        "optimized_prefill": bool(gated) and all(name.startswith("fla.") for name in kernel_names),
        "recurrent_kernels": recurrent_names,
        "optimized_recurrence": bool(gated) and all(name.startswith("fla.") for name in recurrent_names)}
    log.info("Linear attention: %s", app.state.linear_attention)
    app.state.vision_attention = {"query_tiling": False}
    if os.environ.get("CLEF_ROCM_VISION_TILING", "0") == "1":
        if STARTUP_STRATEGY.get("runtime_backend") != "rocm" or ATTENTION_BACKEND != "math":
            raise RuntimeError("Experimental vision tiling requires ROCm math attention")
        from vision_math_attention import install_vision_math_attention
        chunk = int(os.environ.get("CLEF_VISION_QUERY_CHUNK_TOKENS", "256"))
        vision_layers = install_vision_math_attention(loaded, query_chunk=chunk)
        app.state.vision_attention = {"query_tiling": True, "query_chunk_tokens": chunk,
                                     "dense_threshold_tokens": 1024, "layers": vision_layers}
    log.info("Loaded %s NF4 linear layers on %s", count, torch.cuda.get_device_name())
    return loaded, proc, count


@asynccontextmanager
async def lifespan(app):
    global model, processor, context_budget
    app.state.startup_ready = False
    warmup_setting = os.environ.get("CLEF_WARMUP", "1")
    if warmup_setting not in {"0", "1"}:
        raise ValueError("CLEF_WARMUP must be 0 or 1")
    app.state.startup_warmup = {"enabled": warmup_setting == "1", "status": "pending"}
    app.state.matrix_tuning = {"enabled": False}
    tuned_file = os.environ.get("CLEF_ROCM_TUNABLEOP_FILE", "")
    if tuned_file:
        if STARTUP_STRATEGY.get("runtime_backend") != "rocm":
            raise RuntimeError("ROCm TunableOp results require the ROCm runtime")
        if not Path(tuned_file).is_file():
            raise RuntimeError(f"Missing offline TunableOp results: {tuned_file}")
        os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"
        tunable = torch.cuda.tunable
        tunable.set_filename(tuned_file)
        if not tunable.read_file(tuned_file):
            raise RuntimeError("Offline TunableOp results failed runtime validation")
        tunable.tuning_enable(False)
        tunable.record_untuned_enable(False)
        tunable.enable(True)
        app.state.matrix_tuning = {"enabled": True, "live_tuning": False,
                                   "backend": "rocblas", "results": len(tunable.get_results())}
    model, processor, count = load_model()
    app.state.nf4_layers = count
    engine.initialize_memory_budget()
    config = model.language_model.config.text_config
    calibrated = (engine.active_context_offload in {'none','auto'} and ATTENTION_BACKEND == "efficient" and PREFILL_INITIAL_CHUNK_TOKENS == 8192
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
    if engine.active_context_offload=='auto':
        if torch.version.hip or not PREFILL_CHUNK_TOKENS:
            raise RuntimeError('Adaptive context requires CUDA and incremental prefill')
        from adaptive_context import AdaptiveContext
        from streamed_kv import efficient_block
        available=True
        try:
            sample=torch.ones((1,1,8,config.head_dim),device='cuda',dtype=COMPUTE_DTYPE)
            probe,lse=efficient_block(sample,sample,sample,config.head_dim**-.5)
            if not bool(torch.isfinite(lse[...,:sample.shape[-2]]).all()) or not torch.allclose(probe,sample,atol=.001):
                raise RuntimeError('Attention probe produced invalid values')
            del sample,probe,lse
        except RuntimeError:
            available=False
            log.exception('Bounded attention unavailable; retaining native context path')
        properties=torch.cuda.get_device_properties(0)
        engine.context_policy=AdaptiveContext(config,model.head.memory_projection.out_features,
            native_budget=context_budget,stream_enabled=available,
            reserve_mib=int(os.environ.get('CLEF_ROUTING_RESERVE_MIB','256')),
            calibration_path=os.environ.get('CLEF_ROUTING_CALIBRATION',str(ROOT/'routing-calibration.json')),
            fingerprint={'gpu':str(getattr(properties,'uuid',properties.name)),
                'capability':f'{properties.major}.{properties.minor}','profile':PROFILE_NAME,
                'revision':SOURCE_REVISION,'dtype':str(COMPUTE_DTYPE),'torch':torch.__version__,
                'cuda':torch.version.cuda,'initial':PREFILL_INITIAL_CHUNK_TOKENS,'chunk':PREFILL_CHUNK_TOKENS,
                'head_width':model.head.memory_projection.out_features,'config':config.to_dict(),
                'linear':app.state.linear_attention})
    app.state.text_batching = TextBatchAdmission(config,
        padded_tokens=int(os.environ.get("CLEF_BATCH_PADDED_TOKENS", "6000")),
        record_tokens=int(os.environ.get("CLEF_BATCH_RECORD_TOKENS", "4096")),
        length_ratio=float(os.environ.get("CLEF_BATCH_LENGTH_RATIO", ".75")))
    if request_queue.max_batch_size > 1 and STARTUP_STRATEGY.get("runtime_backend") != "cuda":
        raise RuntimeError("Opportunistic text batching is currently CUDA-only")
    refresh_context_budget()
    app.state.cache_stats = {"prefix_cache": engine.stats(), "input_cache": input_cache.stats()}
    await request_queue.start()
    try:
        from .startup_warmup import warmup_on_worker

        def finish_warmup():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            engine.initialize_memory_budget()
            refresh_context_budget()

        app.state.startup_warmup["status"] = "running"
        try:
            app.state.startup_warmup = await warmup_on_worker(request_queue,
                lambda payload: run_decision(DecisionRequest(**payload)), MODEL_ID,
                enabled=warmup_setting == "1", image_pooling=POOLING_DEFAULT, finish=finish_warmup)
        except Exception:
            app.state.startup_warmup["status"] = "failed"
            log.exception("Startup warmup failed; service will not become ready")
            raise
        app.state.cache_stats = {"prefix_cache": engine.stats(), "input_cache": input_cache.stats()}
        app.state.startup_ready = True
        yield
    finally:
        app.state.startup_ready = False
        await request_queue.close()


API_KEY = configured_api_key()
DEMO_MODE = os.environ.get("CLEF_DEMO_MODE", "0") == "1"
app = FastAPI(title=f"Clef {PROFILE_NAME.title()} NF4", lifespan=lifespan,
              dependencies=[Security(HTTPBearer(auto_error=False))] if API_KEY else [])
app.add_middleware(QueueAdmissionMiddleware, budget=request_queue.budget)
# Added last so invalid credentials are rejected before queue/body admission.
app.add_middleware(APIKeyMiddleware, api_key=API_KEY,
                   portal_auto_auth=os.environ.get("CLEF_PORTAL_AUTO_AUTH", "1") == "1",
                   cookie_secure=os.environ.get("CLEF_PORTAL_COOKIE_SECURE", "0") == "1")


@app.get("/ui-config", include_in_schema=False)
def ui_config():
    return JSONResponse({"api_key_required": bool(API_KEY), "demo_mode": DEMO_MODE,
                         "portal_auto_auth": os.environ.get("CLEF_PORTAL_AUTO_AUTH", "1") == "1"},
                        headers={"Cache-Control": "no-store"})


@app.get("/readyz", include_in_schema=False)
def ready():
    # Model startup and warmup finish before HTTP serving. This public probe is
    # also used by Docker when detailed /health requires authorization.
    if not getattr(app.state, "startup_ready", False):
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ok"}

if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

    @app.get("/", include_in_schema=False)
    def playground():
        return FileResponse(UI_DIR / "index.html")


def refresh_context_budget():
    result=context_budget.snapshot(free_bytes=torch.cuda.mem_get_info()[0],
        reserved_bytes=torch.cuda.memory_reserved(), allocated_bytes=torch.cuda.memory_allocated(),
        cache_bytes=engine._bytes())
    if engine.context_policy is not None:
        from host_memory import available_memory
        available=torch.cuda.mem_get_info()[0]+max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())+engine._bytes()
        host=available_memory()
        limits=engine.context_policy.limits(available,host['available_bytes']+engine.host.bytes if host else None)
        longest=max((len(e['ids']) for e in list(engine.host.entries.values())),default=0)
        cached_limits=engine.context_policy.limits(available,host['available_bytes']+engine.host.bytes if host else None,cached_tokens=longest)
        accepted=max(max(limits.values()),max(cached_limits.values()))
        if context_budget.mode=='configured' and os.environ.get('CLEF_CONTEXT_LIMIT_MODE','auto')=='fixed':
            accepted=min(accepted,MAX_LENGTH)
        result['max_input_tokens']=accepted
        # Encoder limits are independent; retain explicit image caps unless auto.
        result['max_input_tokens_with_images']=min(accepted,MAX_IMAGE_LENGTH) if MAX_IMAGE_LENGTH is not None else accepted
        result['context_limit'].update(mode='memory_adaptive',estimated=True,
            computed_max_input_tokens=accepted,configured_cap_applied=os.environ.get('CLEF_CONTEXT_LIMIT_MODE','auto')=='fixed',
            routing_limits={'gpu_native':limits['none'],'gpu_tiled':limits['kv_gpu'],'cpu_streamed':limits['kv_stream']},
            cold_max_input_tokens=max(limits.values()),cached_max_input_tokens=max(cached_limits.values()),
            routing_policy=engine.context_policy.stats(),fallback_reason=None)
    return result


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
        "ready": app.state.startup_ready,
        "startup_warmup": app.state.startup_warmup,
        "queue": request_queue.stats(),
        "text_batching": app.state.text_batching.stats(),
        "status": "ok", "model": PROFILE["repository"], "startup_strategy": STARTUP_STRATEGY, "quantization": "nf4",
        "compact_embeddings": (MODEL_DIR / "compact_quantization.json").is_file(),
        "source_revision": SOURCE_REVISION, "gpu": torch.cuda.get_device_name(),
        "compute_dtype": PROFILE["compute_dtype"], "nf4_layers": app.state.nf4_layers,
        "linear_attention": app.state.linear_attention,
        "vision_attention": app.state.vision_attention,
        "matrix_tuning": app.state.matrix_tuning,
        **context_metadata(),
        "input": ["text", "json", "images"],
        "image_input": {"format": "base64 data URL", "max_images": 16,
                        "max_processed_pixels_per_image": processor.image_processor.size["longest_edge"],
                        "max_vision_patch_tokens_per_image": MAX_VISION_PATCH_TOKENS or None,
                        "resize_location": "client_for_gui", "processor_defaults_for_api": True,
                        "legacy_fidelity_max_pixels": MAX_IMAGE_PIXELS,
                        "default_fidelity": "standard", "fidelities": IMAGE_FIDELITIES,
                        "pooling_default": POOLING_DEFAULT, "pooling_factor": 2},
        "attention_backend": ATTENTION_BACKEND,
        "optimizations": {"compact_schema": True, "single_pass_preprocessing": True,
                          "observed_chunked_workspace_admission": (STARTUP_STRATEGY.get("runtime_backend") == "rocm"
                              and os.environ.get("CLEF_ROCM_OBSERVED_PREFILL_WORKSPACE", "0") == "1"),
                          "chunked_prefill": {"enabled": PREFILL_CHUNK_TOKENS > 0, "text_only": not IMAGE_PREFILL_ENABLED, "images_enabled": IMAGE_PREFILL_ENABLED, "vision_batch_images": 1,
                                              "initial_chunk_tokens": PREFILL_INITIAL_CHUNK_TOKENS,
                                              "chunk_tokens": PREFILL_CHUNK_TOKENS,
                                              "active_context_offload": engine.active_context_offload,
                                              "cpu_head_prepare": os.environ.get("CLEF_HEAD_CPU_PREPARE", "0") == "1",
                                              "head_prepare_chunk_tokens": int(os.environ.get("CLEF_HEAD_PREPARE_CHUNK_TOKENS", "4096")),
                                              "rocm_long_prompt_threshold": ROCM_LONG_PROMPT_THRESHOLD or None,
                                              "rocm_long_chunk_tokens": ROCM_LONG_CHUNK_TOKENS if ROCM_LONG_PROMPT_THRESHOLD else None},
                          "prefix_cache_enabled": PREFIX_CACHE_ENABLED and GPU_CACHE_SUPPORTED,
                          "image_feature_cache_enabled": INPUT_CACHE_ENABLED and GPU_CACHE_SUPPORTED,
                          "promote_repeated_boundary": engine.promote_repeated_boundary,
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
    max_mib=os.environ.get("CLEF_PREPROCESS_CACHE_MIB", "auto"),
    max_entries=None if os.environ.get("CLEF_PREPROCESS_CACHE_ENTRIES", "auto").lower() == "auto"
        else int(os.environ["CLEF_PREPROCESS_CACHE_ENTRIES"]),
    token_mib=int(os.environ.get("CLEF_TOKEN_CACHE_MIB", "8")),
    token_entries=int(os.environ.get("CLEF_TOKEN_CACHE_ENTRIES", "1024")), decode=decode_image,
    ram_budget=ram_budget, image_fraction=float(os.environ.get("CLEF_PREPROCESS_CACHE_RAM_FRACTION", "0.25")))


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                       **context_metadata(),
                                       **({"max_vision_patch_tokens_per_image": MAX_VISION_PATCH_TOKENS} if MAX_VISION_PATCH_TOKENS else {})}]}


@app.post("/v1/systemone")
async def decide(request: DecisionRequest, http_request: Request):
    try:
        return await request_queue.run(request, http_request)
    except QueueError as exc:
        return exc.response()


def request_record(request: DecisionRequest):
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
    return record


def media_cache_key(encoded, images, options):
    """Per-image prefix key, or the whole-media key when images can't be mapped."""
    if images:
        try:
            return MediaPrefix.build(processor, encoded, images, options)
        except ValueError:
            log.warning("Image prefix boundaries unavailable; using whole-media cache key")
    return hashlib.sha256(json.dumps([images, options], separators=(',', ':')).encode()).hexdigest()


def prepare_decision(request, overlapped=False, record=None):
    started = time.perf_counter()
    try:
        record = request_record(request) if record is None else record
        before = input_cache.usage()
        encoded, boundary, checkpoints = encode_compact(processor, record, with_checkpoints=True,
            input_cache=input_cache, cache_enabled=request.input_cache and INPUT_CACHE_ENABLED)
        after = input_cache.usage()
        usage = {'preprocessing_ms':round((time.perf_counter()-started)*1000,1),
            'image_preprocess_cache_hits':after['image_hits']-before['image_hits'],
            'image_preprocess_cache_misses':after['image_misses']-before['image_misses'],
            'token_cache_hits':after['token_hits']-before['token_hits'],
            'token_cache_misses':after['token_misses']-before['token_misses'],
            'cpu_preparation_overlapped':bool(overlapped)}
        key = media_cache_key(encoded, request.images, record.get('media_kwargs'))
        record.pop('_cache_image_snapshots',None)
        return PreparedDecision(request,record,encoded,boundary,checkpoints,usage,key)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400,str(exc)) from exc


def prepare_queued_decision(request):
    cap = int(os.environ.get('CLEF_CPU_PREPARE_MIB','1024'))
    record = request_record(request)
    if os.environ.get('CLEF_CPU_PREPARE_OVERLAP','0')!='1':
        maximum=int(os.environ.get('CLEF_INTERLEAVE_MAX_TOKENS','2048'))
        if request.images or len(str(request.state))+len(str(request.context))>maximum*4:return None
        if not preparation_fits(request,cap):return None
        prepared=prepare_decision(request,True,record)
        return prepared if len(prepared.encoded.input_ids)<=maximum else None
    if request.images and request.input_cache and INPUT_CACHE_ENABLED:
        snapshots = input_cache.image_snapshots(processor,record)
        sample = ram_budget.memory_probe()
        if snapshots is not None and sample is not None:
            size = sum(size for _, _, size in snapshots)
            if size*1.25 <= min(cap*2**20,sample['available_bytes']//8):
                record['_cache_image_snapshots'] = snapshots
                return prepare_decision(request,True,record)
    if not preparation_fits(request, cap):
        return None
    return prepare_decision(request, True, record)


def offer_short_request(cursor,total):
    if os.environ.get('CLEF_CHUNK_INTERLEAVE','0') != '1' or STARTUP_STRATEGY.get('runtime_backend')!='cuda':return
    prepared=request_queue.peek_prepared()
    if prepared is None or prepared.images or len(prepared.encoded.input_ids)>int(os.environ.get('CLEF_INTERLEAVE_MAX_TOKENS','2048')):return
    if total-cursor<=len(prepared.encoded.input_ids):return
    required=context_budget.workspace(len(prepared.encoded.input_ids))+engine.prefill_reserve_bytes
    if not context_budget.computed or available_cuda_memory()<required:return
    workspace=engine.request_workspace_bytes
    profile=gpu_profiles.stack[-1]
    profile['peak']=max(profile['peak'],torch.cuda.max_memory_allocated())
    if engine.async_restore is not None:engine.async_restore.finish()
    # The child runs uncached to bound its footprint alongside the parent's
    # live KV/hidden state. It produces the same typed decisions as an ordinary
    # uncached request and cannot recursively interleave another request.
    # Keep identity with the queued prepared object for FIFO admission, then
    # use a per-request override in the handler rather than changing payloads.
    saved_hook=engine.progress_hook
    engine.progress_hook=None
    try:
        prepared.usage['_interleave_uncached']=True
        request_queue.run_interleaved(prepared,int(os.environ.get('CLEF_INTERLEAVE_MAX_PER_PARENT','4')))
    finally:
        prepared.usage.pop('_interleave_uncached',None)
        engine.progress_hook=saved_hook
        engine.prepare_request(workspace)
        profile['peak']=max(profile['peak'],torch.cuda.max_memory_allocated())


def run_decision(request: DecisionRequest):
    started = time.perf_counter()
    lock.acquire()
    if not hasattr(gpu_profiles,'stack'):gpu_profiles.stack=[]
    gpu_profiles.stack.append({'peak':0})
    try:
        prepared = request if isinstance(request, PreparedDecision) else prepare_decision(request)
        request, record = prepared.request, prepared.record
        encoded, boundary, checkpoints = prepared.encoded, prepared.boundary, prepared.checkpoints
        preprocess_usage = {key:value for key,value in prepared.usage.items() if not key.startswith('_')}
        reuse_inputs = request.input_cache and INPUT_CACHE_ENABLED
        original_tokens = len(encoded.input_ids)
        # Vision encoding precedes optional feature pooling and has its own
        # unmerged patch workload. Bound the experimentally validated HIP path
        # before submitting any GPU work; pooling cannot bypass this guard.
        if MAX_VISION_PATCH_TOKENS and encoded.media is not None:
            for index, grid in enumerate(encoded.media["image_grid_thw"].tolist(), 1):
                patches = math.prod(grid)
                if patches > MAX_VISION_PATCH_TOKENS:
                    return JSONResponse(status_code=413, content={
                        "detail": f"Image {index} exceeds {MAX_VISION_PATCH_TOKENS} vision patch tokens; downscale this image",
                        "image_index": index, "vision_patch_tokens": patches,
                        "max_vision_patch_tokens_per_image": MAX_VISION_PATCH_TOKENS})
        if request.image_pooling and request.images:
            original = encoded
            encoded, boundary = pool_record(encoded, boundary, model.language_model.config.image_token_id)
            checkpoints = remap_checkpoints(checkpoints, original, encoded)
        input_key = prepared.input_key
        if request.image_pooling and isinstance(input_key, MediaPrefix):
            input_key = input_key.rebuild(processor, encoded)
        limits = refresh_context_budget()
        request_limit = limits["max_input_tokens_with_images" if request.images else "max_input_tokens"]
        if len(encoded.input_ids) > request_limit:
            return JSONResponse(status_code=413, content={
                "detail": f"Request exceeds {request_limit} input tokens",
                "input_tokens": len(encoded.input_ids), "max_input_tokens": request_limit,
                "context_limit_mode": limits["context_limit"]["mode"]})
        execution_plan=None
        if engine.context_policy is not None:
            from host_memory import available_memory
            host=available_memory()
            available=limits['context_limit']['available_request_mib']*2**20
            spans=[q.question_span for q in encoded.questions]+[s for q in encoded.questions for s in q.option_spans]
            schema_bytes=max((b-a for a,b in spans),default=0)*engine.context_policy.hidden*2
            media_start=encoded.media['token_offset'] if encoded.media else None
            media_end=media_start+len(encoded.media['mm_token_type_ids']) if encoded.media else 0
            match,_=engine._match(model,encoded.input_ids,boundary,input_key,request.image_pooling,media_start,media_end)
            source=engine.host.entries.get(match)
            protected_cpu=source['bytes'] if source else 0
            cached_tokens=len(source['ids']) if source else 0
            del source
            host_capacity=host['available_bytes']+max(0,engine.host.bytes-protected_cpu) if host else None
            execution_plan=engine.context_policy.choose(len(encoded.input_ids),available,host_capacity,schema_bytes=schema_bytes,cached_tokens=cached_tokens)
            if execution_plan is None:
                return JSONResponse(status_code=413,content={'detail':'Request exceeds measured memory admission for its matching prefix','input_tokens':len(encoded.input_ids),'max_input_tokens':limits['context_limit'].get('cold_max_input_tokens',request_limit)})
            if execution_plan['mode']!='none':
                engine.reclaim_host_workspace(len(encoded.input_ids),execution_plan['mode'],model,input_key,request.image_pooling,boundary,encoded)
        cache_preparation = engine.prepare_request(execution_plan['workspace_bytes'] if execution_plan and (execution_plan['mode']!='none' or len(encoded.input_ids)>PREFILL_INITIAL_CHUNK_TOKENS) else context_budget.workspace(len(encoded.input_ids))
            if context_budget.computed and len(encoded.input_ids)>PREFILL_INITIAL_CHUNK_TOKENS else None,
            observed_chunked=(STARTUP_STRATEGY.get("runtime_backend") == "rocm"
                and os.environ.get("CLEF_ROCM_OBSERVED_PREFILL_WORKSPACE", "0") == "1"
                and PREFILL_CHUNK_TOKENS > 0 and len(encoded.input_ids)>PREFILL_INITIAL_CHUNK_TOKENS))
        routing_cache_start=engine._bytes()
        torch.cuda.reset_peak_memory_stats()
        caching = request.prefix_cache and PREFIX_CACHE_ENABLED and GPU_CACHE_SUPPORTED and not prepared.usage.get('_interleave_uncached',False)
        initial_chunk, chunk_size = PREFILL_INITIAL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS
        if (STARTUP_STRATEGY.get("runtime_backend") == "rocm" and ROCM_LONG_PROMPT_THRESHOLD
                and len(encoded.input_ids) > ROCM_LONG_PROMPT_THRESHOLD and chunk_size):
            initial_chunk = min(initial_chunk, ROCM_LONG_CHUNK_TOKENS)
            chunk_size = min(chunk_size, ROCM_LONG_CHUNK_TOKENS)
        engine.progress_hook = offer_short_request if not request_queue.nesting else None
        gpu_timer = GPUTimer(torch.cuda)
        def infer(cache, features=True):
            attention = (sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if ATTENTION_BACKEND == "efficient"
                         else sdpa_kernel(SDPBackend.MATH) if ATTENTION_BACKEND == "math" else nullcontext())
            with attention, gpu_timer.measure():
                return engine.forward(model, processor, encoded, boundary, input_key,
                    pooling=request.image_pooling, cache_enabled=cache, checkpoint_boundaries=checkpoints,
                    feature_cache_enabled=features and reuse_inputs and GPU_CACHE_SUPPORTED,
                    prefill_chunk_tokens=chunk_size,
                    prefill_initial_chunk_tokens=initial_chunk, prefill_images=IMAGE_PREFILL_ENABLED,
                    execution_plan=execution_plan)
        attempts=[]
        while True:
            try:
                logits,cache_usage=infer(caching)
                break
            except torch.cuda.OutOfMemoryError as exc:
                if engine.context_policy is None:
                    # Legacy behavior retains its single cache-free retry.
                    exc.__traceback__=None;engine.clear();gc.collect();torch.cuda.empty_cache()
                    if not caching and not reuse_inputs:raise
                    logits,cache_usage=infer(False,False)
                    cache_usage['prefix_cache']='memory_fallback';break
                failed_mode=execution_plan['mode']
                engine.context_policy.record_oom(failed_mode,len(encoded.input_ids),available)
                attempts.append(failed_mode)
                exc.__traceback__=None
                engine.spill_gpu_cache();gc.collect();torch.cuda.empty_cache()
                if failed_mode=='kv_stream':raise
                host=available_memory()
                available=torch.cuda.mem_get_info()[0]+max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())+engine._bytes()
                execution_plan=engine.context_policy.choose(len(encoded.input_ids),available,host['available_bytes'] if host else None,
                    schema_bytes=schema_bytes,minimum_mode='kv_gpu' if failed_mode=='none' else 'kv_stream')
                if execution_plan is None:raise
                engine.prepare_request(execution_plan['workspace_bytes'])
        if execution_plan is not None:
            cache_usage.update(execution_strategy={'none':'gpu_native','kv_gpu':'gpu_tiled','kv_stream':'cpu_streamed'}[execution_plan['mode']],
                context_switch_reason=execution_plan['reason'],context_fallbacks=attempts)
            if execution_plan['mode']!='none' or engine._bytes()==0:
                engine.context_policy.observe(execution_plan['mode'],len(encoded.input_ids),
                    max(0,torch.cuda.max_memory_allocated()-(engine.base_bytes or 0)-min(routing_cache_start,engine._bytes())),reused_tokens=cache_usage.get('reused_prefix_tokens',0))
        result = {"model": request.model, "answers": answers(record, encoded, logits),
                  "usage": {"batch_size":1, "input_tokens": len(encoded.input_ids), "output_tokens": 0,
                            "unpooled_input_tokens": original_tokens, "compact_schema": True,
                            "image_pooling": bool(request.image_pooling and request.images),
                            **preprocess_usage, **cache_preparation, **cache_usage}}
        if engine.context_policy is None and cache_usage.get("prefill_mode") == "chunked" and engine._bytes() == 0 and (not request.images or context_budget.image_computed):
            context_budget.observe(len(encoded.input_ids),
                max(torch.cuda.max_memory_allocated(),gpu_profiles.stack[-1]['peak']) - engine.base_bytes)
        result["usage"].update(accepted_max_input_tokens=request_limit,
                               context_limit_mode=limits["context_limit"]["mode"])
        json.dumps(result, allow_nan=False)
        torch.cuda.synchronize()
        result["usage"].update(gpu_time_ms=round(gpu_timer.elapsed_ms, 1),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            peak_allocated_mib=round(max(torch.cuda.max_memory_allocated(),gpu_profiles.stack[-1]['peak']) / 2**20, 1))
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
        if engine.context_policy is None and "encoded" in locals() and (not request.images or context_budget.image_computed):
            context_budget.record_oom(len(encoded.input_ids))
        exc.__traceback__ = None
        if engine.context_policy is None:engine.clear()
        else:engine.spill_gpu_cache()
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(503, "Insufficient GPU memory; shorten the request") from None
    finally:
        try:
            engine.finish_request()
            engine.progress_hook=None
        finally:
            gpu_profiles.stack.pop()
            lock.release()


def text_batch_key(request):
    if isinstance(request, PreparedDecision): request = request.request
    # Raw context equality is a cheap first filter. GPU execution additionally
    # checks exact encoded prefix identity and each row's actual token count.
    if request.images:
        return None
    return (request.model, request.prefix_cache, request.input_cache, request.context)


def run_decision_batch(requests, emit):
    """One worker, zero collection wait; split or emit serial replies immediately."""
    batch_started = time.perf_counter()
    admission = app.state.text_batching

    def single(index, reason):
        began = time.perf_counter()
        try:
            result = run_decision(requests[index])
            if isinstance(result, dict):
                result['usage'].update(batch_size=1, batch_fallback_reason=reason,
                    batch_worker_wait_ms=round((began-batch_started)*1000,1))
        except Exception as exc:
            result = exc
        emit(index, result)

    def group(indices):
        if len(indices) == 1:
            single(indices[0], 'single_request')
            return
        reason = None
        started = time.perf_counter()
        with lock:
            prepared = []
            # Invalid/oversized members do not poison unrelated requests.
            for index in indices:
                request = requests[index]
                try:
                    cached = request if isinstance(request, PreparedDecision) else prepare_decision(request)
                    request, record = cached.request, cached.record
                    e, boundary, points = cached.encoded, cached.boundary, cached.checkpoints
                    limit = refresh_context_budget()['max_input_tokens']
                    if len(e.input_ids) > limit:
                        emit(index, JSONResponse(status_code=413, content={
                            'detail': f'Request exceeds {limit} input tokens', 'input_tokens': len(e.input_ids),
                            'max_input_tokens': limit, 'context_limit_mode': context_budget.mode}))
                        continue
                    prepared.append((index, request, record, e, boundary, points, limit, cached.usage))
                except (ValueError, KeyError, TypeError) as exc:
                    emit(index, HTTPException(400,str(exc)))
                except Exception as exc:
                    emit(index, exc)
            indices = [p[0] for p in prepared]
            if not indices:
                return
            lengths = [len(p[3].input_ids) for p in prepared]
            if len(indices) < 2:
                reason = 'single_valid_request'
            else:
                reason = admission.reason(lengths)
            if reason is None:
                encoded = [p[3] for p in prepared]
                boundaries = [p[4] for p in prepared]
                common = shared_tokens(encoded, boundaries)
                caching = requests[indices[0]].prefix_cache and PREFIX_CACHE_ENABLED and GPU_CACHE_SUPPORTED
                key = hashlib.sha256(json.dumps([[],None],separators=(',',':')).encode()).hexdigest()
                plan = engine.plan_text_batch(model, encoded, boundaries, key, caching)
                projected = admission.workspace(lengths, common if plan['enabled'] else 0)
                # Cache bytes can be reclaimed before inference; never count
                # them as both live workspace and extra free capacity.
                reclaimable = engine._bytes()
                if plan['reason']:
                    reason = plan['reason']
                elif available_cuda_memory()+reclaimable < projected+engine.prefill_reserve_bytes:
                    reason = 'memory_estimate'
                else:
                    cache_preparation = engine.prepare_request(projected, protected=plan['key'])
                    if available_cuda_memory() < projected+engine.prefill_reserve_bytes:
                        reason = 'memory_headroom'
                    else:
                        preprocess_ms = (time.perf_counter()-started)*1000
                        retained = engine._bytes()
                        torch.cuda.reset_peak_memory_stats()
                        try:
                            attention = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
                            caching = requests[indices[0]].prefix_cache and PREFIX_CACHE_ENABLED and GPU_CACHE_SUPPORTED
                            key = hashlib.sha256(json.dumps([[],None],separators=(',',':')).encode()).hexdigest()
                            gpu_timer = GPUTimer(torch.cuda)
                            with attention, gpu_timer.measure():
                                logits, usage = engine.forward_text_batch(model, processor, encoded, boundaries, key,
                                    cache_enabled=caching, checkpoint_boundaries=[p[5] for p in prepared], cache_plan=plan)
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter()-started)*1000
                            peak = torch.cuda.max_memory_allocated()
                            admission.observe(lengths, max(0,peak-(engine.base_bytes or 0)-min(retained,engine._bytes())))
                            for p,row in zip(prepared,logits):
                                index, request, record, e, boundary, points, limit, prep = p
                                result={'model':request.model,'answers':answers(record,e,row),'usage':{
                                    'input_tokens':len(e.input_ids),'output_tokens':0,'unpooled_input_tokens':len(e.input_ids),
                                    'compact_schema':True,'image_pooling':False, 'preprocessing_ms':round(preprocess_ms,1), **prep, **cache_preparation, **usage,
                                    'prefix_tokens':boundary,'new_prefix_tokens':boundary-usage['reused_prefix_tokens'],
                                    'accepted_max_input_tokens':limit,'context_limit_mode':context_budget.mode,
                                    'batch_size':len(indices),'batch_padded_tokens':len(indices)*max(lengths),
                                    'batch_padding_tokens':len(indices)*max(lengths)-sum(lengths),
                                    'batch_worker_wait_ms':round((started-batch_started)*1000,1),
                                    'gpu_time_ms':round(gpu_timer.elapsed_ms,1),
                                    'latency_ms':round(elapsed,1),'peak_allocated_mib':round(peak/2**20,1)}}
                                json.dumps(result,allow_nan=False)
                                emit(index,result)
                            return
                        except BatchUnavailable as exc:
                            exc.__traceback__ = None
                            reason = str(exc)
                        except torch.cuda.OutOfMemoryError as exc:
                            # Aggregate OOM must not lower the single-request
                            # context ceiling or return a spurious 413.
                            exc.__traceback__ = None
                            admission.oom(lengths)
                            engine.clear();gc.collect();torch.cuda.empty_cache()
                            reason = 'gpu_oom'
                        except Exception as exc:
                            exc.__traceback__ = None
                            reason = 'batch_runtime_error'
                            log.exception('Text batch failed; isolating requests')
                        finally:
                            engine.finish_request()
            engine.finish_request()
        admission.fallback(reason)
        if reason in {'better_individual_prefix','exact_state_promotion','batch_runtime_error'} or len(indices) < 2:
            for index in indices:
                single(index, reason)
        else:
            # Downsize by contiguous groups; preserve arrival order and bound
            # retries (4 -> 2 -> 1). No collection delay or model sharding.
            split = max(1,len(indices)//2)
            group(indices[:split]);group(indices[split:])

    group(list(range(len(requests))))
