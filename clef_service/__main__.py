"""python -m clef_service [--full] starts the unified service."""
import argparse
import json
import logging
import os
from pathlib import Path

from .hardware import configure_strategy, detect_strategy, select_gpu
from .models import ROOT, download_source, ensure_checkpoint, profile_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-GPU Clef: detect, download, cache, and serve.")
    parser.add_argument("action", nargs="?", choices=["serve", "download", "prepare", "inspect"], default="serve")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", choices=["flash", "full"], default=os.environ.get("CLEF_PROFILE", "flash"))
    models.add_argument("--full", action="store_true", help="Download/prepare/serve larger Clef 27B NF4")
    parser.add_argument("--gpu", help="Select one GPU index or UUID before CUDA initialization")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("CLEF_CACHE_DIR", ROOT / "models")),
                        help="Persistent parent cache; each model has its own subdirectory")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ["CLEF_DATA_DIR"]) if os.environ.get("CLEF_DATA_DIR") else None, help="Legacy exact profile directory (no model subdirectory)")
    parser.add_argument("--checkpoint", type=Path, help="Use an existing prepared checkpoint; no download or preparation")
    parser.add_argument("--offline", action="store_true", help="Require a cached checkpoint/source; no downloads")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-length", type=int, help="Override computed admission with a fixed token cap")
    parser.add_argument("--max-image-length", help="Image token limit, or auto")
    parser.add_argument("--context-reserve-mib", type=int)
    parser.add_argument("--prefill-initial-chunk-tokens", type=int)
    parser.add_argument("--prefill-chunk-tokens", type=int)
    parser.add_argument("--allow-slow-kernels", action="store_true", help="Permit native fallback if FLA is unavailable")
    parser.add_argument("--linear-prefill-backend", choices=["auto", "torch", "fla", "adaptive"])
    parser.add_argument("--image-pooling", action="store_true", help="Enable optional experimental 2x2 pooling")
    parser.add_argument("--image-prefill", action="store_true", help="Compatibility flag; image prefill defaults on")
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    profile = "full" if args.full else args.model
    config = profile_config(profile)
    data = (args.data_dir or args.cache_dir / profile).expanduser().resolve()
    os.environ["CLEF_PROFILE"] = profile
    os.environ["CLEF_DATA_DIR"] = str(data)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HOME", str(args.cache_dir.expanduser().resolve() / "huggingface"))
    os.environ.setdefault("CLEF_ATTENTION_BACKEND", "efficient")
    os.environ.setdefault("CLEF_IMAGE_PREFILL", "1")
    os.environ.setdefault("CLEF_MAX_IMAGE_LENGTH", "24576" if profile == "flash" else "auto")
    for field, env in [("max_length", "CLEF_MAX_LENGTH"), ("max_image_length", "CLEF_MAX_IMAGE_LENGTH"),
                       ("context_reserve_mib", "CLEF_CONTEXT_RESERVE_MIB"),
                       ("prefill_initial_chunk_tokens", "CLEF_PREFILL_INITIAL_CHUNK_TOKENS"),
                       ("prefill_chunk_tokens", "CLEF_PREFILL_CHUNK_TOKENS"),
                       ("linear_prefill_backend", "CLEF_LINEAR_PREFILL_BACKEND")]:
        if getattr(args, field) is not None:
            os.environ[env] = str(getattr(args, field))
    if args.max_length is not None:
        os.environ["CLEF_CONTEXT_LIMIT_MODE"] = "fixed"
    if args.image_pooling:
        os.environ["CLEF_IMAGE_POOLING"] = "1"
    if args.no_prefix_cache:
        os.environ["CLEF_PREFIX_CACHE"] = "0"
    if args.allow_slow_kernels:
        os.environ["CLEF_REQUIRE_FAST_LINEAR_ATTENTION"] = "0"
        os.environ.setdefault("CLEF_LINEAR_PREFILL_BACKEND", "auto")
    try:
        if args.action == "download":
            data.mkdir(parents=True, exist_ok=True)
            download_source(data, config, args.offline)
            return
        select_gpu(args.gpu)
        strategy = detect_strategy(profile)
        configure_strategy(strategy)
        logging.getLogger("clef").info("Selected strategy: %s", strategy.metadata())
        if args.action == "inspect":
            print(json.dumps(strategy.metadata(), indent=2))
            return
        external = args.checkpoint
        if external is None and os.environ.get("CLEF_MODEL_DIR"):
            external = data / os.environ["CLEF_MODEL_DIR"]
        checkpoint = ensure_checkpoint(data, profile, args.offline, external)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Clef startup: {exc}\n")
    if args.action == "prepare":
        print(f"Checkpoint ready: {checkpoint}")
        return
    os.environ["CLEF_MODEL_DIR"] = str(checkpoint)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Keep compiled kernels separate across architectures and dtype.
    cache = data / "cache" / f"sm{strategy.capability.replace('.', '')}-{strategy.compute_dtype}"
    for name in ["triton", "torch-kernels"]:
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(cache / "torch-kernels"))
    import uvicorn
    uvicorn.run("clef_service.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
