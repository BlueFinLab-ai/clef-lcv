"""python -m clef_service [--full] starts the unified service."""
import argparse
import json
import logging
import math
import os
from pathlib import Path

from .hardware import configure_strategy, detect_strategy, select_gpu, prepare_runtime, runtime_backend_from_build
from .prebuilt_launchers import install_prebuilt_launchers
from .models import ROOT, download_source, ensure_checkpoint, profile_config
from .access import configured_api_key, configured_hf_token


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-GPU Clef: detect, download, cache, and serve.")
    parser.add_argument("action", nargs="?", choices=["serve", "download", "prepare", "inspect"], default="serve")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", choices=["flash", "full"], default=os.environ.get("CLEF_PROFILE", "flash"))
    models.add_argument("--full", action="store_true", help="Download/prepare/serve larger Clef 27B NF4")
    parser.add_argument("--experimental-rocm", action="store_true", help="Opt into the tested RX 580 gfx803 Flash strategy in a HIP build")
    parser.add_argument("--gpu", help="Select one GPU index or UUID before GPU initialization")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("CLEF_CACHE_DIR", ROOT / "models")),
                        help="Persistent parent cache; each model has its own subdirectory")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ["CLEF_DATA_DIR"]) if os.environ.get("CLEF_DATA_DIR") else None, help="Legacy exact profile directory (no model subdirectory)")
    parser.add_argument("--checkpoint", type=Path, help="Use an existing prepared checkpoint; no download or preparation")
    parser.add_argument("--offline", action="store_true", help="Require a cached checkpoint/source; no downloads")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    keys = parser.add_mutually_exclusive_group()
    keys.add_argument("--api-key", help="Require this Bearer key for API clients; the portal is authorized automatically")
    keys.add_argument("--api-key-file", type=Path, help="Require the Bearer key from this secret file")
    parser.add_argument("--require-api-key", action="store_true", help="Fail startup if no API key is configured")
    parser.add_argument("--demo", action="store_true", help="Replace server addresses in the portal with placeholders")
    parser.add_argument("--hf-token-file", type=Path,
                        help="Hugging Face token file for model downloads; or set HF_TOKEN / HF_TOKEN_FILE")
    parser.add_argument("--max-length", type=int, help="Override computed admission with a fixed token cap")
    parser.add_argument("--max-image-length", help="Image token limit, or auto")
    parser.add_argument("--context-reserve-mib", type=int)
    parser.add_argument("--prefill-initial-chunk-tokens", type=int)
    parser.add_argument("--prefill-chunk-tokens", type=int)
    parser.add_argument('--prefill-policy', choices=['fixed','memory'],
                        help='Choose prefill chunks from reusable VRAM (CUDA default) or retain fixed chunks')
    parser.add_argument('--prefill-max-chunk-tokens', type=int,
                        help='Maximum new tokens per adaptive chunk (default 16384; VRAM may select less)')
    parser.add_argument('--prefill-target-chunk-tokens',
                        help='Preferred adaptive continuation size, or auto (native/tiled 4K, CPU-streamed 8K)')
    parser.add_argument("--allow-slow-kernels", action="store_true", help="Permit native fallback if FLA is unavailable")
    parser.add_argument("--linear-prefill-backend", choices=["auto", "torch", "fla", "adaptive"])
    parser.add_argument("--batch-size", type=int, choices=[1,2,4], help="Maximum opportunistic CUDA text batch; no collection delay")
    independent = parser.add_mutually_exclusive_group()
    independent.add_argument('--independent-cache-batching',action='store_true',help='Batch qualified CUDA Flash text suffixes with independent cached histories')
    independent.add_argument('--no-independent-cache-batching',action='store_true',help='Disable independent cached-history batching')
    matching = parser.add_mutually_exclusive_group()
    matching.add_argument('--batch-matching', action='store_true', help='Match bounded queued text by length and exact cache ancestry')
    matching.add_argument('--no-batch-matching', action='store_true', help='Use consecutive FIFO text batches')
    parser.add_argument('--batch-lookahead', type=int, choices=range(1,17), help='Inspect at most this many waiting text requests (default 8)')
    parser.add_argument('--batch-max-bypasses', type=int, choices=range(0,9), help='Stop a matching search at a request bypassed this many times (default 2)')
    parser.add_argument("--image-pooling", action="store_true", help="Enable optional experimental 2x2 pooling")
    parser.add_argument("--image-prefill", action="store_true", help="Compatibility flag; image prefill defaults on")
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--cache-frequency-decay-seconds", type=float,
                        default=os.environ.get("CLEF_CACHE_FREQUENCY_DECAY_SECONDS", "900"),
                        help="Halve CPU prefix/image eviction frequency each interval (default 900 seconds; 0 disables aging)")
    parser.add_argument("--ram-cache-reserve-fraction", type=float,
                        default=os.environ.get("CLEF_PREFIX_HOST_CACHE_RESERVE_FRACTION", "0.20"),
                        help="Fraction of service-allocatable CPU RAM reserved outside shared caches (default 0.20 = 20%%)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip default startup text/image warmup before readiness")
    metric_flags = parser.add_mutually_exclusive_group()
    metric_flags.add_argument('--metrics', action='store_true', help='Enable best-effort background metrics at protected /metrics')
    metric_flags.add_argument('--no-metrics', action='store_true', help='Disable optional metrics collection')
    parser.add_argument('--request-log-mode', choices=['off','metadata','payload'],
                        default=os.environ.get('CLEF_REQUEST_LOG_MODE','off'))
    parser.add_argument('--request-log-dir', type=Path,
                        default=Path(os.environ['CLEF_REQUEST_LOG_DIR']) if os.environ.get('CLEF_REQUEST_LOG_DIR') else None)
    parser.add_argument('--request-log-images', action='store_true', help='Include received image bytes in optional payload archive')
    parser.add_argument('--request-log-queue-mib', type=int)
    parser.add_argument('--request-log-rotate-mib', type=int)
    parser.add_argument('--request-log-backups', type=int)
    parser.add_argument('--request-log-retention-days', type=float)
    parser.add_argument('--request-log-archive-mib', type=int)
    args = parser.parse_args(argv)
    if args.independent_cache_batching or args.no_independent_cache_batching:
        os.environ['CLEF_INDEPENDENT_CACHE_BATCHING']='1' if args.independent_cache_batching else '0'
    if args.batch_matching or args.no_batch_matching:
        os.environ['CLEF_BATCH_MATCHING'] = '1' if args.batch_matching else '0'
    if args.batch_lookahead is not None:
        os.environ['CLEF_BATCH_LOOKAHEAD'] = str(args.batch_lookahead)
    if args.batch_max_bypasses is not None:
        os.environ['CLEF_BATCH_MAX_BYPASSES'] = str(args.batch_max_bypasses)
    if args.prefill_max_chunk_tokens is not None and args.prefill_max_chunk_tokens<1:
        parser.error('--prefill-max-chunk-tokens must be positive')
    if args.prefill_target_chunk_tokens is not None and args.prefill_target_chunk_tokens!='auto':
        try:
            if int(args.prefill_target_chunk_tokens)<1:raise ValueError
        except ValueError:
            parser.error('--prefill-target-chunk-tokens must be auto or a positive integer')
    if args.metrics or args.no_metrics:
        os.environ['CLEF_METRICS'] = '1' if args.metrics else '0'
    os.environ['CLEF_REQUEST_LOG_MODE'] = args.request_log_mode
    if args.request_log_dir is not None:
        os.environ['CLEF_REQUEST_LOG_DIR'] = str(args.request_log_dir)
    if args.request_log_images:
        os.environ['CLEF_REQUEST_LOG_IMAGES'] = '1'
    for field in ('queue_mib','rotate_mib','backups','retention_days','archive_mib'):
        value = getattr(args, 'request_log_'+field)
        if value is not None:
            os.environ['CLEF_REQUEST_LOG_'+field.upper()] = str(value)
    from runtime.observability import ObservationConfig
    try:
        ObservationConfig.from_env()
    except (ValueError, OverflowError) as exc:
        parser.error(str(exc))
    if (not math.isfinite(args.cache_frequency_decay_seconds) or args.cache_frequency_decay_seconds < 0
            or 0 < args.cache_frequency_decay_seconds < 1e-9):
        parser.error("--cache-frequency-decay-seconds must be finite and nonnegative (at least 1 ns when enabled)")
    os.environ["CLEF_CACHE_FREQUENCY_DECAY_SECONDS"] = str(args.cache_frequency_decay_seconds)
    if not math.isfinite(args.ram_cache_reserve_fraction) or not 0 <= args.ram_cache_reserve_fraction < 1:
        parser.error("--ram-cache-reserve-fraction must be finite and in [0, 1)")
    os.environ["CLEF_PREFIX_HOST_CACHE_RESERVE_FRACTION"] = str(args.ram_cache_reserve_fraction)
    if args.api_key is not None:
        if not args.api_key:
            parser.error("--api-key cannot be empty")
        os.environ.pop("CLEF_API_KEY_FILE", None)
        os.environ["CLEF_API_KEY"] = args.api_key
    if args.api_key_file is not None:
        os.environ.pop("CLEF_API_KEY", None)
        os.environ["CLEF_API_KEY_FILE"] = str(args.api_key_file)
    if args.require_api_key:
        os.environ["CLEF_REQUIRE_API_KEY"] = "1"
    if args.demo:
        os.environ["CLEF_DEMO_MODE"] = "1"
    if args.hf_token_file is not None:
        os.environ.pop("HF_TOKEN", None)
        os.environ["HF_TOKEN_FILE"] = str(args.hf_token_file)
    try:
        configured_api_key()
        # huggingface_hub reads HF_TOKEN itself; resolve a token file into it.
        hf_token = configured_hf_token()
    except ValueError as exc:
        parser.exit(2, f"Clef startup: {exc}\n")
    os.environ.pop("HF_TOKEN_FILE", None)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    profile = "full" if args.full else args.model
    config = profile_config(profile)
    data = (args.data_dir or args.cache_dir / profile).expanduser().resolve()
    os.environ["CLEF_PROFILE"] = profile
    os.environ["CLEF_DATA_DIR"] = str(data)
    os.environ.setdefault("HF_HOME", str(args.cache_dir.expanduser().resolve() / "huggingface"))
    if args.experimental_rocm:
        os.environ["CLEF_EXPERIMENTAL_ROCM"] = "1"
    for field, env in [("batch_size", "CLEF_BATCH_MAX_SIZE"), ("max_length", "CLEF_MAX_LENGTH"), ("max_image_length", "CLEF_MAX_IMAGE_LENGTH"),
                       ("context_reserve_mib", "CLEF_CONTEXT_RESERVE_MIB"),
                       ("prefill_initial_chunk_tokens", "CLEF_PREFILL_INITIAL_CHUNK_TOKENS"),
                       ("prefill_chunk_tokens", "CLEF_PREFILL_CHUNK_TOKENS"),
                       ('prefill_policy', 'CLEF_PREFILL_POLICY'),
                       ('prefill_max_chunk_tokens', 'CLEF_PREFILL_MAX_CHUNK_TOKENS'),
                       ('prefill_target_chunk_tokens', 'CLEF_PREFILL_TARGET_CHUNK_TOKENS'),
                       ("linear_prefill_backend", "CLEF_LINEAR_PREFILL_BACKEND")]:
        if getattr(args, field) is not None:
            os.environ[env] = str(getattr(args, field))
    if args.max_length is not None:
        os.environ["CLEF_CONTEXT_LIMIT_MODE"] = "fixed"
    if args.image_pooling:
        os.environ["CLEF_IMAGE_POOLING"] = "1"
    if args.no_prefix_cache:
        os.environ["CLEF_PREFIX_CACHE"] = "0"
    if args.no_warmup:
        os.environ["CLEF_WARMUP"] = "0"
    if args.allow_slow_kernels:
        os.environ["CLEF_REQUIRE_FAST_LINEAR_ATTENTION"] = "0"
        os.environ.setdefault("CLEF_LINEAR_PREFILL_BACKEND", "auto")
    try:
        if args.action == "download":
            data.mkdir(parents=True, exist_ok=True)
            download_source(data, config, args.offline)
            return
        runtime_backend = runtime_backend_from_build()
        prepare_runtime(runtime_backend)
        select_gpu(args.gpu, runtime_backend=runtime_backend)
        install_prebuilt_launchers(runtime_backend)
        strategy = detect_strategy(profile)
        effective = configure_strategy(strategy)
        logging.getLogger("clef").info("Selected strategy: %s", effective)
        if args.action == "inspect":
            print(json.dumps(effective, indent=2))
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
    # Serving is offline, so the download token is no longer needed.
    os.environ.pop("HF_TOKEN", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Keep compiled kernels separate across architectures and dtype.
    architecture = strategy.capability if strategy.runtime_backend == "rocm" else f"sm{strategy.capability.replace('.', '')}"
    cache = data / "cache" / f"{architecture}-{strategy.compute_dtype}"
    for name in ["triton", "torch-kernels"]:
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(cache / "torch-kernels"))
    import uvicorn
    uvicorn.run("clef_service.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
