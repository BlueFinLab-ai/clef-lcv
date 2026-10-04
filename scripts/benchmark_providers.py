"""Compare the public email fixture on Clef, TypeSafe JEV and other SystemOne-compatible services.

Install httpx for this optional client tool. Secrets come from environment or an
owner-only key file; they are never written to results. No automatic retries:
errors count, rather than silently changing the measured workload.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from benchmark_emails import DEFAULT_DATASET, load_dataset, make_request, summarize


def secret(env_name, filename=None):
    value = Path(filename).read_text().strip() if filename else os.environ.get(env_name, "").strip()
    if not value:
        raise ValueError(f"Set {env_name} or provide --key-file")
    return value


def native_request(row, provider, model, categories, guide, caching):
    body = make_request(row, model, categories, guide, caching)
    if provider in {"jev", "systemone"}:
        body = {k: body[k] for k in ("model", "state", "questions")}
        body["state"] = {"classification_guide": guide, **body["state"]}
    return body


def validate_choice(answer, categories, rounded_probabilities=False):
    category = answer["answers"]["category"]
    probabilities = category["probabilities"]
    if category["choice"] not in categories or set(probabilities) != set(categories):
        raise ValueError("Invalid response categories")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1 for v in probabilities.values()):
        raise ValueError("Invalid probabilities")
    # Observed JEV responses round each of twelve buckets to two decimals.
    # Account for that accumulated rounding only when every value has that
    # precision. Keep the original probabilities; do not normalize them.
    tolerance = .005
    if rounded_probabilities and all(abs(v - round(v, 2)) < 1e-8 for v in probabilities.values()):
        tolerance = .005 * len(probabilities) + 1e-8
    if abs(sum(probabilities.values()) - 1) > tolerance:
        raise ValueError("Invalid probability sum")
    if probabilities[category["choice"]] < max(probabilities.values()) - .00001:
        raise ValueError("Choice is not the probability maximum")
    return category["choice"], probabilities


def invoke(client, args, row, categories, guide):
    result = {"id": row["id"], "reference_category": row["reference_category"]}
    body = native_request(row, args.provider, args.model, categories, guide, args.cache == "on")
    # Wire-body hash records exactly what was sent, without retaining credentials.
    content = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    result["request_sha256"] = hashlib.sha256(content).hexdigest()
    started = time.perf_counter()
    try:
        response = client.post(args.url + "/v1/systemone", content=content)
        result["status"] = response.status_code
        response.raise_for_status()
        answer = response.json()
        # Preserve the bounded category result before validation for diagnosis.
        result["native_answer"] = answer.get("answers", {}).get("category")
        choice, probabilities = validate_choice(answer, categories, rounded_probabilities=args.provider == "jev")
        if abs(sum(probabilities.values()) - 1) > .005:
            result["probability_rounding_warning"] = {"reported_sum": sum(probabilities.values())}
        result.update(choice=choice, probabilities=probabilities, usage=answer.get("usage", {}),
                      provider_usage=answer.get("usage", {}), response_model=answer.get("model"))
        if result.get("response_model") != args.model:
            raise ValueError("Response model differs from requested model")
    except Exception as error:
        # Do not export exception strings, arbitrary server text or request headers.
        result["error"] = type(error).__name__
    result["client_wall_ms"] = (time.perf_counter() - started) * 1000
    return result


def trial_summary(rows, wall, categories):
    summary = summarize(rows, wall, categories)
    good = [r for r in rows if "error" not in r]
    summary["total_input_tokens"] = sum(r["provider_usage"].get("input_tokens", 0) for r in good)
    summary["probability_rounding_warnings"] = sum("probability_rounding_warning" in r for r in good)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("clef", "jev", "systemone"))
    parser.add_argument("--url", help="Base URL of a local Clef or SystemOne-compatible service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-file", type=Path, help="Optional owner-only key file; otherwise JEV_API_KEY")
    parser.add_argument("--cache", choices=("on", "off"), default="on", help="Clef cache flags only; other services control their own caching")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trials", default="1,4,4", help="Ordered client concurrencies, one 100-email pass each")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, categories, guide, manifest = load_dataset(args.dataset)
    concurrencies = [int(v) for v in args.trials.split(",")]
    if any(v < 1 for v in concurrencies) or args.timeout <= 0:
        parser.error("Positive concurrencies and timeout required")
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            parser.error("Invalid record limit")
        rows = rows[:args.limit]
    if args.output.exists():
        parser.error("Output already exists")
    headers = {"Content-Type": "application/json"}
    route = args.provider
    if args.provider == "jev":
        args.url = "https://api.typesafe.ai"
        headers["Authorization"] = "Bearer " + secret("JEV_API_KEY", args.key_file)
    elif not args.url:
        parser.error("--url required for local Clef/SystemOne")
    from urllib.parse import urlsplit
    parsed = urlsplit(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("Invalid base URL")
    args.url = args.url.rstrip("/")
    report = {
        "schema_version": 1, "provider": args.provider, "route": route, "model": args.model,
        "started_utc": datetime.now(timezone.utc).isoformat(), "completed": False,
        "dataset": manifest, "sample_count": len(rows),
        "clef_cache_flags": args.cache if args.provider == "clef" else None,
        "connection_pool": "persistent httpx client, HTTP/1.1", "automatic_retries": False,
        "methodology": "Each email is an independent request. Same guide, email fields and category definitions; reference labels never sent. Clef receives its context/cache extensions; JEV receives guide in documented state. All services return a choice and probability distribution. Client time includes network, queueing, inference and response; GPU kernel time is not measured. Hosted caching is provider-controlled. No server cache reset: passes inherit existing state. Authored synthetic labels are not adjudicated accuracy; work and other have no positives.",
        "trials": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(args.output)
    import httpx
    with httpx.Client(headers=headers, timeout=args.timeout, follow_redirects=False,
                      limits=httpx.Limits(max_connections=max(concurrencies), max_keepalive_connections=max(concurrencies))) as client:
        if args.provider in {"clef", "systemone"}:
            health = client.get(args.url + "/health").json()
            report["service_before"] = {k: health.get(k) for k in ("model", "gpu", "quantization", "compute_dtype", "source_revision", "linear_attention", "optimizations", "queue")}
            if args.provider == "systemone":
                report["service_before"].update({k: health.get(k) for k in ("models", "device", "encoding", "max_len", "head_max_len", "adapter", "max_input_tokens", "quantization", "compute_dtype", "queue_policy")})
                report["methodology"] += " Generic SystemOne receives the guide in state and no Clef extensions. Local adapter configuration is recorded separately."
        save()
        for number, concurrency in enumerate(concurrencies, 1):
            started_utc = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(invoke, client, args, row, categories, guide) for row in rows]
                for future in as_completed(futures):
                    results.append(future.result())
                    if len(results) % 25 == 0:
                        print(f"{args.model} trial {number} c={concurrency}: {len(results)}/{len(rows)}", flush=True)
            wall = time.perf_counter() - started
            results.sort(key=lambda r: r["id"])
            summary = trial_summary(results, wall, categories)
            report["trials"].append({"number": number, "concurrency": concurrency, "started_utc": started_utc,
                                     "completed_utc": datetime.now(timezone.utc).isoformat(), "summary": summary, "rows": results})
            save()
            print(f"{args.model} c={concurrency}: {wall:.3f}s, {summary['reference_matches']}/{summary['successful']} reference agreement, {summary['errors']} errors", flush=True)
        report.update(completed=True, completed_utc=datetime.now(timezone.utc).isoformat())
        save()
    if any(t["summary"]["errors"] for t in report["trials"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
