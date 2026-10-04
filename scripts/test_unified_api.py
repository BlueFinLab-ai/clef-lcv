"""Text/vision/cache/queue smoke for one running unified service."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import time

import httpx
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("url")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
client = httpx.Client(base_url=args.url, timeout=300)
deadline = time.monotonic()+300
while True:
    try:
        health = client.get("/health").raise_for_status().json()
        break
    except (httpx.TransportError, httpx.HTTPStatusError):
        if time.monotonic() > deadline:
            raise
        time.sleep(1)
discovery = client.get("/v1/models").raise_for_status().json()["data"][0]
model = discovery["id"]
assert model in {"clef", "clef-flash"} and health["startup_strategy"]["profile"] in {"full", "flash"}
assert client.get("/").status_code == client.get("/ui/app.js").status_code == 200
assert client.get("/openapi.json").json()["paths"]["/v1/systemone"]
assert health["optimizations"]["chunked_prefill"]["images_enabled"]
questions = {"urgent": {"type": "noul", "instructions": "Does this require immediate action?"},
    "action": {"type": "choice", "instructions": "What action is appropriate?",
        "criteria": {"act": "Act immediately", "wait": "Wait until later"}},
    "impact": {"type": "score", "instructions": "Rate the impact.",
        "criteria": ["None", "Partial disruption", "Complete outage"]}}
request = {"model": model, "context": "Review the current incident.",
    "state": "Checkout is down. Customers cannot place orders. Act immediately.", "questions": questions}
trials = []


def run(name, payload):
    started = time.perf_counter()
    response = client.post("/v1/systemone", json=payload).raise_for_status().json()
    trials.append({"name": name, "wall_seconds": time.perf_counter()-started,
                   "answers": response["answers"], "usage": response["usage"]})
    assert set(response["answers"]) == set(payload["questions"])
    assert response["usage"]["accepted_max_input_tokens"] > 0
    return response


cold, warm = run("text-first", request), run("text-repeat", request)
assert cold["answers"] == warm["answers"]
assert warm["usage"]["prefix_cache"] == "hit"
image = Image.new("RGB", (256, 256), "red")
stream = BytesIO()
image.save(stream, format="PNG")
url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
vision = {"model": model, "state": "Inspect this solid-color picture.", "images": [url],
    "questions": {"color": {"type": "choice", "instructions": "What color is the picture?",
        "criteria": {"red": "Red", "blue": "Blue"}}}}
v1, v2 = run("vision-first", vision), run("vision-repeat", vision)
assert v1["answers"] == v2["answers"] and v2["usage"]["prefix_cache"] == "hit"
assert v1["answers"]["color"]["choice"] == "red"
run("vision-pooled", {**vision, "image_pooling": True})
run("sixteen-images", {**vision, "images": [url]*16,
    "media_kwargs": {"images_kwargs": {"do_resize": False}}})
bad = client.post("/v1/systemone", json={**request, "model": "other"})
assert bad.status_code == 400
bad = client.post("/v1/systemone", json={**vision, "images": [url]*17})
assert bad.status_code == 422
too_large = {**request, "state": "word "*(discovery["max_input_tokens"]+128)}
bad = client.post("/v1/systemone", json=too_large)
assert bad.status_code == 413 and bad.json()["input_tokens"] > bad.json()["max_input_tokens"]
# >8K exercises continuation prefill; turns off caching to check the compute path.
chunked_tested = discovery["max_input_tokens"] > 8500
if chunked_tested:
    run("chunked-text", {**request, "state": "word "*8200+request["state"], "prefix_cache": False})
def queued(i):
    payload = {**request, "state": request["state"]+f" Incident {i}."}
    return client.post("/v1/systemone", json=payload).raise_for_status().json()
with ThreadPoolExecutor(max_workers=4) as pool:
    queued_results = list(pool.map(queued, range(4)))
assert all(r["usage"]["queue_wait_ms"] >= 0 for r in queued_results)
assert any(r["usage"]["queue_wait_ms"] > 0 for r in queued_results)
health = client.get("/health").raise_for_status().json()
assert health["queue"]["waiting_requests"] == 0
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"health": health, "discovery": discovery, "trials": trials,
    "queued_usage": [r["usage"] for r in queued_results], "chunked_tested": chunked_tested, "passed": True}, indent=2) + "\n")
print(f"PASS {model}: text, image, repeats, pooling, 16 images, chunked prefill tested={chunked_tested}, four queued calls, errors, UI/discovery")
