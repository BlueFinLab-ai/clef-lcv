"""Bounded synthetic warmup on the serving GPU worker, before readiness."""
import asyncio
import base64
from io import BytesIO
import json
import logging
import time

from PIL import Image, ImageDraw

log = logging.getLogger("clef.warmup")


def warmup_requests(model_id, *, image_pooling=False):
    """Exercise all typed answers plus the common 256px vision path, uncached."""
    questions = {
        "status": {"type": "choice", "instructions": "What is the service status?",
                   "criteria": {"ready": "Service is working normally.", "offline": "Service is offline."}},
        "urgent": {"type": "noul", "instructions": "Does the service need urgent repair?"},
        "impact": {"type": "score", "instructions": "Rate the service disruption.",
                   "criteria": ["No disruption", "Partial disruption", "Complete outage"]},
    }
    text = {"model": model_id, "state": "Synthetic startup check: the service is working normally.",
            "questions": questions, "prefix_cache": False, "input_cache": False}
    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).ellipse((64, 64, 192, 192), fill="red")
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    image_questions = {
        **questions,
        "color": {"type": "choice", "instructions": "What color is the circle?",
                  "criteria": {"red": "Red", "blue": "Blue"}},
    }
    vision = {**text, "questions": image_questions,
              "images": ["data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")],
              "image_pooling": image_pooling}
    return [("text", text), ("image_256", vision)]


async def warmup_on_worker(queue, handler, model_id, *, enabled=True,
                           image_pooling=False, finish=None):
    """Use the existing executor without admitting/counting synthetic user jobs.

    The HTTP lifespan must await this before yielding. Any inference error stops
    startup; returning a 413/503 response is an error too. Model confidence is
    deliberately not a readiness check.
    """
    if not enabled:
        return {"enabled": False, "status": "skipped", "duration_ms": 0.0, "requests": []}
    if queue.executor is None:
        raise RuntimeError("Start the GPU worker before running warmup")

    def run():
        started = time.perf_counter()
        rows = []
        log.info("Startup warmup started; HTTP readiness waits for text and 256px image inference")
        try:
            for name, payload in warmup_requests(model_id, image_pooling=image_pooling):
                tick = time.perf_counter()
                response = handler(payload)
                if not isinstance(response, dict):
                    raise RuntimeError(f"Warmup {name} returned HTTP {getattr(response, 'status_code', 'error')}")
                # Detect broken/non-finite inference without imposing semantic
                # accuracy or a probability threshold on readiness.
                if set(response.get("answers", {})) != set(payload["questions"]):
                    raise RuntimeError(f"Warmup {name} returned incomplete answers")
                json.dumps(response, allow_nan=False)
                usage = response.get("usage", {})
                row = {"name": name, "duration_ms": round((time.perf_counter() - tick) * 1000, 1),
                       "input_tokens": usage.get("input_tokens"),
                       "peak_allocated_mib": usage.get("peak_allocated_mib")}
                rows.append(row)
                log.info("Startup warmup %s completed in %.1f ms", name, row["duration_ms"])
        finally:
            # Free synthetic request workspace on the same worker even if it
            # failed. Keep compiled kernels/library initialization available.
            if finish is not None:
                finish()
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        log.info("Startup warmup complete in %.1f ms; service can become ready", elapsed)
        return {"enabled": True, "status": "complete", "duration_ms": elapsed, "requests": rows}

    return await asyncio.get_running_loop().run_in_executor(queue.executor, run)
