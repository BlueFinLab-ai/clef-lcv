"""Bounded FIFO admission with one model worker; queued jobs hold no GPU tensors."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import time

from starlette.responses import JSONResponse


class QueueError(Exception):
    def __init__(self, status, code, detail):
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail

    def response(self):
        return JSONResponse({"detail": self.detail, "error": self.code},
                            status_code=self.status,
                            headers={"Retry-After": "1"} if self.status in {429, 503} else None)


class PayloadTicket:
    def __init__(self, budget):
        self.budget, self.bytes, self.refs = budget, 0, 1

    def grow(self, count):
        if self.bytes + count > self.budget.request_limit:
            raise QueueError(413, "request_body_too_large", "Request body exceeds the configured wire-size limit")
        if self.budget.bytes + count > self.budget.byte_limit:
            self.budget.rejected += 1
            raise QueueError(429, "queue_full", "Request queue payload budget is full; retry later")
        self.bytes += count
        self.budget.bytes += count

    def retain(self):
        self.refs += 1

    def release(self):
        self.refs -= 1
        if self.refs == 0:
            self.budget.bytes -= self.bytes
            self.budget.count -= 1


class PayloadBudget:
    # All mutations occur on the ASGI event loop, including worker completion.
    def __init__(self, count_limit, byte_limit, request_limit):
        self.count_limit, self.byte_limit, self.request_limit = count_limit, byte_limit, request_limit
        self.count = self.bytes = self.rejected = 0
        self.accepting = True

    def open(self):
        if not self.accepting:
            raise QueueError(503, "queue_shutdown", "Service is shutting down; retry later")
        if self.count >= self.count_limit:
            self.rejected += 1
            raise QueueError(429, "queue_full", "Request queue is full; retry later")
        self.count += 1
        return PayloadTicket(self)


class QueueAdmissionMiddleware:
    """Bound wire payloads and admission count before FastAPI parses JSON."""
    def __init__(self, app, budget):
        self.app, self.budget = app, budget

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/v1/systemone":
            return await self.app(scope, receive, send)
        ticket = None
        try:
            ticket = self.budget.open()
            # Check actual received bytes too; Content-Length is not trusted.
            headers = dict(scope.get("headers", []))
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    size = int(length)
                except ValueError:
                    size = 0  # The HTTP server handles malformed framing.
                if size > self.budget.request_limit:
                    raise QueueError(413, "request_body_too_large", "Request body exceeds the configured wire-size limit")
            chunks = []
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                ticket.grow(len(chunk))
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            chunks.clear()
            scope["clef_payload_ticket"] = ticket
            delivered = False

            async def replay():
                nonlocal delivered, body
                if not delivered:
                    delivered = True
                    value, body = body, b""
                    return {"type": "http.request", "body": value, "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        except QueueError as exc:
            await exc.response()(scope, receive, send)
        finally:
            if ticket is not None:
                ticket.release()


@dataclass(eq=False)
class Job:
    payload: object
    future: asyncio.Future
    submitted: float
    ticket: PayloadTicket | None
    started: float | None = None


class DecisionQueue:
    def __init__(self, handler, *, max_waiting=64, wait_seconds=300,
                 payload_mib=256, request_mib=224):
        if max_waiting < 1 or wait_seconds <= 0 or payload_mib <= 0 or request_mib <= 0:
            raise ValueError("Queue limits must be positive")
        self.handler, self.max_waiting, self.wait_seconds = handler, max_waiting, wait_seconds
        self.budget = PayloadBudget(max_waiting + 1, int(payload_mib * 2**20), int(request_mib * 2**20))
        self.pending = deque()
        self.pending_bytes = 0
        self.active = None
        self.completed = self.failed = self.cancelled = self.timed_out = 0
        self.ready = asyncio.Event()
        self.worker = self.executor = None
        self.closing = False

    @classmethod
    def from_env(cls, handler):
        return cls(handler, max_waiting=int(os.environ.get("CLEF_QUEUE_MAX_WAITING", "64")),
                   wait_seconds=float(os.environ.get("CLEF_QUEUE_WAIT_SECONDS", "300")),
                   payload_mib=float(os.environ.get("CLEF_QUEUE_PAYLOAD_MIB", "256")),
                   request_mib=float(os.environ.get("CLEF_REQUEST_BODY_MIB", "224")))

    async def start(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-gpu")
        self.worker = asyncio.create_task(self._work())

    async def close(self):
        self.closing = True
        self.budget.accepting = False
        while self.pending:
            job = self.pending.popleft()
            self.pending_bytes -= job.ticket.bytes if job.ticket else 0
            if not job.future.done():
                job.future.set_exception(QueueError(503, "queue_shutdown", "Service is shutting down; retry later"))
            self._release(job)
        self.ready.set()
        if self.worker is not None:
            await self.worker  # An active CUDA call finishes before teardown.
        if self.executor is not None:
            self.executor.shutdown(wait=True)

    def stats(self):
        return {"enabled": True, "policy": "fifo", "gpu_workers": 1,
                "active_requests": int(self.active is not None), "waiting_requests": len(self.pending),
                "max_waiting_requests": self.max_waiting, "wait_timeout_seconds": self.wait_seconds,
                "admitted_requests": self.budget.count, "admitted_payload_mib": round(self.budget.bytes / 2**20, 3),
                "payload_limit_mib": self.budget.byte_limit / 2**20,
                "request_body_limit_mib": self.budget.request_limit / 2**20,
                "completed": self.completed, "failed": self.failed, "cancelled": self.cancelled,
                "timed_out": self.timed_out, "rejected": self.budget.rejected,
                "accepting": not self.closing}

    def _release(self, job):
        job.payload = None
        if job.ticket is not None:
            job.ticket.release()
            job.ticket = None

    def _cancel(self, job):
        if job.future.done():
            return
        job.future.cancel()
        if job.started is None:
            self.pending.remove(job)
            self.pending_bytes -= job.ticket.bytes if job.ticket else 0
            self._release(job)

    async def run(self, payload, request=None):
        if self.closing or self.worker is None:
            raise QueueError(503, "queue_shutdown", "Service is unavailable; retry later")
        if len(self.pending) >= self.max_waiting:
            self.budget.rejected += 1
            raise QueueError(429, "queue_full", "Request queue is full; retry later")
        ticket = request.scope.get("clef_payload_ticket") if request is not None else None
        if ticket is not None:
            ticket.retain()  # Preserve admission while a disconnected active job finishes.
        job = Job(payload, asyncio.get_running_loop().create_future(), time.perf_counter(), ticket)
        self.pending.append(job)
        self.pending_bytes += ticket.bytes if ticket else 0
        self.ready.set()
        try:
            while not job.future.done():
                await asyncio.wait({job.future}, timeout=.1)
                if job.future.done():
                    break
                if request is not None and await request.is_disconnected():
                    self.cancelled += 1
                    self._cancel(job)
                    return JSONResponse({"detail": "Client disconnected", "error": "client_disconnected"}, status_code=499)
                if job.started is None and time.perf_counter() - job.submitted >= self.wait_seconds:
                    self.timed_out += 1
                    self._cancel(job)
                    raise QueueError(504, "queue_timeout", "Request exceeded the queue wait deadline; retry later")
            result = job.future.result()
        except asyncio.CancelledError:
            if not job.future.done():
                self.cancelled += 1
                self._cancel(job)
            raise
        wait_ms = round((job.started - job.submitted) * 1000, 1)
        total_ms = round((time.perf_counter() - job.submitted) * 1000, 1)
        if isinstance(result, dict) and "usage" in result:
            result["usage"].update(queue_wait_ms=wait_ms, server_total_ms=total_ms)
        elif isinstance(result, JSONResponse):
            result.headers["X-Clef-Queue-Wait-Ms"] = str(wait_ms)
            result.headers["X-Clef-Server-Time-Ms"] = str(total_ms)
        return result

    async def _work(self):
        while True:
            await self.ready.wait()
            self.ready.clear()
            if not self.pending:
                if self.closing:
                    return
                continue
            job = self.pending.popleft()
            self.pending_bytes -= job.ticket.bytes if job.ticket else 0
            if time.perf_counter() - job.submitted >= self.wait_seconds:
                self.timed_out += 1
                job.future.set_exception(QueueError(504, "queue_timeout", "Request exceeded the queue wait deadline; retry later"))
                self._release(job)
            else:
                self.active = job
                job.started = time.perf_counter()
                try:
                    result = await asyncio.get_running_loop().run_in_executor(self.executor, self.handler, job.payload)
                except Exception as exc:
                    self.failed += 1
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    self.completed += 1
                    if not job.future.done():
                        job.future.set_result(result)
                finally:
                    self.active = None
                    self._release(job)
            if self.pending or self.closing:
                self.ready.set()
