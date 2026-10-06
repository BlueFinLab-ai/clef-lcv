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
    delivered: bool = False
    preparation: object = None
    interleaves: int = 0


class DecisionQueue:
    def __init__(self, handler, *, max_waiting=64, wait_seconds=300,
                 payload_mib=256, request_mib=224, max_batch_size=1, batch_handler=None, batch_key=None,
                 prepare_handler=None):
        if max_waiting < 1 or wait_seconds <= 0 or payload_mib <= 0 or request_mib <= 0:
            raise ValueError("Queue limits must be positive")
        if max_batch_size not in {1, 2, 4}:
            raise ValueError("Batch size must be 1, 2 or 4")
        if max_batch_size > 1 and (batch_handler is None or batch_key is None):
            raise ValueError("Batching requires a batch handler and compatibility key")
        self.max_batch_size, self.batch_handler, self.batch_key = max_batch_size, batch_handler, batch_key
        self.handler, self.max_waiting, self.wait_seconds = handler, max_waiting, wait_seconds
        self.prepare_handler = prepare_handler
        self.prepare_executor = self.prepare_slot = None
        self.prepared_requests = self.preparation_skips = self.preparation_failures = 0
        self.interleaved_requests = 0
        self.nesting = False
        self.loop = None
        self.budget = PayloadBudget(max_waiting + max_batch_size, int(payload_mib * 2**20), int(request_mib * 2**20))
        self.pending = deque()
        self.pending_bytes = 0
        self.active = None
        self.completed = self.failed = self.cancelled = self.timed_out = 0
        self.batch_dispatches = self.grouped_requests = 0
        self.ready = asyncio.Event()
        self.worker = self.executor = None
        self.closing = False

    @classmethod
    def from_env(cls, handler, *, batch_handler=None, batch_key=None, prepare_handler=None):
        return cls(handler, max_waiting=int(os.environ.get("CLEF_QUEUE_MAX_WAITING", "64")),
                   wait_seconds=float(os.environ.get("CLEF_QUEUE_WAIT_SECONDS", "300")),
                   payload_mib=float(os.environ.get("CLEF_QUEUE_PAYLOAD_MIB", "256")),
                   request_mib=float(os.environ.get("CLEF_REQUEST_BODY_MIB", "224")),
                   max_batch_size=int(os.environ.get("CLEF_BATCH_MAX_SIZE", "1")),
                   batch_handler=batch_handler, batch_key=batch_key,
                   prepare_handler=prepare_handler if (os.environ.get('CLEF_CPU_PREPARE_OVERLAP','0') == '1'
                       or os.environ.get('CLEF_CHUNK_INTERLEAVE','0') == '1') else None)

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-gpu")
        if self.prepare_handler is not None:
            self.prepare_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='clef-cpu-prepare')
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
        if self.prepare_executor is not None:
            self.prepare_executor.shutdown(wait=True)
            await asyncio.sleep(0)  # Deliver retained-ticket completion callbacks.

    def stats(self):
        return {"enabled": True, "policy": "opportunistic_fifo" if self.max_batch_size > 1 else "fifo", "gpu_workers": 1,
                "active_requests": len(self.active) if self.active is not None else 0,
                "max_batch_size": self.max_batch_size, "collection_delay_ms": 0,
                "batch_dispatches": self.batch_dispatches, "grouped_requests": self.grouped_requests, "waiting_requests": len(self.pending),
                "max_waiting_requests": self.max_waiting, "wait_timeout_seconds": self.wait_seconds,
                "admitted_requests": self.budget.count, "admitted_payload_mib": round(self.budget.bytes / 2**20, 3),
                "payload_limit_mib": self.budget.byte_limit / 2**20,
                "request_body_limit_mib": self.budget.request_limit / 2**20,
                "completed": self.completed, "failed": self.failed, "cancelled": self.cancelled,
                "timed_out": self.timed_out, "rejected": self.budget.rejected,
                "cpu_preparation": {"enabled":self.prepare_handler is not None, "workers":1 if self.prepare_executor is not None else 0,
                    "queued_slots":int(self.prepare_slot is not None), "max_queued_slots":1,
                    "prepared_requests":self.prepared_requests,"skipped_requests":self.preparation_skips,
                    "failed_requests":self.preparation_failures},
                "interleaved_requests":self.interleaved_requests,
                "accepting": not self.closing}

    def _schedule_prepare(self):
        if self.closing or self.active is None or self.prepare_executor is None or self.prepare_slot is not None or not self.pending:
            return
        job = self.pending[0]
        if job.future.done() or job.started is not None:
            return
        self.prepare_slot = job
        ticket = job.ticket
        if ticket is not None: ticket.retain()
        loop = asyncio.get_running_loop()
        job.preparation = self.prepare_executor.submit(self.prepare_handler, job.payload)
        def complete(future):
            def finish():
                try:
                    if future.result() is None: self.preparation_skips += 1
                    else: self.prepared_requests += 1
                except Exception:
                    self.preparation_failures += 1
                finally:
                    if ticket is not None: ticket.release()
            loop.call_soon_threadsafe(finish)
        job.preparation.add_done_callback(complete)

    def peek_prepared(self):
        job=self.prepare_slot
        if job is None or job.preparation is None or not job.preparation.done():return None
        try:return job.preparation.result()
        except Exception:return None

    async def _take_interleaved(self,prepared,max_per_parent):
        if self.closing or self.nesting or not self.active or len(self.active)!=1 or not self.pending:return None
        parent=self.active[0]
        if parent.interleaves>=max_per_parent:return None
        job=self.pending[0]
        if (job.future.done() or time.perf_counter()-job.submitted>=self.wait_seconds
                or job.preparation is None or not job.preparation.done()):return None
        try:
            if job.preparation.result() is not prepared:return None
        except Exception:return None
        job=self._take()
        if job is None:return None
        job.started=time.perf_counter();parent.interleaves+=1
        self.active=[parent,job];self.nesting=True
        return job

    def run_interleaved(self,prepared,max_per_parent=4):
        """Called only at a safe chunk boundary on the existing GPU worker."""
        job=asyncio.run_coroutine_threadsafe(self._take_interleaved(prepared,max_per_parent),self.loop).result()
        if job is None:return False
        try:
            result=self.handler(prepared)
            if isinstance(result,dict):result['usage']['chunk_interleaved']=True
        except Exception as exc:result=exc
        async def finish():
            self._deliver(job,result);self._release(job)
            self.active=self.active[:1];self.nesting=False
            self.interleaved_requests+=1
            self._schedule_prepare()
        asyncio.run_coroutine_threadsafe(finish(),self.loop).result()
        return True

    def _release(self, job):
        if self.prepare_slot is job: self.prepare_slot = None
        job.preparation = None
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
        self._schedule_prepare()
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

    def _deliver(self, job, result):
        # This callback always runs on the ASGI loop. An active cancellation
        # suppresses delivery but still counts completed GPU work once.
        if job.delivered:
            return
        job.delivered = True
        if isinstance(result, Exception):
            self.failed += 1
            if not job.future.done():
                job.future.set_exception(result)
        else:
            self.completed += 1
            if not job.future.done():
                job.future.set_result(result)

    def _take(self):
        while self.pending:
            job = self.pending.popleft()
            self.pending_bytes -= job.ticket.bytes if job.ticket else 0
            if job.future.done():
                self._release(job)
            elif time.perf_counter() - job.submitted >= self.wait_seconds:
                self.timed_out += 1
                job.future.set_exception(QueueError(504, "queue_timeout", "Request exceeded the queue wait deadline; retry later"))
                self._release(job)
            else:
                if self.prepare_slot is job: self.prepare_slot = None
                return job
        return None

    async def _work(self):
        while True:
            await self.ready.wait()
            self.ready.clear()
            first = self._take()
            if first is None:
                if self.closing:
                    return
                continue
            jobs = [first]
            # No collection timer: only already waiting consecutive requests
            # can join. Never bypass an older incompatible/image request.
            key = self.batch_key(first.payload) if self.max_batch_size > 1 else None
            while key is not None and len(jobs) < self.max_batch_size and self.pending:
                candidate = self.pending[0]
                if candidate.future.done() or time.perf_counter()-candidate.submitted >= self.wait_seconds:
                    # Let _take remove this cancelled/expired head without
                    # accidentally consuming the following incompatible job.
                    self.pending.popleft()
                    self.pending_bytes -= candidate.ticket.bytes if candidate.ticket else 0
                    if not candidate.future.done():
                        self.timed_out += 1
                        candidate.future.set_exception(QueueError(504, "queue_timeout", "Request exceeded the queue wait deadline; retry later"))
                    self._release(candidate)
                    continue
                if self.batch_key(candidate.payload) != key:
                    break
                jobs.append(self._take())
            self.active = jobs
            for job in jobs:
                job.started = time.perf_counter()
            self._schedule_prepare()
            loop = asyncio.get_running_loop()
            if len(jobs) > 1:
                self.batch_dispatches += 1
                self.grouped_requests += len(jobs)

            def emit(index, result):
                loop.call_soon_threadsafe(self._deliver, jobs[index], result)

            def execute():
                def payload(job):
                    if job.preparation is None: return job.payload
                    value = job.preparation.result()
                    return job.payload if value is None else value
                if len(jobs) == 1:
                    try: result = self.handler(payload(jobs[0]))
                    except Exception as exc: result = exc
                    emit(0, result)
                else:
                    # Serial fallbacks emit their result immediately rather
                    # than delaying the first reply until every fallback ends.
                    values, mapping = [], []
                    for i, job in enumerate(jobs):
                        try:
                            values.append(payload(job)); mapping.append(i)
                        except Exception as exc:
                            emit(i, exc)
                    if len(values) == 1:
                        try: result = self.handler(values[0])
                        except Exception as exc: result = exc
                        emit(mapping[0], result)
                    elif values:
                        self.batch_handler(values, lambda i, result: emit(mapping[i], result))

            try:
                await loop.run_in_executor(self.executor, execute)
            except Exception as exc:
                for job in jobs:
                    self._deliver(job, exc)
            finally:
                # call_soon_threadsafe deliveries are queued before executor
                # completion. Retain all wire tickets until worker teardown so
                # decoded payloads still held by it remain budgeted.
                for job in jobs:
                    if not job.delivered:
                        self._deliver(job, RuntimeError("Batch worker did not deliver a result"))
                    self._release(job)
                self.active = None
            if self.pending or self.closing:
                self.ready.set()
