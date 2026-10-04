"""Exercise real worker serialization, bounded ASGI admission and cancellation."""
import asyncio
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse
from request_queue import DecisionQueue, QueueAdmissionMiddleware, QueueError


async def until(predicate):
    deadline = time.perf_counter() + 3
    while not predicate():
        assert time.perf_counter() < deadline, 'Test condition did not become ready'
        await asyncio.sleep(.005)


async def serialization():
    order, threads = [], set()
    active = peak = 0
    def handler(value):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        threads.add(threading.get_ident()); order.append(value)
        time.sleep(.005)
        active -= 1
        return {'value': value, 'usage': {'latency_ms': 5}}
    q = DecisionQueue(handler, max_waiting=16)
    await q.start()
    try:
        rows = await asyncio.gather(*(q.run(i) for i in range(12)))
        assert order == list(range(12)) and peak == 1 and len(threads) == 1
        assert [r['value'] for r in rows] == order
        assert rows[-1]['usage']['queue_wait_ms'] > 20
        assert q.stats()['completed'] == 12
    finally:
        await q.close()


async def bounds_and_errors():
    gate, entered = threading.Event(), threading.Event()
    calls = []
    def handler(value):
        calls.append(value)
        if value == 'active':
            entered.set(); assert gate.wait(3)
        if value == 'error':
            raise ValueError('bad fixture')
        if value == 'large':
            return JSONResponse({'detail': 'too many tokens', 'max_input_tokens': 123}, status_code=413)
        return {'value': value, 'usage': {}}
    q = DecisionQueue(handler, max_waiting=2)
    await q.start()
    try:
        a = asyncio.create_task(q.run('active')); await until(entered.is_set)
        b = asyncio.create_task(q.run('cancel')); c = asyncio.create_task(q.run('next'))
        await until(lambda: len(q.pending) == 2)
        try:
            await q.run('overflow')
        except QueueError as e:
            assert e.status == 429 and e.response().headers['Retry-After'] == '1'
        else:
            raise AssertionError('Queue overflow accepted')
        b.cancel()
        try: await b
        except asyncio.CancelledError: pass
        gate.set(); await asyncio.gather(a, c)
        assert calls == ['active', 'next'] and q.stats()['cancelled'] == 1
        try: await q.run('error')
        except ValueError: pass
        else: raise AssertionError('Worker exception lost')
        response = await q.run('large')
        assert response.status_code == 413 and 'X-Clef-Queue-Wait-Ms' in response.headers
        assert (await q.run('recovery'))['value'] == 'recovery'
    finally:
        gate.set(); await q.close()


async def deadlines_and_shutdown():
    gate, entered = threading.Event(), threading.Event()
    calls = []
    def handler(value):
        calls.append(value); entered.set(); assert gate.wait(3)
        return {'usage': {}}
    q = DecisionQueue(handler, wait_seconds=.03)
    await q.start()
    try:
        a = asyncio.create_task(q.run('active')); await until(entered.is_set)
        try: await q.run('expires')
        except QueueError as e: assert e.status == 504 and e.code == 'queue_timeout'
        else: raise AssertionError('Deadline not enforced')
        assert q.stats()['active_requests'] == 1 and not q.pending
        gate.set(); await a
        assert calls == ['active']
    finally:
        gate.set(); await q.close()
    # Shutdown rejects pending work and waits for the one active call.
    gate.clear(); entered.clear()
    q = DecisionQueue(handler)
    await q.start()
    a = asyncio.create_task(q.run('active')); await until(entered.is_set)
    b = asyncio.create_task(q.run('pending')); await until(lambda: len(q.pending) == 1)
    closing = asyncio.create_task(q.close())
    try: await b
    except QueueError as e: assert e.status == 503
    else: raise AssertionError('Shutdown pending job executed')
    assert not closing.done()
    gate.set(); await a; await closing
    assert q.budget.count == q.budget.bytes == 0


async def asgi_admission_and_disconnect():
    gate, entered = threading.Event(), threading.Event()
    calls = []
    def handler(value):
        calls.append(value['value'])
        if value['value'] == 'hold':
            entered.set(); assert gate.wait(3)
        return {'value': value['value'], 'usage': {}}
    q = DecisionQueue(handler, max_waiting=2, payload_mib=700/2**20, request_mib=512/2**20)
    app = FastAPI()
    app.add_middleware(QueueAdmissionMiddleware, budget=q.budget)
    @app.post('/v1/systemone')
    async def post(payload: dict, request: Request):
        try: return await q.run(payload, request)
        except QueueError as e: return e.response()
    @app.get('/health')
    async def health(): return q.stats()
    await q.start()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test',
                                 headers={'Content-Type':'application/json'}) as client:
        try:
            # Reject oversized known and streamed bodies before JSON/model work.
            r = await client.post('/v1/systemone', content=b'x'*513)
            assert r.status_code == 413 and q.budget.count == 0
            async def chunks():
                yield b'x'*300
                yield b'x'*300
            r = await client.post('/v1/systemone', content=chunks())
            assert r.status_code == 413 and q.budget.bytes == 0
            r = await client.post('/v1/systemone', content=b'{')
            assert r.status_code == 422 and q.budget.count == q.budget.bytes == 0
            body = json.dumps({'value':'hold', 'padding':'x'*250}).encode()
            a = asyncio.create_task(client.post('/v1/systemone', content=body))
            await until(entered.is_set)
            b = asyncio.create_task(client.post('/v1/systemone', content=json.dumps({'value':'next', 'padding':'x'*250})))
            await until(lambda: len(q.pending) == 1)
            r = await client.post('/v1/systemone', content=body)
            assert r.status_code == 429 and r.json()['error'] == 'queue_full'
            assert (await client.get('/health')).json()['waiting_requests'] == 1
            b.cancel()
            try: await b
            except asyncio.CancelledError: pass
            await until(lambda: not q.pending)
            assert q.budget.count == 1
            # Replay the body once, then report disconnect: pending inference
            # must be removed without decoding/preprocessing/GPU execution.
            sent = []
            messages = [{'type':'http.request','body':b'{"value":"disconnect"}'}, {'type':'http.disconnect'}]
            async def receive():
                if messages: return messages.pop(0)
                await asyncio.sleep(10)
            async def send(message): sent.append(message)
            scope = {'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
                     'scheme':'http','path':'/v1/systemone','raw_path':b'/v1/systemone','query_string':b'',
                     'headers':[(b'content-type',b'application/json')],'client':('test',1),'server':('test',80)}
            await app(scope, receive, send)
            assert sent[0]['status'] == 499 and not q.pending
            assert q.budget.count == 1
            gate.set(); assert (await a).status_code == 200
            await until(lambda: q.budget.count == 0)
            assert calls == ['hold'] and q.budget.bytes == 0
            # Count bound, independent of byte capacity.
            tickets = [q.budget.open() for _ in range(3)]
            r = await client.post('/v1/systemone', json={'value':'overflow'})
            assert r.status_code == 429
            for ticket in tickets: ticket.release()
        finally:
            gate.set(); await q.close()
    # A disconnected active call must continue to reserve its wire budget.
    gate.clear(); entered.clear()
    q = DecisionQueue(handler)
    await q.start()
    ticket = q.budget.open(); ticket.grow(100)
    class Disconnected:
        scope = {'clef_payload_ticket':ticket}
        async def is_disconnected(self): return True
    try:
        response = await q.run({'value':'hold'}, Disconnected())
        ticket.release()  # ASGI middleware has finished.
        assert response.status_code == 499 and q.budget.count == 1 and q.budget.bytes == 100
        assert q.stats()['active_requests'] == 1
        gate.set(); await until(lambda: q.active is None)
        assert q.budget.count == q.budget.bytes == 0
    finally:
        gate.set(); await q.close()


async def main():
    await serialization()
    await bounds_and_errors()
    await deadlines_and_shutdown()
    await asgi_admission_and_disconnect()
    print('PASS: FIFO, one worker, errors/recovery, queue count/bytes, streamed body caps, deadlines, disconnects and shutdown')


if __name__ == '__main__':
    asyncio.run(main())
