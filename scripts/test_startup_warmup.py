"""CPU checks of warmup gating, worker reuse, failure cleanup and opt-out."""
import asyncio
import base64
from io import BytesIO
from pathlib import Path
import sys
import threading

from PIL import Image
from starlette.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from clef_service.startup_warmup import warmup_on_worker, warmup_requests
from request_queue import DecisionQueue


async def check():
    for model in ('clef-flash', 'clef'):
        requests = warmup_requests(model, image_pooling=True)
        assert {q['type'] for _, r in requests for q in r['questions'].values()} == {'choice', 'noul', 'score'}
        for _, request in requests:
            assert request['model'] == model and not request['input_cache'] and not request['prefix_cache']
        image_request = requests[1][1]
        assert image_request['image_pooling']
        image = Image.open(BytesIO(base64.b64decode(image_request['images'][0].split(',', 1)[1])))
        assert image.size == (256, 256) and image.getpixel((128, 128)) == (255, 0, 0)

    gate, entered = threading.Event(), threading.Event()
    threads = []
    def handler(request):
        threads.append(threading.get_ident())
        if request.get('model'):
            entered.set()
            assert gate.wait(3)
        return {'answers': {name: {'type': 'noul', 'noul': 0.0} for name in request['questions']},
                'usage': {'input_tokens': 42}}
    queue = DecisionQueue(handler)
    await queue.start()
    try:
        finished = []
        def finish(): finished.append(threading.get_ident())
        warm = asyncio.create_task(warmup_on_worker(queue, handler, 'clef-flash', finish=finish))
        for _ in range(300):
            if entered.is_set(): break
            await asyncio.sleep(.005)
        assert entered.is_set() and not warm.done(), 'Readiness must await real inference'
        assert queue.stats()['completed'] == 0
        gate.set()
        status = await warm
        assert status['status'] == 'complete' and len(status['requests']) == 2
        assert queue.stats()['completed'] == 0 and not queue.pending
        await queue.run({'questions': {'actual_user_request': {}}})
        assert len(set(threads + finished)) == 1, 'Warmup must use the actual serving worker'
        assert queue.stats()['completed'] == 1

        for response in (JSONResponse({'detail': 'capacity'}, status_code=413),
                         JSONResponse({'detail': 'memory'}, status_code=503),
                         {'answers': {}},
                         {'answers': {name: {'probability': float('nan')} for name in warmup_requests('clef')[0][1]['questions']}}):
            before = len(finished)
            try:
                await warmup_on_worker(queue, lambda _: response, 'clef', finish=finish)
            except (RuntimeError, ValueError): pass
            else: raise AssertionError('Broken warmup accepted')
            assert len(finished) == before + 1

        def fails(_): raise RuntimeError('GPU inference failed')
        before = len(finished)
        try: await warmup_on_worker(queue, fails, 'clef', finish=finish)
        except RuntimeError: pass
        else: raise AssertionError('Inference exception lost')
        assert len(finished) == before + 1
        # Opt-out does no inference or GPU cleanup, even with no started worker.
        skipped = await warmup_on_worker(DecisionQueue(fails), fails, 'clef', enabled=False,
                                       finish=lambda: (_ for _ in ()).throw(AssertionError('Touched GPU')))
        assert skipped['status'] == 'skipped' and skipped['requests'] == []
    finally:
        gate.set()
        await queue.close()


asyncio.run(check())
print('PASS: warmup gate, serving thread, counters, failure cleanup, finite answers, opt-out')
