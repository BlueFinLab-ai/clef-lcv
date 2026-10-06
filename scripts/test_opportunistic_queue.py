"""Real worker tests: no collection delay, FIFO barriers, early replies and errors."""
import asyncio,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from request_queue import DecisionQueue

async def until(predicate):
    deadline=time.perf_counter()+3
    while not predicate():
        assert time.perf_counter()<deadline
        await asyncio.sleep(.005)

async def main():
    first_entered,release_first=threading.Event(),threading.Event()
    batch_entered,release_batch=threading.Event(),threading.Event()
    calls=[]
    def single(p):
        calls.append([p['id']])
        if p['id']==0:first_entered.set();assert release_first.wait(3)
        return {'id':p['id'],'usage':{}}
    def batch(payloads,emit):
        calls.append([p['id'] for p in payloads]);batch_entered.set()
        for index,p in enumerate(payloads):
            if p['id']==2:assert release_batch.wait(3)
            emit(index,{'id':p['id'],'usage':{}})
    q=DecisionQueue(single,max_batch_size=2,batch_handler=batch,
                    batch_key=lambda p:None if p.get('image') else p.get('context'))
    await q.start()
    try:
        a=asyncio.create_task(q.run({'id':0,'context':'guide'}))
        await until(first_entered.is_set)  # No second request or timer needed.
        pending=[asyncio.create_task(q.run({'id':i,'context':'guide','image':i==3})) for i in range(1,6)]
        await until(lambda:len(q.pending)==5)
        release_first.set();assert (await a)['id']==0
        await until(batch_entered.is_set)
        assert (await pending[0])['id']==1  # Reply before second fallback completes.
        assert not pending[1].done() and len(q.pending)==3
        release_batch.set();rows=await asyncio.gather(*pending[1:])
        assert [r['id'] for r in rows]==[2,3,4,5]
        assert calls==[[0],[1,2],[3],[4,5]],calls
        assert q.stats()['collection_delay_ms']==0 and q.stats()['batch_dispatches']==2
    finally:
        release_first.set();release_batch.set();await q.close()
    # An invalid batch member does not fail its neighbour; a fatal batch error
    # does not leave either future or the following request stranded.
    entered,release=threading.Event(),threading.Event()
    def hold(p):
        if p==0:entered.set();assert release.wait(3)
        return {'id':p,'usage':{}}
    def errors(ps,emit):
        emit(0,ValueError('invalid member'));emit(1,{'id':ps[1],'usage':{}})
    q=DecisionQueue(hold,max_batch_size=2,batch_handler=errors,batch_key=lambda p:'same')
    await q.start()
    try:
        a=asyncio.create_task(q.run(0));await until(entered.is_set)
        b=asyncio.create_task(q.run(1));c=asyncio.create_task(q.run(2))
        await until(lambda:len(q.pending)==2);release.set();await a
        try:await b
        except ValueError:pass
        else:raise AssertionError('Lost batch member error')
        assert (await c)['id']==2
        assert (await q.run(3))['id']==3
        assert q.stats()['failed']==1 and q.stats()['completed']==3
    finally:release.set();await q.close()
    # Cancel an active batch member: all wire bytes stay admitted until the
    # GPU bundle finishes, including during shutdown.
    entered.clear();release.clear();batch_entered.clear();release_batch.clear()
    def blocked_batch(ps,emit):
        batch_entered.set();assert release_batch.wait(3)
        for i,p in enumerate(ps):emit(i,{'id':p,'usage':{}})
    q=DecisionQueue(hold,max_batch_size=2,batch_handler=blocked_batch,batch_key=lambda p:'same')
    await q.start()
    class Request:
        def __init__(self,ticket):self.scope={'clef_payload_ticket':ticket}
        async def is_disconnected(self):return False
    t1=q.budget.open();t1.grow(100);t2=q.budget.open();t2.grow(100)
    try:
        first=asyncio.create_task(q.run(0));await until(entered.is_set)
        one=asyncio.create_task(q.run(1,Request(t1)));two=asyncio.create_task(q.run(2,Request(t2)))
        await until(lambda:len(q.pending)==2);release.set();await first
        await until(batch_entered.is_set)
        one.cancel()
        try:await one
        except asyncio.CancelledError:pass
        t1.release()
        assert q.budget.count==2 and q.budget.bytes==200
        closing=asyncio.create_task(q.close());await asyncio.sleep(.01);assert not closing.done()
        release_batch.set();assert (await two)['id']==2;t2.release();await closing
        assert q.budget.count==q.budget.bytes==0 and q.stats()['cancelled']==1
    finally:
        release.set();release_batch.set()
        if not q.closing:await q.close()
    print('PASS: immediate lone dispatch, backlog-only batches, FIFO image barrier, early fallback replies and member isolation')

asyncio.run(main())
