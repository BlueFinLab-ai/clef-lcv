"""One GPU worker runs a FIFO short request at a safe long-request boundary."""
import asyncio,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from request_queue import DecisionQueue
async def until(fn):
 deadline=time.perf_counter()+4
 while not fn():
  assert time.perf_counter()<deadline
  await asyncio.sleep(.005)
async def main():
 ready=threading.Event();resume=threading.Event();ordering=[];threads=[]
 q=None
 def handler(p):
  threads.append(threading.get_ident())
  if p['id']==0:
   ordering.append('long-start');ready.set();assert resume.wait(4)
   prepared=q.peek_prepared();assert prepared is not None
   assert q.run_interleaved(prepared)
   ordering.append('long-end')
  else:ordering.append('short')
  return {'id':p['id'],'usage':{}}
 q=DecisionQueue(handler,prepare_handler=lambda p:p)
 await q.start()
 try:
  a=asyncio.create_task(q.run({'id':0}));await until(ready.is_set)
  b=asyncio.create_task(q.run({'id':1}));await until(lambda:q.peek_prepared() is not None)
  resume.set();rb=await b;ra=await a
  assert rb['usage']['chunk_interleaved'] and ordering==['long-start','short','long-end']
  assert len(set(threads))==1 and q.interleaved_requests==1
  # Reply futures are delivered before the worker's finally cleanup completes.
  await until(lambda:not q.pending and q.stats()['active_requests']==0)
 finally:resume.set();await q.close()
asyncio.run(main())
print('PASS: FIFO interleave, same GPU thread, early short reply, correct parent continuation and queue cleanup')
