"""Real lookahead lifecycle: overlap, bounded slots, decline, errors and cancellation."""
import asyncio,base64,sys,threading,time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from cpu_preparation import preparation_fits
from request_queue import DecisionQueue,PayloadBudget

async def until(fn):
 end=time.perf_counter()+4
 while not fn():
  assert time.perf_counter()<end
  await asyncio.sleep(.005)

async def main():
 entered,release=threading.Event(),threading.Event()
 prepared=[];handled=[]
 def handler(p):
  value=p['id'];handled.append((value,p.get('prepared',False)))
  if value==0:entered.set();assert release.wait(4)
  return {'id':value,'usage':{}}
 def prepare(p):
  prepared.append(p['id']);return {**p,'prepared':True}
 q=DecisionQueue(handler,prepare_handler=prepare)
 await q.start()
 try:
  a=asyncio.create_task(q.run({'id':0}));await until(entered.is_set)
  pending=[asyncio.create_task(q.run({'id':i})) for i in range(1,4)]
  await until(lambda:prepared==[1])
  await asyncio.sleep(.02);assert prepared==[1] and q.stats()['cpu_preparation']['max_queued_slots']==1
  release.set();rows=await asyncio.gather(a,*pending)
  assert [r['id'] for r in rows]==[0,1,2,3]
  assert handled[0]==(0,False) and handled[1]==(1,True)
 finally:release.set();await q.close()
 # Preparation failure must not poison an otherwise compatible neighbour.
 entered.clear();release.clear()
 def bad(p):
  if p['id']==1:raise ValueError('invalid prepared member')
  return None
 def batch(ps,emit):
  for i,p in enumerate(ps):emit(i,handler(p))
 q=DecisionQueue(handler,prepare_handler=bad,max_batch_size=2,batch_handler=batch,batch_key=lambda p:'same')
 await q.start()
 try:
  a=asyncio.create_task(q.run({'id':0}));await until(entered.is_set)
  b=asyncio.create_task(q.run({'id':1}));c=asyncio.create_task(q.run({'id':2}))
  await until(lambda:q.prepare_slot is not None);release.set();await a
  try:await b
  except ValueError:pass
  else:raise AssertionError('Preparation error lost')
  assert (await c)['id']==2 and q.stats()['cpu_preparation']['failed_requests']==1
 finally:release.set();await q.close()
 # Wire admission survives cancellation until a CPU preparation actually ends.
 entered.clear();release.clear();prep_entered=threading.Event();prep_release=threading.Event()
 def held(p):prep_entered.set();assert prep_release.wait(4);return p
 q=DecisionQueue(handler,prepare_handler=held);await q.start()
 ticket=q.budget.open();ticket.grow(100)
 req=SimpleNamespace(scope={'clef_payload_ticket':ticket},is_disconnected=lambda:None)
 try:
  a=asyncio.create_task(q.run({'id':0}));await until(entered.is_set)
  b=asyncio.create_task(q.run({'id':1},req));await until(prep_entered.is_set)
  b.cancel()
  try:await b
  except asyncio.CancelledError:pass
  ticket.release();assert q.budget.bytes==100
  prep_release.set();await until(lambda:q.budget.bytes==0)
  release.set();await a
 finally:prep_release.set();release.set();await q.close()

b=BytesIO();Image.new('RGB',(512,512),'red').save(b,format='PNG')
request=SimpleNamespace(state='text',context=None,images=['data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()])
assert preparation_fits(request,128,lambda:{'available_bytes':2**30})
assert not preparation_fits(request,1,lambda:{'available_bytes':2**30})
assert not preparation_fits(request,128,lambda:None)
request.images=['invalid'];assert not preparation_fits(request,128,lambda:{'available_bytes':2**30})
asyncio.run(main())
print('PASS: immediate first request, real CPU/GPU overlap, one queued slot, decline, error isolation, retained wire tickets on cancellation, and memory/header bounds')
