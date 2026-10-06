"""Transfer planning and ready-before-use checks on CPU with mocked CUDA events."""
import contextlib,sys,gc,weakref
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import torch
from async_restore import RestorePlan
class Event:
 def __init__(self):self.recorded=False
 def record(self,stream):self.recorded=True
 def synchronize(self):assert self.recorded
class Stream:
 def synchronize(self):pass
 def wait_event(self,event):assert event.recorded,'An unrecorded CUDA event was consumed'
stream=Stream()
old=(torch.cuda.Event,torch.cuda.device,torch.cuda.stream,torch.cuda.current_stream,torch.Tensor.record_stream)
try:
 torch.cuda.Event=Event;torch.cuda.device=lambda *a:contextlib.nullcontext()
 torch.cuda.stream=lambda *a:contextlib.nullcontext();torch.cuda.current_stream=lambda *a:stream
 torch.Tensor.record_stream=lambda *a:None
 source=torch.arange(51,dtype=torch.float32)
 cache=NS(layers=[NS(keys=source[:16].view(1,2,4,2),values=source[16:32].view(1,2,4,2)),
   NS(conv_states=source[32:40],recurrent_states=source[40:48])],global_scalar=torch.tensor(4))
 plan=RestorePlan(cache,(source[48:],),'cpu',stream,[torch.empty(8,dtype=torch.uint8) for _ in range(2)])
 plan.transfer()
 for group in ['hidden','global',0,1]:plan.wait(group)
 assert torch.equal(plan.cache.layers[0].keys,cache.layers[0].keys)
 assert torch.equal(plan.cache.layers[1].recurrent_states,cache.layers[1].recurrent_states)
 assert torch.equal(plan.chunks[0],source[48:]) and plan.cache.global_scalar.item()==4
 plan.cache.layers[1].conv_states.zero_();assert cache.layers[1].conv_states.sum()>0
 assert not plan.groups,'CPU source tensors were retained after DMA completion'
 reference=weakref.ref(plan);gc.disable();del plan
 assert reference() is None,'Restore plans must release GPU destinations without waiting for cyclic GC'
 gc.enable()
finally:
 torch.cuda.Event,torch.cuda.device,torch.cuda.stream,torch.cuda.current_stream,torch.Tensor.record_stream=old
print('PASS: chunked byte transfers, scalar/hidden/KV/recurrent state, independent destinations, ready-event ordering and CPU source release')
