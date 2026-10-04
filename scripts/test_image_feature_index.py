"""CPU structural checks of independent vision-feature retention and miss batching."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/cloudflare'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import torch
from optimized_inference import InferenceEngine

class Visual:
    dtype=torch.float32
    def __init__(self):self.calls=[]
    def __call__(self,pixels,grid_thw,**kwargs):
        self.calls.append(len(grid_thw))
        values=pixels.split([int(g.prod()) for g in grid_thw])
        return SimpleNamespace(pooler_output=torch.stack([v.mean(0) for v in values]))
original=(torch.cuda.mem_get_info,torch.cuda.memory_reserved,torch.cuda.memory_allocated)
try:
    torch.cuda.mem_get_info=lambda:(2**30,2**30)
    torch.cuda.memory_reserved=lambda:0
    torch.cuda.memory_allocated=lambda:0
    engine=InferenceEngine(128,reserve_mib=0,feature_cache_entries=1)
    base=SimpleNamespace(visual=Visual())
    grid=torch.tensor([[1,2,2],[1,2,2]])
    media={'pixel_values':torch.cat([torch.ones(4,2),torch.full((4,2),2.)])}
    encoded=SimpleNamespace(media={'image_cache_keys':('a','b')})
    values,hits,misses=engine._vision(base,encoded,media,grid,True)
    assert (hits,misses)==(0,2) and base.visual.calls==[2] and len(engine.features)==1
    assert torch.equal(values,torch.tensor([[1.,1.],[2.,2.]]))
    values,hits,misses=engine._vision(base,encoded,media,grid,True)
    assert (hits,misses)==(1,1) and base.visual.calls[-1]==1 and len(engine.features)==1
    # Reorder while one entry is resident; cached value and fresh value retain order.
    reverse=SimpleNamespace(media={'image_cache_keys':('b','a')})
    values,hits,misses=engine._vision(base,reverse,{'pixel_values':media['pixel_values'].flip(0)},grid,True)
    assert torch.equal(values,torch.tensor([[2.,2.],[1.,1.]]))
    engine.clear()
    duplicate=SimpleNamespace(media={'image_cache_keys':('a','a')})
    pixels={'pixel_values':torch.ones(8,2)}
    values,hits,misses=engine._vision(base,duplicate,pixels,grid,True)
    assert misses==1 and base.visual.calls[-1]==1 and torch.equal(values,torch.ones(2,2))
    key=next(iter(engine.features));old=engine.features[key].clone()
    values[0,0]=90
    assert torch.equal(engine.features[key],old),'Request result mutated retained feature'
    other=SimpleNamespace(visual=Visual())
    _,hits,misses=engine._vision(other,duplicate,pixels,grid,True)
    assert hits==0 and misses==1,'Model identity leaked'
    assert engine._bytes()<=engine.max_bytes and engine.feature_evictions>0
    engine.clear();engine.feature_limit=0
    engine._vision(base,duplicate,pixels,grid,True)
    assert not engine.features
finally:torch.cuda.mem_get_info,torch.cuda.memory_reserved,torch.cuda.memory_allocated=original
print('PASS: feature LRU, partial misses, order, duplicate computation, independent storage, model isolation, shared memory accounting and zero retention')
