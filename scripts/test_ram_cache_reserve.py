"""Default 20% reserve expands shared capacity without losing pressure guards."""
from pathlib import Path
from types import SimpleNamespace
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'runtime'),str(root/'vendor/cloudflare')]
import torch
from ram_cache_budget import RAMCacheBudget
from host_prefix_cache import HostPrefixCache
from reusable_inputs import InputCache
from optimized_inference import InferenceEngine,tensor_bytes

pool=None;physical=64*2**20;external=[0]
def probe():
 retained=sum(size() for size,_,_ in pool.owners.values()) if pool else 0
 pending=sum(size() for size in pool.pending.values()) if pool else 0
 available=max(0,physical-external[0]-retained-pending)
 return {'available_bytes':available,'system_available_bytes':available,'cgroup_available_bytes':None}
pool=RAMCacheBudget(memory_probe=probe)
host=HostPrefixCache('auto',None,budget=pool)
images=InputCache('auto',None,ram_budget=pool)
assert pool.reserve_fraction==host.reserve_fraction==.20
assert InferenceEngine().host.reserve_fraction==.20
cap=physical-int(physical*.20)
assert pool.estimate('prefix')[0]==cap and pool.estimate('images')[0]==int(cap*.25)
entry={'cache':SimpleNamespace(keys=torch.zeros(2**20)), 'chunks':(), 'ids':tuple(range(128)), 'media_key':('text',False)}
for i in range(4):assert host.put(i,entry,tensor_bytes)
assert pool.estimate('prefix')[0]==cap,'Filling the cache changed its service capacity'
assert host.stats()['reserve_fraction']==.20
before=(host.bytes,host.evictions,host.limit_bytes)
sample=pool.stats();host.stats();images.stats()
assert before==(host.bytes,host.evictions,host.limit_bytes)
assert sample['reserve_bytes']==int(physical*.20)
external[0]=48*2**20
host.trim();images.trim()
assert host.bytes+images.image_bytes<=16*2**20-int(16*2**20*.20)
assert host.evictions>0,'External pressure was not honored'

# Even an explicit large cap obeys visible cgroup headroom at the new reserve.
limited=HostPrefixCache(1024,memory_probe=lambda:{'available_bytes':8*2**20,
 'system_available_bytes':64*2**20,'cgroup_available_bytes':8*2**20})
assert limited.limit_bytes==8*2**20-int(8*2**20*.20)
for fraction in (-.1,1.,float('nan')):
 try:RAMCacheBudget(fraction)
 except ValueError:pass
 else:raise AssertionError('Invalid RAM reserve accepted')
print('PASS: 20% defaults, 80% shared capacity, unchanged image fraction, read-only metrics, pressure trimming and cgroup/fixed-cap guards')
