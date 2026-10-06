"""Shared image/prefix admission cannot independently claim the same RAM."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'vendor/cloudflare')]
import torch
from ram_cache_budget import RAMCacheBudget
from host_prefix_cache import HostPrefixCache
from optimized_inference import tensor_bytes
from reusable_inputs import InputCache
pool=None;physical=64*2**20;extra=[0]
def probe():
 total=sum(size() for size,_,_ in pool.owners.values()) if pool else 0
 available=max(0,physical-extra[0]-total-sum(size() for size in pool.pending.values()))
 return {'available_bytes':available,'system_available_bytes':available,'cgroup_available_bytes':None}
pool=RAMCacheBudget(memory_probe=probe)
host=HostPrefixCache('auto',None,budget=pool)
images=InputCache('auto',None,ram_budget=pool,image_fraction=.25)
entry={'cache':SimpleNamespace(keys=torch.zeros(10*2**18)), 'chunks':(), 'ids':tuple(range(128)), 'media_key':('text',False)}
for i in range(4):assert host.put(i,entry,tensor_bytes)
assert host.bytes==40*2**20
# Processed images can evict a cold prefix when its bank fills the shared pool.
images._put_image('image',((),{'pixels':torch.zeros(11*2**18)}))
assert images.images and host.evictions>=1
assert host.bytes+images.image_bytes<=48*2**20
assert pool.estimate('images')[0]==12*2**20
extra[0]=32*2**20
host.trim();images.trim()
assert host.bytes+images.image_bytes<=24*2**20
# Auto cache accepts >128 entries when metadata fits; one hot image survives LFU.
images.clear();extra[0]=0
for i in range(140):images._put_image(i,((),{'pixels':torch.ones(1)}))
assert len(images.images)==140
images.max_entries=2
images.clear();images._put_image('hot',((),{}));images._put_image('cold',((),{}))
images.image_uses['hot']=(5,1);images._put_image('new',((),{}))
assert set(images.images)=={'hot','new'}
print('PASS: shared RAM headroom, cross-tier eviction, pressure shrink, image fraction, unlimited count and LFU')
