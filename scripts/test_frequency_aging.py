"""Fake-clock pressure tests: stale popularity, images, transfers and no expiry."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root/'runtime'), str(root/'vendor/cloudflare')]
import torch
from frequency_aging import FrequencyAging
from host_prefix_cache import HostPrefixCache
from optimized_inference import InferenceEngine, tensor_bytes
from reusable_inputs import InputCache
from cache_telemetry import CacheTelemetry


class Clock:
    def __init__(self): self.now = 0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += int(seconds*1e9)


clock = Clock()
policy = FrequencyAging(900, clock)
entry = policy.snapshot({})
policy.touch(entry, 7)
assert entry['uses']==7 and policy.effective(entry)==7
clock.advance(899); assert policy.effective(entry)==7
before=copy.copy(entry)
clock.advance(1); assert policy.effective(entry)==3 and entry==before
clock.advance(900); assert policy.effective(entry)==1
clock.advance(900); assert policy.effective(entry)==0 and entry['uses']==7
policy.touch(entry); assert policy.effective(entry)==1 and entry['uses']==8
policy.touch(entry,0); assert entry['last_use']==clock.now

# Preserve elapsed fractional periods across frequent hits and duplicate writes.
clock = Clock(); policy=FrequencyAging(10, clock)
entry=policy.snapshot({}); policy.touch(entry,8)
clock.advance(9); policy.touch(entry)
assert entry['frequency_updated_ns']==0
clock.advance(1); assert policy.effective(entry)==4 and entry['uses']==9
old=policy.snapshot(entry)
clock.advance(90); assert policy.effective(entry)==0
last=entry['last_use'];policy.touch(entry,0);assert entry['last_use']==last
policy.merge(entry,old)
assert policy.effective(entry)==0 and entry['uses']==9 and entry['last_use']==last
clock.advance(10); assert policy.effective(entry)==0
disabled=FrequencyAging(0,clock);assert disabled.effective(entry)==9
for seconds in (-1, float('nan'), float('inf'), 1e-20):
    try:FrequencyAging(seconds)
    except ValueError:pass
    else:raise AssertionError('Invalid decay interval accepted')

def snapshot(elements=128):
    return {'cache':SimpleNamespace(keys=torch.ones(elements)), 'chunks':(),
            'ids':tuple(range(256)), 'media_key':('text',False)}

# Historically popular CPU prefixes age to zero and lose ties by last access.
clock=Clock();policy=FrequencyAging(900,clock);log=CacheTelemetry(frequency=policy)
host=HostPrefixCache(1,2,frequency=policy,telemetry=log)
assert host.put('old',snapshot(),tensor_bytes);host.touch('old',100)
clock.advance(7*900)
assert host.put('new',snapshot(),tensor_bytes)
assert host.put('old',snapshot(),tensor_bytes)
assert host.entries['old']['last_use']==0,'A metadata-free duplicate write refreshed recency'
old_meta=copy.copy({k:v for k,v in host.entries['old'].items() if k in ('uses','last_use','frequency','frequency_updated_ns')})
host.stats(); log.inventory(host.entries,'cpu');log.read()
assert old_meta=={k:host.entries['old'][k] for k in old_meta}
assert host.entries['old']['uses']==100 and policy.effective(host.entries['old'])==0
assert host.put('next',snapshot(),tensor_bytes)
assert set(host.entries)=={'new','next'}
evicted=[e for e in log.events if e['action']=='evict'][-1]
assert evicted['uses']==100 and evicted['eviction_frequency']==0
clock.advance(100000);host.trim();assert len(host.entries)==2,'Age must not expire entries when memory fits'

# A currently used entry retains priority, and effective-frequency ties use LRU.
clock=Clock();policy=FrequencyAging(10,clock)
host=HostPrefixCache(1,2,frequency=policy)
assert host.put('old',snapshot(),tensor_bytes);host.touch('old',8)
clock.advance(40)
assert host.put('current',snapshot(),tensor_bytes);host.touch('current')
assert host.put('third',snapshot(),tensor_bytes)
assert set(host.entries)=={'current','third'}
clock.advance(10);host.touch('third');clock.advance(1);host.touch('current')
assert host.put('fourth',snapshot(),tensor_bytes)
assert set(host.entries)=={'current','fourth'}

# CPU image cache follows exactly the same clock and scoring semantics.
clock=Clock();policy=FrequencyAging(900,clock)
images=InputCache(max_entries=2,frequency=policy)
images._put_image('old',((),{}));images._touch_image('old',100)
clock.advance(7*900)
images._put_image('new',((),{}));before=copy.deepcopy(images.image_uses)
images.stats(); assert before==images.image_uses
images._put_image('next',((),{}))
assert set(images.images)=={'new','next'}
assert set(images.image_uses)==set(images.images)
clock.advance(100000);images.trim();assert len(images.images)==2
images.clear();assert not images.image_uses

# Disabled mode reproduces lifetime-LFU retention for prefixes and images.
clock=Clock();policy=FrequencyAging(0,clock)
host=HostPrefixCache(1,2,frequency=policy);images=InputCache(max_entries=2,frequency=policy)
assert host.put('old',snapshot(),tensor_bytes);host.touch('old',100)
images._put_image('old',((),{}));images._touch_image('old',100)
clock.advance(100000)
for key in ('new','next'):
    assert host.put(key,snapshot(),tensor_bytes);images._put_image(key,((),{}))
assert set(host.entries)==set(images.images)=={'old','next'}

# GPU reuse, spill, CPU restore and re-retention preserve age/count metadata.
original=(torch.cuda.mem_get_info,torch.cuda.memory_reserved,torch.cuda.memory_allocated)
try:
    torch.cuda.mem_get_info=lambda:(2**30,2**30)
    torch.cuda.memory_reserved=torch.cuda.memory_allocated=lambda:0
    engine=InferenceEngine(1,reserve_mib=0,host_cache_mib=1,frequency_decay_seconds=10)
    clock=Clock();engine.frequency.clock=clock
    model=object();source=snapshot()
    assert engine._retain(model,source['ids'],256,'text',False,None,source['cache'],())
    key=next(iter(engine.entries));engine.frequency.touch(engine.entries[key],8)
    stale=engine.frequency.snapshot(engine.entries[key])
    clock.advance(40);assert engine._drop()
    stored=engine.host.entries[key]
    assert stored['uses']==8 and engine.frequency.effective(stored)==0 and stored['last_use']==0
    assert engine.host.put(key,{**source,**stale},tensor_bytes)
    assert engine.frequency.effective(stored)==0 and stored['uses']==8
    _,tier=engine._acquire(key,'cpu');assert tier=='cpu'
    assert stored['uses']==9 and engine.frequency.effective(stored)==1
    assert engine._retain(model,source['ids'],256,'text',False,None,source['cache'],())
    assert engine.entries[key]['uses']==9 and engine.frequency.effective(engine.entries[key])==1
    engine.clear()
finally:
    torch.cuda.mem_get_info,torch.cuda.memory_reserved,torch.cuda.memory_allocated=original

print('PASS: lazy integer decay, partial periods, lifetime counters, no expiry, CPU prefix/image victims, disabled LFU, telemetry, GPU/CPU transfer and duplicate-write protection')
