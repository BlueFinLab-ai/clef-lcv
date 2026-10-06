"""CPU checks for bounded offload, exact identity, independent restore and pressure."""
from pathlib import Path
from types import SimpleNamespace
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'vendor/cloudflare'), str(root / 'runtime')]
import torch
from optimized_inference import InferenceEngine, tensor_bytes
from host_prefix_cache import HostPrefixCache, copy_to_device

storage = torch.arange(4096, dtype=torch.float32)
cache = SimpleNamespace(layers=[SimpleNamespace(keys=storage[:128], values=storage[128:256],
    conv_states=torch.ones(1, 8, 4), recurrent_states=torch.ones(1, 2, 8, 8))])
entry = {'cache': cache, 'chunks': (storage[256:384],), 'ids': tuple(range(256)),
         'media_key': ('image-a', False)}
host = HostPrefixCache(1, 2)
assert host.put('a', entry, tensor_bytes)
assert host.bytes < storage.untyped_storage().nbytes(), 'A short view retained the entire source storage'
copy = copy_to_device(host.entries['a'], 'cpu')
copy['cache'].layers[0].keys.fill_(99)
copy['cache'].layers[0].conv_states.zero_()
copy['cache'].layers[0].recurrent_states.zero_()
copy['chunks'][0].zero_()
assert torch.equal(host.entries['a']['cache'].layers[0].keys, torch.arange(128, dtype=torch.float32))
assert host.entries['a']['cache'].layers[0].conv_states.all()
assert host.entries['a']['cache'].layers[0].recurrent_states.all()
assert host.entries['a']['chunks'][0].sum() > 0
assert host.put('b', entry, tensor_bytes) and host.put('c', entry, tensor_bytes)
assert list(host.entries) == ['b', 'c'] and host.evictions == 1
assert not host.put('oversize', {**entry, 'chunks': (torch.zeros(300000),)}, tensor_bytes)
assert host.bytes == sum(v['bytes'] for v in host.entries.values())
assert not HostPrefixCache().put('a', entry, tensor_bytes)
# Actual reuse drives LFU, with old access breaking ties. Writes are not hits.
lfu = HostPrefixCache(1, 2)
assert lfu.put('hot', entry, tensor_bytes)
lfu.touch('hot', 3)
assert lfu.put('cold', entry, tensor_bytes)
assert lfu.put('cold', entry, tensor_bytes)
assert lfu.entries['cold']['uses'] == 0
assert lfu.put('new', entry, tensor_bytes)
assert set(lfu.entries) == {'hot', 'new'} and lfu.entries['hot']['uses'] == 3
tie = HostPrefixCache(1, 2)
assert tie.put('a', entry, tensor_bytes) and tie.put('b', entry, tensor_bytes)
tie.touch('b'); tie.touch('a')
assert tie.put('c', entry, tensor_bytes)
assert set(tie.entries) == {'a', 'c'}, 'LFU ties must evict the least recent use'

# Auto cap adds back only its own live snapshots, preserving 25% headroom.
external = [0]
auto = None
def probe():
    available = max(0, 64 * 2**20 - external[0] - (auto.bytes if auto else 0) - (auto.pending() if auto and auto.pending else 0))
    return {'available_bytes': available, 'system_available_bytes': available, 'cgroup_available_bytes': None}
auto = HostPrefixCache('auto', None, memory_probe=probe)
assert auto.limit_bytes == 48 * 2**20
assert auto.put('hot', entry, tensor_bytes) and auto.put('cold', entry, tensor_bytes)
auto.touch('hot')
auto.refresh_budget()
assert auto.limit_bytes == 48 * 2**20, 'Growing cache must not recursively shrink its own budget'
external[0] = 64 * 2**20 - auto.bytes
active_budget = auto.limit_bytes
assert auto.stats()['limit_mib'] < active_budget / 2**20
assert auto.limit_bytes == active_budget, 'Health reads must not mutate in-progress CPU copy admission'
auto.trim()
assert set(auto.entries) == {'hot'}, 'External pressure should evict the cold entry first'
assert auto.bytes <= auto.limit_bytes
unknown = HostPrefixCache('auto', None, memory_probe=lambda: None)
assert not unknown.put('a', entry, tensor_bytes), 'Auto capacity must fail closed without a memory reading'
explicit = HostPrefixCache(8, memory_probe=lambda: {'available_bytes': 4 * 2**20,
    'system_available_bytes': 4 * 2**20, 'cgroup_available_bytes': None})
assert explicit.limit_bytes == 3 * 2**20, 'Explicit caps must also respect memory pressure'
unlimited = HostPrefixCache(1, None)
for i in range(140):
    assert unlimited.put(i, {**entry, 'cache': SimpleNamespace(keys=torch.ones(1)), 'chunks': ()}, tensor_bytes)
assert len(unlimited.entries) == 140
# Aliased GPU views must be charged for distinct CPU copies before allocation.
import host_prefix_cache as host_module
aliased = torch.zeros(150000)
large_views = {**entry, 'cache': SimpleNamespace(keys=aliased[:], values=aliased[:])}
original_copy = host_module.copy_to_device
def forbidden_copy(*args):
    raise AssertionError('An oversized CPU copy was allocated before admission')
try:
    host_module.copy_to_device = forbidden_copy
    assert not HostPrefixCache(1).put('alias', large_views, tensor_bytes)
finally:
    host_module.copy_to_device = original_copy


original = (torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated)
try:
    torch.cuda.mem_get_info = lambda: (2**30, 2**30)
    torch.cuda.memory_reserved = torch.cuda.memory_allocated = lambda: 0
    model = object()
    engine = InferenceEngine(0, reserve_mib=0, host_cache_mib=1)
    assert engine._retain(model, entry['ids'], 256, 'image-a', False, 128,
                          entry['cache'], entry['chunks'])
    assert not engine.entries and engine.host.entries
    key, common = engine._match(model, entry['ids'], 256, 'image-a', False, 128, 192)
    assert key is not None and common == 256
    assert engine._match(model, entry['ids'], 256, 'image-b', False, 128, 192)[0] is None
    assert engine._match(model, entry['ids'], 256, 'image-a', True, 128, 192)[0] is None
    assert engine._match(object(), entry['ids'], 256, 'image-a', False, 128, 192) == (None, 0)
    restored, tier = engine._acquire(key, 'cpu')
    assert tier == 'cpu' and engine.host.hits == 1
    restored['cache'].layers[0].values.zero_()
    assert engine.host.entries[key]['cache'].layers[0].values.sum() > 0
    # RAM is not counted as reclaimable GPU capacity.
    assert engine._bytes() == 0 and engine.host.bytes > 0
    torch.cuda.mem_get_info = lambda: (0, 2**30)
    assert engine._acquire(key, 'cpu') == (None, 'recompute')
    assert engine.restore_rejections == 1 and key in engine.host.entries
    engine.clear()
    assert not engine.host.entries and engine.host.bytes == 0
    # Normal GPU eviction spills state while preserving exact matching metadata.
    torch.cuda.mem_get_info = lambda: (2**30, 2**30)
    engine = InferenceEngine(1, reserve_mib=0, host_cache_mib=1)
    assert engine._retain(model, entry['ids'], 256, 'image-a', False, 128, cache, entry['chunks'])
    assert engine._drop() and engine.host.stats()['offloads'] == 1
    assert not engine.entries and engine._match(model, entry['ids'], 256, 'image-a', False, 128, 192)[0]
    # GPU reuse counts survive the spill into the LFU CPU tier.
    engine = InferenceEngine(1, reserve_mib=0, host_cache_mib=1)
    assert engine._retain(model, entry['ids'], 256, 'image-a', False, 128, cache, entry['chunks'])
    key = next(iter(engine.entries))
    assert engine._acquire(key, 'cpu')[1] == 'gpu'
    last_use = engine.entries[key]['last_use']
    assert engine._drop() and engine.host.entries[key]['uses'] == 1
    assert engine.host.entries[key]['last_use'] == last_use, 'Offload must preserve real GPU access time'
finally:
    torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated = original

print('PASS: automatic RAM budget, pressure trimming, LFU/ties, unlimited entries, CPU view compaction, exact identity, independent state, restore rejection, GPU accounting and inherited reuse')
