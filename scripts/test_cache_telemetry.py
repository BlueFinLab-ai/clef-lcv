"""Diagnostics must be bounded, content-free, read-only and account for victims."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'runtime'), str(root / 'vendor/cloudflare')]
import torch
from cache_telemetry import CacheTelemetry
from host_prefix_cache import HostPrefixCache
from optimized_inference import tensor_bytes
from ram_cache_budget import RAMCacheBudget

trace = CacheTelemetry(4)
key = (1, 'private-image-url', False, 'private-token-hash')
outer = trace.begin()
trace.record('lookup', 'index', key, reason='miss')
inner = trace.begin()
trace.record('hit', 'cpu', key)
inner_trace = trace.end(inner)
trace.record('hit', 'gpu', key)
outer_trace = trace.end(outer)
assert inner_trace['request_id'] != outer_trace['request_id']
assert len(trace.events) == 4 and trace.read()['cursor_gap']
rows = trace.read(3)['events']
assert rows[0]['request_id'] == inner_trace['request_id']
assert rows[-1]['request_id'] == outer_trace['request_id']
assert trace.fingerprint(key) == rows[0]['key']
assert CacheTelemetry().fingerprint(key) != trace.fingerprint(key)
assert 'private-' not in json.dumps(trace.read())
assert trace.stats()['counters']['gpu.hit'] == 1
disabled = CacheTelemetry(0)
disabled.record('hit', 'gpu', key)
assert not disabled.events and disabled.stats()['counters']['gpu.hit'] == 1

entry = {'cache': SimpleNamespace(keys=torch.ones(128)), 'chunks': (),
         'ids': tuple(range(256)), 'media_key': ('private-image-url', False)}
trace = CacheTelemetry()
host = HostPrefixCache(1, 2, telemetry=trace)
assert host.put('hot', entry, tensor_bytes)
host.touch('hot', 3)
assert host.put('cold', entry, tensor_bytes)
assert host.put('new', entry, tensor_bytes)
evictions = [row for row in trace.events if row['action']=='evict']
assert len(evictions)==1 and evictions[0]['reason']=='entry_limit'
assert evictions[0]['key']==trace.fingerprint('cold') and evictions[0]['uses']==0
assert evictions[0]['released_bytes']==512
before = (host.bytes, list(host.entries), host.evictions, host.limit_bytes)
inventory = trace.inventory(host.entries, 'cpu')
host.stats(); trace.read(); trace.stats()
assert before == (host.bytes, list(host.entries), host.evictions, host.limit_bytes)
assert inventory['entries']==2 and not inventory['truncated']
assert 'private-' not in json.dumps(inventory)
assert host.put('hot', entry, tensor_bytes)
assert trace.events[-1]['action']=='duplicate'
assert not host.put('oversize', {**entry, 'chunks': (torch.zeros(300000),)}, tensor_bytes)
assert trace.events[-1]['reason']=='snapshot_exceeds_byte_budget'
assert trace.events[-1]['requested_bytes']>2**20

budget = RAMCacheBudget(memory_probe=lambda: {'available_bytes': 32*2**20,
    'system_available_bytes': 32*2**20, 'cgroup_available_bytes': None})
bank = HostPrefixCache('auto', None, budget=budget, telemetry=trace)
assert bank.put('one', entry, tensor_bytes)
state = budget.stats()
assert state['retained_bytes_by_tier']['prefix']==bank.bytes
assert state['total_cache_budget_bytes']+state['reserve_bytes']==state['service_capacity_bytes']
assert bank.evict_one()
assert trace.events[-1]['reason']=='shared_ram_budget'
print('PASS: bounded private telemetry, nested request attribution, LFU victims, exact bytes and read-only RAM budgets')
