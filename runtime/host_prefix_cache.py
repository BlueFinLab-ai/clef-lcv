"""Bounded, process-local CPU snapshots. No disk storage or GPU allocations here."""
import copy
import time
from collections import OrderedDict
from threading import RLock
from functools import wraps

import torch
from host_memory import available_memory
from shared_host_blocks import FrozenTensor, SnapshotPlan, storages


def copy_bytes(value, seen=None):
    """Charge each distinct tensor view for the compact copy it will allocate."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, FrozenTensor):
        return value.nbytes
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(copy_bytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(copy_bytes(v, seen) for v in value)
    if hasattr(value, '__dict__'):
        return copy_bytes(vars(value), seen)
    return 0


def copy_to_device(value, device, preserve=None):
    """Preserve cache classes/metadata while replacing every tensor independently.

    Memoization preserves repeated tensor references. Views are compacted so a
    short hidden-state slice cannot pin the whole original request's storage.
    Copies are synchronous: a snapshot is never published before it is complete.
    """
    memo = dict(preserve or {})
    seen = set()
    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if id(item) in memo:return
        if isinstance(item, FrozenTensor):
            memo[id(item)] = item.materialize(device)
        elif isinstance(item, torch.Tensor):
            memo[id(item)] = item.detach().to(device=device, copy=True, memory_format=torch.contiguous_format)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, '__dict__'):
            visit(vars(item))
    visit(value)
    del visit  # Release the memo's GPU copies when this call returns, not at GC.
    return copy.deepcopy(value, memo)


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


class HostPrefixCache:
    def __init__(self, limit_mib=0, max_entries=128, reserve_fraction=.25, memory_probe=None, budget=None, shared_blocks=False):
        self.auto = str(limit_mib).lower() == 'auto'
        if (not self.auto and int(limit_mib) < 0) or (max_entries is not None and max_entries < 0):
            raise ValueError('Host prefix cache limits must be nonnegative')
        if not 0 <= reserve_fraction < 1:
            raise ValueError('Host prefix cache reserve fraction must be in [0, 1)')
        self.configured_bytes = None if self.auto else int(limit_mib) * 2**20
        self.limit_bytes = self.configured_bytes or 0
        self.max_entries = None if max_entries is None else int(max_entries)
        self.reserve_fraction = reserve_fraction
        self.memory_probe = memory_probe or available_memory
        self.memory_sample = None
        self.reserve_bytes = 0
        self.entries = OrderedDict()
        self.bytes = 0
        self.shared_blocks = shared_blocks
        self.storage_refs = {}
        self.shared_saves = 0
        self.budget = budget
        self.lock = budget.lock if budget is not None else RLock()
        if budget is not None:
            budget.register('prefix', lambda: self.bytes, self.evict_one)
        self.offloads = self.hits = self.evictions = self.failures = 0
        self.offload_ms = 0.0
        self.pending = None
        self.refresh_budget()

    def estimate_budget(self):
        if self.configured_bytes == 0:
            return 0, 0, None
        if self.budget is not None:
            budget, reserve, sample = self.budget.estimate('prefix')
            return budget if self.auto else min(self.configured_bytes, budget), reserve, sample
        sample = self.memory_probe()
        if sample is not None:
            # Add back only this tier's live allocations. This makes the cap
            # stable as we fill it, while other services/workloads lower it.
            capacity = sample['available_bytes'] + self.bytes + (self.pending() if self.pending else 0)
            reserve = int(capacity * self.reserve_fraction)
            budget = max(0, capacity - reserve)
            return budget if self.auto else min(self.configured_bytes, budget), reserve, sample
        else:
            # Auto mode fails closed; explicit caps retain their legacy behavior.
            return self.configured_bytes or 0, 0, None

    def refresh_budget(self):
        self.limit_bytes, self.reserve_bytes, self.memory_sample = self.estimate_budget()
        return self.limit_bytes

    @property
    def enabled(self):
        return (self.auto or self.configured_bytes > 0) and self.max_entries != 0

    @synchronized
    def _evict(self,protected=None):
        # LFU; least recent access breaks ties deterministically. Reads of cache
        # metadata and repeated snapshot writes are not counted as actual reuse.
        candidates=[key for key in self.entries if key!=protected]
        if not candidates:return False
        key = min(candidates, key=lambda k: (self.entries[k].get('uses', 0), self.entries[k]['last_use']))
        old = self.entries.pop(key)
        if self.shared_blocks:
            for ident in old['storage_ids']:
                size, refs = self.storage_refs[ident]
                if refs == 1:
                    self.bytes -= size; del self.storage_refs[ident]
                else:self.storage_refs[ident]=(size,refs-1)
        else:self.bytes -= old['bytes']
        self.evictions += 1
        return True

    @synchronized
    def evict_one(self,protected=None):
        if not self.entries: return False
        return self._evict(protected)

    @synchronized
    def trim(self):
        if self.budget is not None:
            with self.budget.lock:
                self.budget.admit('prefix', maximum=self.configured_bytes)
        self.refresh_budget()
        while self.entries and (self.bytes > self.limit_bytes or
                (self.max_entries is not None and len(self.entries) > self.max_entries)):
            self._evict()

    @synchronized
    def touch(self, key, uses=1):
        if key in self.entries:
            entry = self.entries[key]
            entry['uses'] = entry.get('uses', 0) + uses
            entry['last_use'] = time.monotonic_ns()

    @synchronized
    def put(self, key, entry, byte_counter, *, adopt_cpu=False):
        if self.budget is not None:
            with self.budget.lock:
                return self._put(key, entry, byte_counter,adopt_cpu=adopt_cpu)
        return self._put(key, entry, byte_counter,adopt_cpu=adopt_cpu)

    def _put(self, key, entry, byte_counter, *, adopt_cpu=False):
        self.trim()
        if not self.enabled:
            return False
        # Different views can share GPU storage but require independent compact
        # CPU allocations. Check the copy's size before allocating, not the GPU
        # storage total; repeated references to the same tensor still count once.
        parent_key=entry.get('parent_key')
        parent = self.entries.get(parent_key) if self.shared_blocks else None
        if parent is not None:
            start=entry.get('media_start')
            namespace=(key[0],'text',False) if start is None or len(parent['ids'])<=start else key[:3]
            if (not isinstance(key,tuple) or not isinstance(parent_key,tuple) or parent_key[:3]!=namespace
                    or len(parent['ids'])>len(entry['ids']) or entry['ids'][:len(parent['ids'])]!=parent['ids']):
                parent=None
        plan = SnapshotPlan(entry,parent) if self.shared_blocks and not adopt_cpu else None
        if key in self.entries:
            self.entries[key]['uses'] = max(self.entries[key]['uses'], entry.get('uses', 0))
            self.entries[key]['last_use'] = max(self.entries[key]['last_use'], entry.get('last_use', 0))
            return True
        estimate = (lambda:plan.retained_increment(self.storage_refs)) if plan is not None else (lambda:copy_bytes((entry['cache'],entry['chunks'])))
        if not adopt_cpu and not self._admit_snapshot(estimate,check_count=True):return False
        started = time.perf_counter()
        try:
            if adopt_cpu:
                preserve={}
                for layer in entry['cache'].layers:
                    for name in ('keys','values'):
                        value=getattr(layer,name,None)
                        if isinstance(value,FrozenTensor):preserve[id(value)]=value
                        elif isinstance(value,torch.Tensor) and value.ndim==4:
                            if value.device.type!='cpu':raise ValueError('Adoption requires CPU KV')
                            preserve[id(value)]=FrozenTensor(value.shape,value.dtype,2,(value,),value.element_size())
                for value in entry['chunks']:
                    if value.device.type!='cpu':raise ValueError('Adoption requires CPU hidden vectors')
                    preserve[id(value)]=FrozenTensor(value.shape,value.dtype,1,(value,),value.element_size())
                cache,chunks=copy_to_device((entry['cache'],entry['chunks']),'cpu',preserve)
            else:cache, chunks = plan.copy() if plan is not None else copy_to_device((entry['cache'], entry['chunks']), 'cpu')
        except (MemoryError, RuntimeError):
            self.failures += 1
            return False
        storage_map=storages((cache,chunks)) if self.shared_blocks else None
        increment = (lambda:sum(length for ident,length in storage_map.items() if ident not in self.storage_refs)) if storage_map is not None else (lambda:byte_counter((cache,chunks)))
        self.pending=increment
        if self.budget is not None:self.budget.pending['prefix']=increment
        try:
            admitted=self._admit_snapshot(increment)
        finally:
            self.pending=None
            if self.budget is not None:self.budget.pending.pop('prefix',None)
        if not admitted:return False
        size=increment()
        self.entries[key] = {'cache': cache, 'chunks': chunks, 'ids': entry['ids'],
                             'media_key': entry['media_key'], 'bytes': copy_bytes((cache,chunks)) if self.shared_blocks else size,
                             'uses': entry.get('uses', 0), 'last_use': entry.get('last_use', time.monotonic_ns())}
        self.bytes += size
        if self.shared_blocks:
            self.entries[key]['storage_ids']=tuple(storage_map)
            for ident, length in storage_map.items():
                self.storage_refs[ident]=(length,self.storage_refs.get(ident,(length,0))[1]+1)
            if plan is not None and plan.reused:self.shared_saves+=1
        self.offloads += 1
        self.offload_ms += (time.perf_counter() - started) * 1000
        return True

    def _admit_snapshot(self,increment,check_count=False):
        while True:
            self.refresh_budget();size=increment()
            if size>self.limit_bytes:return False
            count_full=check_count and self.max_entries is not None and len(self.entries)>=self.max_entries
            if count_full or self.bytes+size>self.limit_bytes:
                if not self.evict_one():return False
                continue
            if self.budget is not None:
                if not self.budget.admit('prefix',size,self.configured_bytes):return False
                if increment()!=size:continue  # A shared ancestor may have been evicted.
            return True

    @synchronized
    def clear(self):
        self.entries.clear()
        self.bytes = 0
        self.storage_refs.clear()

    @synchronized
    def stats(self):
        # Health runs outside the GPU worker. A CPU copy in progress consumes RAM
        # before it is indexed, so telemetry must not change an active admission.
        limit, reserve, sample = self.estimate_budget()
        return {'entries': len(self.entries), 'mib': round(self.bytes / 2**20, 2),
                'limit_mib': limit / 2**20, 'max_entries': self.max_entries,
                'budget_mode': 'auto' if self.auto else 'fixed', 'eviction_policy': 'lfu_lru_tiebreak',
                'reserve_fraction': self.reserve_fraction, 'reserve_mib': round(reserve / 2**20, 2),
                'available_mib': None if sample is None else round(sample['available_bytes'] / 2**20, 2),
                'cgroup_available_mib': None if sample is None or sample['cgroup_available_bytes'] is None
                    else round(sample['cgroup_available_bytes'] / 2**20, 2),
                'offloads': self.offloads, 'hits': self.hits, 'evictions': self.evictions,
                'shared_blocks':self.shared_blocks,'shared_saves':self.shared_saves,
                'logical_mib':round(sum(v['bytes'] for v in self.entries.values())/2**20,2) if self.shared_blocks else round(self.bytes/2**20,2),
                'failures': self.failures, 'offload_ms': round(self.offload_ms, 1)}
