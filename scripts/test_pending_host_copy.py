"""A large unpublished snapshot must be credited after allocation exactly once."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'vendor/cloudflare')]
import torch
from host_prefix_cache import HostPrefixCache,copy_to_device
from ram_cache_budget import RAMCacheBudget
from optimized_inference import tensor_bytes

for shared_budget in (False,True):
    allocated=[0];capacity=100*1024
    def probe():return {'available_bytes':capacity-allocated[0],'system_available_bytes':capacity-allocated[0],'cgroup_available_bytes':None}
    pool=RAMCacheBudget(memory_probe=probe) if shared_budget else None
    host=HostPrefixCache('auto',None,memory_probe=probe,budget=pool)
    data={'cache':SimpleNamespace(keys=torch.zeros(16*1024)), 'chunks':(), 'ids':tuple(range(128)),'media_key':('text',False)}
    def copied(value,device):
        result=copy_to_device(value,device);allocated[0]+=tensor_bytes(result);return result
    with patch('host_prefix_cache.copy_to_device',copied):
        assert host.put('large',data,tensor_bytes)
    assert host.bytes==64*1024 and host.estimate_budget()[0]==75*1024
    assert host.pending is None and (pool is None or not pool.pending)
    assert not host.evict_one(protected='large')
    assert 'large' in host.entries
print('PASS: allocated-but-unpublished RAM credited once; standalone/shared budgets; protected eviction')
