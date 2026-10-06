"""CPU checks for request/idle budgets and snapshot versus branch allocations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/cloudflare'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import torch
from optimized_inference import InferenceEngine
G=2**30
original=(torch.cuda.mem_get_info,torch.cuda.memory_reserved,torch.cuda.memory_allocated,torch.cuda.empty_cache)
allocated=17*G
flushes=[]
try:
    torch.cuda.empty_cache=lambda:flushes.append(True)
    torch.cuda.mem_get_info=lambda:(24*G-allocated,24*G)
    torch.cuda.memory_reserved=lambda:allocated
    torch.cuda.memory_allocated=lambda:allocated
    elastic=InferenceEngine('auto',utilization=1.,reserve_mib=1024,elastic=True)
    elastic.base_bytes=17*G
    elastic._bytes=lambda:sum(e['simulated_bytes'] for e in elastic.entries.values())
    elastic.prepare_request(3*G)
    assert elastic._budget()==3*G
    # Three GiB of working tensors already exist; reserving that peak again
    # would reject a snapshot which fits with one GiB of fixed free headroom.
    allocated=20*G
    assert elastic._required_free(2*G)==3*G
    legacy=InferenceEngine('auto',utilization=1.,reserve_mib=1024)
    legacy.base_bytes=17*G;legacy.prepare_request(3*G)
    assert legacy._required_free(2*G)==5*G
    assert elastic._required_free(2*G)<24*G-allocated<legacy._required_free(2*G)
    # The branch clone is active request state already covered by workspace.
    elastic.entries['snapshot']={'simulated_bytes':2*G}
    allocated=19*G
    assert elastic._required_free(2*G,request_clone=True)==4*G
    assert elastic._required_free(2*G)==6*G
    elastic.finish_request()
    assert elastic._budget()==6*G and len(elastic.entries)==1
    assert elastic.stats()['phase']=='idle'
    # An incoming larger prompt reclaims idle cache before model work starts.
    elastic.prepare_request(int(5.5*G))
    assert not elastic.entries and elastic.evictions==1
    assert elastic._budget()==G//2
    # Other processes reduce the budget; cached allocator blocks count as reusable.
    allocated=17*G
    torch.cuda.mem_get_info=lambda:(5*G,24*G)
    elastic.finish_request()
    assert elastic._budget()==4*G
    # Fixed cache limits still apply and legacy release is a no-op.
    fixed=InferenceEngine(128,reserve_mib=1024,elastic=True)
    fixed.base_bytes=17*G;fixed.prepare_request(3*G);fixed.finish_request()
    assert fixed.max_bytes==128*2**20
    legacy.finish_request();assert legacy.request_workspace_bytes==3*G
    # Reproduce the measured small-card admission decision: the 8K snapshot
    # fits beside observed workspace at 256 MiB, but must leave for 24K prefill.
    M=2**20
    adaptive=InferenceEngine('auto',utilization=1.,reserve_mib=256,
                             prefill_reserve_mib=512,elastic=True)
    adaptive.base_bytes=4990*M
    adaptive._bytes=lambda:sum(e['simulated_bytes'] for e in adaptive.entries.values())
    torch.cuda.memory_allocated=lambda:adaptive.base_bytes+adaptive._bytes()
    torch.cuda.memory_reserved=torch.cuda.memory_allocated
    torch.cuda.mem_get_info=lambda:(8192*M-torch.cuda.memory_allocated()-574*M,8192*M)
    adaptive.workspace_bytes=1989*M
    adaptive.entries['8k']={'simulated_bytes':343*M}
    before=len(flushes)
    prep=adaptive.prepare_request()
    assert prep['cache_headroom_mib']==256 and prep['cache_evicted_mib']==0
    assert len(adaptive.entries)==1 and len(flushes)==before
    prep=adaptive.prepare_request(2000*M)
    assert prep['cache_headroom_mib']==512 and prep['cache_evicted_mib']==343
    assert not adaptive.entries and len(flushes)==before+1
    assert adaptive.stats()['effective_reserve_mib']==512
    # No blanket flush on repeated prefill without an eviction.
    adaptive.prepare_request(2000*M);assert len(flushes)==before+1
    adaptive.finish_request()
    assert adaptive.stats()['effective_reserve_mib']==256
    assert adaptive._budget()==2372*M
    # Fixed-limit ROCm has no calibrated projection. A short-call observation
    # must not override the larger measured chunked peak on the next prefill.
    adaptive.workspace_bytes=224*M
    adaptive.chunked_workspace_bytes=2000*M
    adaptive.entries['branch']={'simulated_bytes':343*M}
    adaptive.prepare_request()
    assert adaptive.request_workspace_bytes is None and len(adaptive.entries)==1
    prep=adaptive.prepare_request(observed_chunked=True)
    assert adaptive.request_workspace_bytes==2000*M
    assert prep['cache_headroom_mib']==512 and prep['cache_evicted_mib']==343
    assert not adaptive.entries
    adaptive.prepare_request(2100*M, observed_chunked=True)
    assert adaptive.request_workspace_bytes==2100*M
    adaptive.finish_request()
    assert adaptive.request_workspace_bytes==0 and adaptive._reserve()==256*M
    # A caller's larger minimum is never reduced by the prefill override.
    assert elastic.prefill_reserve_bytes==1024*M
finally:
    torch.cuda.mem_get_info,torch.cuda.memory_reserved,torch.cuda.memory_allocated,torch.cuda.empty_cache=original
print('PASS: idle expansion, adaptive prefill margin, selective allocator release, transient clones, pressure eviction, other GPU consumers and legacy/fixed policy')
