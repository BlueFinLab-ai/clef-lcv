"""CPU checks for exact ancestry, media separation, memory accounting and eviction."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/cloudflare'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import torch
from optimized_inference import InferenceEngine

original = (torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated)
try:
    torch.cuda.mem_get_info = lambda: (8 * 2**30, 12 * 2**30)
    torch.cuda.memory_reserved = lambda: 4 * 2**30
    torch.cuda.memory_allocated = lambda: 4 * 2**30
    model = object()
    engine = InferenceEngine(64, 8, reserve_mib=0)
    tokens = tuple(range(2000))
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=torch.zeros(1,4,128,16),
        values=torch.zeros(1,4,128,16), conv_states=torch.zeros(1,32,4), recurrent_states=torch.zeros(1,4,16,16))])
    chunks = [torch.zeros(1,128,32)]
    assert engine._retain(model, tokens, 128, 'image-A', False, 256, cache, chunks)
    assert engine._retain(model, tokens, 1024, 'image-A', False, 256, cache, chunks)
    key, common = engine._match(model, tokens, 1800, 'image-A', False, 256, 512)
    assert len(engine.entries[key]['ids']) == 1024 and common == 1024
    key, common = engine._match(model, tokens, 1800, 'image-B', False, 256, 512)
    assert len(engine.entries[key]['ids']) == 128 and common == 256, 'Changed media reused image state'
    key, common = engine._match(model, tokens, 1800, 'image-A', True, 256, 512)
    assert len(engine.entries[key]['ids']) == 128, 'Changed pooling reused image state'
    diverged = tokens[:700] + (-1,) + tokens[701:]
    key, common = engine._match(model, diverged, 1800, 'image-A', False, 256, 512)
    assert len(engine.entries[key]['ids']) == 128 and common == 700
    assert engine._match(object(), tokens, 1800, 'image-A', False, 256, 512) == (None,0)
    seen_key=engine._input_key(model,tokens,1800,'image-A',False,256)
    engine._remember(seen_key,tokens,1800,'image-A',False)
    assert seen_key in engine.observations
    # CPU observations can discover an uncached junction, but cannot impersonate
    # a GPU checkpoint or reuse media from another image.
    key,common=engine._match(model,diverged,1800,'image-A',False,256,512)
    assert common==700 and len(engine.entries[key]['ids'])==128
    parent = engine.entries[key]['cache']
    cache.layers[0].recurrent_states.fill_(42)
    assert not parent.layers[0].recurrent_states.any(), 'Source modified saved parent'
    engine.max_entries = 1
    engine._trim()
    assert len(engine.entries) == 1 and engine.evictions == 1
    engine.max_bytes = 0
    engine._trim()
    assert not engine.entries
    key,common=engine._match(model,tokens,1800,'image-A',False,256,512)
    assert key is None and common==1800
    engine.clear()
    assert not engine.observations
    auto = InferenceEngine('auto', reserve_mib=1024, utilization=.9)
    auto.base_bytes = 4 * 2**30
    assert abs(auto._budget()/2**30 - 5.8) < .001
    auto.workspace_bytes = 2 * 2**30
    assert abs(auto._budget()/2**30 - 4.8) < .001
finally:
    torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated = original
print('PASS: longest exact ancestry, divergent prefixes, image/pooling isolation, model identity, independent recurrent states, eviction and automatic budgets')
