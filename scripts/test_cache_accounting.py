"""CPU checks for exact cache storage totals, metadata exclusion and eviction."""
from pathlib import Path
from types import SimpleNamespace
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'vendor/cloudflare'), str(root / 'runtime')]
import torch
from optimized_inference import InferenceEngine, tensor_bytes


def original_total(engine):
    # Reference the complete retained objects, including CPU token metadata.
    return tensor_bytes([list(engine.entries.values()), list(engine.features.values())])


def consistent(engine):
    assert engine._bytes() == original_total(engine)


storage = torch.arange(256, dtype=torch.float32)
hidden = torch.zeros(128, dtype=torch.float32)
suffix = torch.zeros(64, dtype=torch.float32)
parent = SimpleNamespace(layers=[SimpleNamespace(keys=storage[:128], values=storage[128:])])
child = SimpleNamespace(layers=[SimpleNamespace(keys=storage.clone(), values=torch.ones(32))])
engine = InferenceEngine(8, reserve_mib=0, elastic=True)
engine.entries['parent'] = {'cache': parent, 'chunks': (hidden[:64],),
                            'ids': tuple(range(24000)), 'media_key': ('image-a', False), 'bytes': -1}
engine.entries['child'] = {'cache': child, 'chunks': (hidden[64:], suffix),
                           'ids': tuple(range(48000)), 'media_key': ('image-b', True), 'bytes': -1}
engine.features['shared'] = hidden[:16]
engine.features['separate'] = suffix.clone()
consistent(engine)
assert engine._bytes() == 1024 + 1024 + 128 + 512 + 256 + 256

# Token ancestry and other non-tensor snapshot fields must never be inspected
# by VRAM accounting, regardless of their length or content.
class ForbiddenMetadata:
    @property
    def __dict__(self):
        raise AssertionError('GPU accounting inspected CPU metadata')

for entry in engine.entries.values():
    entry['ids'] = ForbiddenMetadata()
    entry['media_key'] = ForbiddenMetadata()
    entry['bytes'] = ForbiddenMetadata()
assert engine._bytes() == 3200
assert engine.stats()['entries'] == 2
for entry in engine.entries.values():
    entry.update(ids=tuple(range(1000)), media_key=('image', False), bytes=0)
consistent(engine)

original = (torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated)
try:
    torch.cuda.mem_get_info = lambda: (4 * 2**30, 4 * 2**30)
    torch.cuda.memory_reserved = lambda: 0
    torch.cuda.memory_allocated = lambda: 0
    engine.max_bytes = 2176
    engine._trim(protected='child')
    assert 'parent' not in engine.entries and 'child' in engine.entries
    assert engine._bytes() == 2176 and engine.evictions == 1
    consistent(engine)
    # The shared hidden storage survives in the feature after its checkpoint
    # is evicted; storage sizes, rather than view sizes, remain authoritative.
    assert engine._drop()
    assert engine._bytes() == 512 + 256
    consistent(engine)
    assert engine._drop()
    assert engine._bytes() == 256
    consistent(engine)
    engine.max_bytes = 0
    engine._trim()
    assert engine._bytes() == 0 and not engine.features

    # Actual branch retention clones mutable state but shares immutable chunks.
    model = object()
    engine = InferenceEngine(8, reserve_mib=0)
    tokens = tuple(range(24000))
    assert engine._retain(model, tokens, 1024, 'text', False, None, parent, (hidden,))
    assert engine._retain(model, tokens, 2048, 'text', False, None, parent, (hidden, suffix))
    consistent(engine)
    for entry in engine.entries.values():
        assert entry['bytes'] == tensor_bytes((entry['cache'], entry['chunks']))
    assert engine._drop()
    consistent(engine)
    engine.clear()
    assert engine._bytes() == 0 and not engine.observations
finally:
    torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated = original

print('PASS: exact unique-storage totals, shared views/branches/features, no metadata walk, '
      'protected eviction, feature eviction, cloned retention and clear')
