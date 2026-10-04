"""Measure cache accounting on CPU fixtures without loading a model or using CUDA."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'vendor/cloudflare'), str(root / 'runtime')]
import torch
from optimized_inference import InferenceEngine, tensor_bytes

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path)
p.add_argument('--entries', type=int, default=8)
p.add_argument('--tokens', type=int, default=24000)
p.add_argument('--repeats', type=int, default=25)
a = p.parse_args()
assert a.entries > 0 and a.tokens > 0 and a.repeats > 0
engine = InferenceEngine(8, max_entries=a.entries)
shared = torch.zeros(16)
for i in range(a.entries):
    cache = SimpleNamespace(layers=[SimpleNamespace(keys=torch.zeros(4), values=torch.zeros(4),
        conv_states=torch.zeros(4), recurrent_states=torch.zeros(4)) for _ in range(64)])
    engine.entries[i] = {'cache': cache, 'chunks': (shared, torch.zeros(16)),
                         'ids': tuple(range(a.tokens)), 'media_key': ('fixture', False)}
original = lambda: tensor_bytes([list(engine.entries.values()), list(engine.features.values())])
expected = original()
assert engine._bytes() == expected
rows = []
for name, function in [('original_snapshot_walk', original), ('tensor_fields_only', engine._bytes)]:
    for _ in range(3):
        assert function() == expected
    samples = []
    for _ in range(a.repeats):
        tick = time.perf_counter()
        value = function()
        samples.append((time.perf_counter() - tick) * 1000)
        assert value == expected
    rows.append({'method': name, 'median_ms': statistics.median(samples),
                 'mean_ms': statistics.mean(samples), 'samples_ms': samples})
result = {'scope': 'CPU-only synthetic 64-layer states with shared hidden storage. '
          'Tests Python accounting overhead, not end-to-end inference speed.',
          'entries': a.entries, 'tokens_per_entry': a.tokens, 'tensor_bytes': expected,
          'identical_byte_totals': True, 'rows': rows,
          'speedup': rows[0]['median_ms'] / rows[1]['median_ms']}
if a.output:
    a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({**result, 'rows': [{k: v for k, v in r.items() if k != 'samples_ms'} for r in rows]}, indent=2))
