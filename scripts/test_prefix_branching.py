"""GPU validation of shared context, divergent branches, media identity and eviction.

Synthetic inputs only. Run in the service GPU environment with CLEF_DATA_DIR set.
"""
import argparse
import copy
import importlib
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('profile', choices=['full', 'flash'])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'builds' / args.profile))
app = importlib.import_module('clef_app')
from optimized_inference import encode_compact, pool_record, remap_checkpoints, InferenceEngine, tensor_bytes
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

model, processor, _ = app.load_model()
engine = InferenceEngine()
questions = {'color': {'type': 'choice', 'instructions': 'What is the stated or visible color?',
    'criteria': {'red': 'Red', 'blue': 'Blue', 'other': 'Another color'}},
    'bright': {'type': 'noul', 'instructions': 'Is this brightly lit?'},
    'detail': {'type': 'score', 'instructions': 'Rate the available detail.', 'criteria': ['Little', 'Some', 'A lot']}}
context = 'Use the supplied evidence. Evaluate each question independently. ' * 180

def prepare(record, pooling=False):
    encoded, boundary, points = encode_compact(processor, record, with_checkpoints=True)
    if pooling:
        original = encoded
        encoded, boundary = pool_record(encoded, boundary, model.language_model.config.image_token_id)
        points = remap_checkpoints(points, original, encoded)
    return encoded, boundary, points

def run(record, media_key='text', pooling=False, expect_hit=None, expect_vision=None):
    encoded, boundary, points = prepare(record, pooling)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        baseline, _ = engine.forward(model, processor, encoded, boundary, media_key, pooling, False)
        actual, usage = engine.forward(model, processor, encoded, boundary, media_key, pooling, True, points)
    delta = max(float((a.float().softmax(-1)-b.float().softmax(-1)).abs().max()) for a,b in zip(actual, baseline))
    assert delta < .035, (usage, delta)
    near_ties = 0
    for a,b in zip(actual, baseline):
        if a.argmax() != b.argmax():
            probabilities = b.float().softmax(-1).sort(descending=True).values
            margin = float(probabilities[0]-probabilities[1])
            assert margin < 2 * delta, (usage, margin, delta)
            near_ties += 1
    if expect_hit is not None:
        assert (usage['reused_prefix_tokens'] > 0) == expect_hit, usage
    if expect_vision is not None:
        assert usage['vision_prefix_reused'] == expect_vision, usage
    assert engine._bytes() <= engine.max_bytes, engine.stats()
    print({'check': run.count, 'usage': usage, 'max_probability_delta': delta, 'near_tie_choice_changes': near_ties, 'cache': engine.stats()}, flush=True)
    run.count += 1
    return usage
run.count = 0

record = {'context': context, 'state': 'This is a bright red alert.', 'questions': questions}
run(record, expect_hit=False)
run({**record, 'state': 'This is a bright blue alert.'}, expect_hit=True)
run(record, expect_hit=True)
snapshot = {k: copy.deepcopy(v['cache']) for k,v in engine.entries.items()}
run({**record, 'questions': dict(reversed(list(questions.items())))}, expect_hit=True)
def tensors(value):
    if isinstance(value, torch.Tensor): return [value]
    if isinstance(value, dict): return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (tuple,list)): return [t for v in value for t in tensors(v)]
    return tensors(vars(value)) if hasattr(value, '__dict__') else []
for key, old in snapshot.items():
    if key in engine.entries:
        assert all(torch.equal(a,b) for a,b in zip(tensors(old), tensors(engine.entries[key]['cache']))), 'Saved parent mutated'
del snapshot
images = [Image.new('RGB', (512,384), 'red'), Image.new('RGB', (384,512), 'blue')]
for pooling in [False, True]:
    image_record = {**record, 'state': 'Inspect these images.', 'images': images,
        'media_kwargs': {'images_kwargs': {'min_pixels': 65536, 'max_pixels': 65536}}}
    run(image_record, 'red-blue', pooling, expect_hit=True, expect_vision=False)
    run({**image_record, 'state': 'Inspect the lighting in these images.'}, 'red-blue', pooling, expect_hit=True, expect_vision=True)
    run({**image_record, 'images': list(reversed(images))}, 'blue-red', pooling, expect_hit=True, expect_vision=False)
    run(image_record, 'red-blue', pooling, expect_hit=True, expect_vision=True)
# Automatic discovery without an explicit context field.
engine.clear()
auto = {'state': context + '\nUnique request: red.', 'questions': questions}
run(auto, expect_hit=False)
run({**auto, 'state': context + '\nUnique request: blue.'}, expect_hit=False)
third = run({**auto, 'state': context + '\nUnique request: another.'}, expect_hit=True)
assert third['reused_prefix_tokens'] > 1024, third
# Force eviction while preserving correctness; then simulate a zero cache budget.
engine.max_entries = 2
for i in range(3): run({**record, 'context': f'Independent guide {i}. ' * 160})
assert len(engine.entries) <= 2 and engine.evictions > 0
engine.auto_budget = False
engine.max_bytes = 0
run(record, expect_hit=False)
assert not engine.entries
print(f'PASS: {run.count} branch checks; exact media identity, immutable parents, all answer types, pooling, adaptive boundaries and eviction', flush=True)
