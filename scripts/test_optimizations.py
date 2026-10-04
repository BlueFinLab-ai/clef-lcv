"""GPU integration checks for the optimized native decision path.

Run with the same environment/data directory as the service. No private inputs
are required. Optional --image paths are used locally and never saved.
"""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('profile', choices=['full', 'flash'])
parser.add_argument('--image', action='append', default=[])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'builds' / args.profile))
app = importlib.import_module('clef_app')
from optimized_inference import encode_compact, pool_record, InferenceEngine, answers
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

model, processor, _ = app.load_model()
engine = InferenceEngine(256, 1)
questions = {'color': {'type': 'choice', 'instructions': 'What is the predominant visible color?',
    'criteria': {'red': 'Red', 'blue': 'Blue', 'other': 'Another color'}},
    'bright': {'type': 'noul', 'instructions': 'Is this brightly lit?'},
    'detail': {'type': 'score', 'instructions': 'How much visual detail is present?', 'criteria': ['Little', 'Some', 'A lot']}}
images = [Image.new('RGB', (512, 512), 'red'), Image.new('RGB', (384, 512), 'blue')]
for path in args.image:
    with Image.open(path) as image:
        images.append(image.convert('RGB'))
checks = 0
for count, side, pooling in [(0, 256, False), (1, 256, False), (4, 256, True), (1, 1024, True), (1, 1024, False)]:
    supplied = (images * 4)[:count]
    if args.image and count == 1 and side == 1024:
        supplied = [images[2]]
    record = {'state': 'Inspect these images.' if count else 'The message is a bright red alert.', 'questions': questions}
    if count:
        record.update(images=supplied, media_kwargs={'images_kwargs': {'min_pixels': min(65536, side**2), 'max_pixels': side**2}})
    encoded, boundary = encode_compact(processor, record)
    if pooling:
        encoded, boundary = pool_record(encoded, boundary, model.language_model.config.image_token_id)
    assert all(encoded.input_ids[s:e] for q in encoded.questions for s,e in q.option_spans)
    key = hashlib.sha256(str((count, side, pooling)).encode()).hexdigest()
    engine.clear()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        fresh, _ = engine.forward(model, processor, encoded, boundary, key, pooling, False)
        cold, cold_usage = engine.forward(model, processor, encoded, boundary, key, pooling, True)
        warm, usage = engine.forward(model, processor, encoded, boundary, key, pooling, True)
    def compare(left, right):
        delta = max(float((a.float().softmax(-1)-b.float().softmax(-1)).abs().max()) for a,b in zip(left,right))
        assert delta < .03, delta
        for a,b in zip(left,right):
            assert a.argmax() == b.argmax(), 'Different selected option'
        return delta
    delta = max(compare(fresh, cold), compare(fresh, warm))
    assert usage['prefix_cache'] == 'hit', (usage, engine.stats())
    changed = {**record, 'questions': {'new': {'type': 'noul', 'instructions': 'Is there visible text?'}, **dict(reversed(list(questions.items())))}}
    next_encoded, next_boundary = encode_compact(processor, changed)
    if pooling:
        next_encoded, next_boundary = pool_record(next_encoded, next_boundary, model.language_model.config.image_token_id)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual, next_usage = engine.forward(model, processor, next_encoded, next_boundary, key, pooling, True)
        reference, _ = engine.forward(model, processor, next_encoded, next_boundary, key, pooling, False)
    assert next_usage['prefix_cache'] == 'hit'
    delta = max(delta, compare(actual, reference))
    print({'images': count, 'side': side, 'pooling': pooling, 'tokens': len(encoded.input_ids), 'max_probability_delta': delta, 'cache': engine.stats()}, flush=True)
    checks += 1
print(f'PASS: {checks} fresh/cold/warm/changed-question cases, all native answer types', flush=True)
