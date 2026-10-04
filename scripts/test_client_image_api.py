"""Validate client-sized PNGs and upstream media_kwargs on a running Clef service.

Requires private browser-prepared fixtures named client-WIDTHxHEIGHT-0.png.
The fixture is the beach/ship photo; only scene/object questions are tested.
"""
import argparse
import base64
import copy
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', required=True)
p.add_argument('--images-dir', type=Path, required=True)
p.add_argument('--original', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()


def request(path, body=None):
    req = urllib.request.Request(a.url + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


deadline = time.monotonic() + 240
while True:
    try:
        health = request('/health')
        if health['image_input'].get('resize_location') == 'client_for_gui':
            break
    except OSError:
        pass
    if time.monotonic() > deadline:
        raise RuntimeError('Updated service did not become ready')
    time.sleep(2)
model = request('/v1/models')['data'][0]['id']
questions = {'ship': {'type': 'noul', 'instructions': 'Is a cruise ship visible in the background?'},
             'clothing': {'type': 'choice', 'instructions': 'Which color predominates in the foreground clothing?',
                          'criteria': {'red': 'Red', 'blue': 'Blue', 'other': 'Another color'}}}
base = {'model': model, 'state': 'Inspect the supplied image. ' + 'Shared context. ' * 80,
        'questions': questions}
result = {'completed': False, 'url': a.url, 'health': health, 'trials': []}


def data_url(path):
    mime = 'image/png' if path.suffix == '.png' else 'image/jpeg'
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()


def check(name, body):
    started = time.perf_counter()
    response = request('/v1/systemone', body)
    for answer in response['answers'].values():
        values = [answer['noul']] if answer['type'] == 'noul' else list(answer['probabilities'].values())
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
    result['trials'].append({'name': name, 'wall_seconds': time.perf_counter() - started,
                            'request_bytes': len(json.dumps(body).encode()), **response})
    print('PASS', name, response['usage']['input_tokens'], response['usage']['latency_ms'], flush=True)
    return response


sizes = [(96, 128), (192, 288), (416, 576), (864, 1152)]
standard = None
for width, height in sizes:
    body = {**base, 'images': [data_url(a.images_dir / f'client-{width}x{height}-0.png')],
            'media_kwargs': {'images_kwargs': {'do_resize': False}}}
    response = check(f'client-{width}x{height}', body)
    assert response['usage']['processed_images'] == [{'width': width, 'height': height}]
    assert response['usage']['processor_resize_enabled'] is False
    if width == 192:
        standard = copy.deepcopy(body)
        warm = check('client-repeat', body)
        assert warm['usage']['prefix_cache'] == 'hit'
        assert warm['usage']['image_preprocess_cache_hits'] == 1
        assert warm['answers'] == response['answers']
        assert warm['answers']['ship']['noul'] > .5 and warm['answers']['clothing']['choice'] == 'red'
pooled = {**standard, 'image_pooling': True}
response = check('client-pooled', pooled)
assert response['usage']['processed_images'] == [{'width': 192, 'height': 288}]
assert response['usage']['image_pooling'] is True
assert response['usage']['unpooled_input_tokens'] > response['usage']['input_tokens']

original = {**base, 'images': [data_url(a.original)]}
legacy = check('legacy-explicit-standard', {**original, 'image_fidelity': 'standard'})
native = check('native-explicit-budget', {**original,
    'media_kwargs': {'images_kwargs': {'min_pixels': 65536, 'max_pixels': 65536}}})
assert native['answers'] == legacy['answers']
assert native['usage']['processed_images'] == legacy['usage']['processed_images']
assert native['usage']['image_fidelity'] is None
native_default = check('native-defaults-small-image', {**base, 'images': standard['images']})
assert native_default['usage']['processor_resize_enabled'] is True
assert native_default['usage']['image_fidelity'] is None
try:
    request('/v1/systemone', {**standard, 'image_fidelity': 'standard'})
except urllib.error.HTTPError as exc:
    assert exc.code == 400
    result['conflicting_options'] = json.load(exc)
else:
    raise AssertionError('Conflicting processor/fidelity options accepted')
text = {'model': model, 'state': 'Checkout is down; act immediately.',
        'questions': {'urgent': {'type': 'noul', 'instructions': 'Does this need urgent attention?'}}}
text_response = check('text-regression', text)
assert text_response['answers']['urgent']['noul'] > .5
cap = request('/v1/models')['data'][0]['max_input_tokens']
try:
    request('/v1/systemone', {**text, 'state': 'word ' * (cap + 1000)})
except urllib.error.HTTPError as exc:
    assert exc.code == 413
    result['over_limit'] = json.load(exc)
else:
    raise AssertionError('Over-limit text accepted')
result['final_models'] = request('/v1/models')
result['completed'] = True
a.output.write_text(json.dumps(result, indent=2) + '\n')
print('COMPLETE', flush=True)
