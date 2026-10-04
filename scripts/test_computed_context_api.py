"""Live computed-context admission, busy discovery, text/vision and 413 checks.

Capture a reference before updating the service, then verify after deployment.
Fixtures are synthetic; --reference and --output store test records/results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import urllib.error
import urllib.request

from PIL import Image
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', required=True)
p.add_argument('--project', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--phase', choices=['baseline', 'verify'], required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--reference-tokens', type=int, required=True)
p.add_argument('--output', type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.project / 'vendor/cloudflare'))
sys.path.insert(0, str(a.project / 'runtime'))
from optimized_inference import encode_compact

def request(path, body=None, timeout=240):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(a.url + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

deadline = time.monotonic() + 240
while True:
    try:
        health = request('/health', timeout=3)
        if a.phase == 'baseline' or health.get('context_limit', {}).get('mode') == 'memory_estimate':
            break
    except OSError:
        pass
    if time.monotonic() >= deadline:
        raise RuntimeError('Service did not become ready')
    time.sleep(3)
model = 'clef-flash' if 'flash' in health['model'] else 'clef'
tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
processor = SimpleNamespace(tokenizer=tokenizer)
questions = {
    'start': {'type': 'choice', 'instructions': 'What alert color is explicitly stated at the START of the archive?', 'criteria': {'red': 'Red', 'blue': 'Blue'}},
    'end': {'type': 'choice', 'instructions': 'What alert color is explicitly stated at the END of the archive?', 'criteria': {'red': 'Red', 'blue': 'Blue'}},
    'urgent': {'type': 'noul', 'instructions': 'Does the final incident require immediate attention?'},
    'detail': {'type': 'score', 'instructions': 'Rate the urgency of the final incident.', 'criteria': ['Can wait', 'This week', 'Today']},
}

def record(words=0):
    return {'model': model, 'state': 'START ALERT COLOR: RED.\n' + 'word ' * words +
            '\nEND ALERT COLOR: BLUE. Checkout is down; act immediately.',
            'questions': questions, 'prefix_cache': False, 'input_cache': False}

def exact(target):
    lo, hi = 0, target
    while lo <= hi:
        mid = (lo + hi) // 2
        body = record(mid)
        count = len(encode_compact(processor, body)[0].input_ids)
        if count == target:
            return body
        if count < target:
            lo = mid + 1
        else:
            hi = mid - 1
    raise AssertionError(f'Cannot construct {target}-token input')

image = BytesIO()
Image.new('RGB', (64, 64), 'blue').save(image, format='PNG')
image_body = {'model': model, 'state': 'Inspect the supplied picture.',
              'images': ['data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()],
              'questions': {'blue': {'type': 'noul', 'instructions': 'Is the dominant image color blue?'}},
              'prefix_cache': False, 'input_cache': False}

if a.phase == 'baseline':
    cases = []
    for name, body in [('short', record()), ('long', exact(a.reference_tokens)), ('image', image_body)]:
        result = request('/v1/systemone', body)
        cases.append({'name': name, 'body': body, 'response': result})
        print('BASELINE', name, result['usage'], flush=True)
    a.reference.write_text(json.dumps(cases, indent=2) + '\n')
    sys.exit(0)

def vector(answer):
    if answer['type'] == 'noul':
        return [answer['noul']]
    return list(answer['probabilities'].values())

def selected(answer):
    if answer['type'] == 'choice':
        return answer['choice']
    if answer['type'] == 'noul':
        return answer['noul'] >= .5
    return max(answer['probabilities'], key=answer['probabilities'].get)

models = request('/v1/models')
metadata = models['data'][0]
assert metadata['max_input_tokens'] == health['max_input_tokens']
assert metadata['max_input_tokens_with_images'] == health['max_input_tokens_with_images']
assert health['optimizations']['chunked_prefill']['enabled']
assert metadata['context_limit']['computed_max_input_tokens'] == metadata['max_input_tokens']
assert not metadata['context_limit']['configured_cap_applied']
results = {'model': model, 'models': models, 'regressions': [], 'long_text': [], 'completed': False}
for case in json.loads(a.reference.read_text()):
    current = request('/v1/systemone', case['body'])
    previous = case['response']
    assert current['usage']['input_tokens'] == previous['usage']['input_tokens']
    assert all(selected(value) == selected(previous['answers'][key]) for key, value in current['answers'].items())
    drift = max(abs(x-y)*100 for key, value in current['answers'].items()
                for x,y in zip(vector(value), vector(previous['answers'][key])))
    assert drift <= 1.0, (case['name'], drift)
    mode = current['usage']['prefill_mode']
    expected_mode = 'chunked' if case['name'] == 'long' and a.reference_tokens > 8192 else 'single_pass'
    assert mode == expected_mode, (case['name'], mode, expected_mode)
    item = {'case': case['name'], 'selected_match': True, 'max_probability_delta_pp': drift, 'usage': current['usage']}
    results['regressions'].append(item)
    print('REGRESSION', json.dumps(item), flush=True)

targets = sorted({10240, metadata['max_input_tokens']} - {a.reference_tokens})
for target in targets:
    if target > metadata['max_input_tokens']:
        continue
    body = exact(target)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(request, '/v1/systemone', body)
        time.sleep(1)
        busy_metadata = request('/v1/models')['data'][0]
        assert busy_metadata['context_snapshot_status'] == 'last_idle_busy'
        assert busy_metadata['max_input_tokens'] == metadata['max_input_tokens']
        result = pending.result()
    results.setdefault('busy_discovery', []).append(busy_metadata)
    usage = result['usage']
    assert usage['input_tokens'] == target
    assert usage['prefill_mode'] == 'chunked'
    expected_chunks = 1 + (target - 8192 + 4095) // 4096
    assert usage['prefill_chunks'] == expected_chunks
    # Record answer quality separately from capacity; the known long-context
    # artifact is not a serving regression when the unchanged head reproduces it.
    facts_correct = result['answers']['start']['choice'] == 'red' and result['answers']['end']['choice'] == 'blue'
    item = {'tokens': target, 'start_end_facts_correct': facts_correct, 'usage': usage}
    results['long_text'].append(item)
    print('LONG_TEXT', json.dumps(item), flush=True)

for label, cap, images in [('text', metadata['max_input_tokens'], []),
                           ('images', metadata['max_input_tokens_with_images'], image_body['images'])]:
    body = exact(cap + 1) if label == 'text' else record()
    if label == 'images':
        body['state'] = ' x' * (cap + 1024)
    body['images'] = images
    try:
        request('/v1/systemone', body)
    except urllib.error.HTTPError as exc:
        assert exc.code == 413
        rejected = json.load(exc)
        assert rejected['detail'] == f'Request exceeds {cap} input tokens'
        assert rejected.get('max_input_tokens', cap) == cap
        assert rejected.get('input_tokens', cap+1) > cap
    else:
        raise AssertionError('Oversized input was accepted')
    results[label + '_overlength_status'] = 413

body = record(256)
body.update(prefix_cache=True, input_cache=True)
request('/v1/systemone', body)
recovery = request('/v1/systemone', body)
assert recovery['usage']['prefix_cache'] == 'hit', recovery['usage']
results['short_cache_recovery'] = recovery['usage']
results['completed'] = True
assert a.output is not None
a.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
print('PASS', model, flush=True)
