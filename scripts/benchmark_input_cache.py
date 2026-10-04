"""Validate and benchmark input reuse; optional private images never enter results.

Run on exactly one selected GPU. Results contain aggregate timings and counts.
"""
import argparse
import base64
from io import BytesIO
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('profile', choices=['full', 'flash'])
parser.add_argument('--images', nargs=2, type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'builds' / args.profile))
app = importlib.import_module('clef_app')
from optimized_inference import InferenceEngine, encode_compact, pool_record, remap_checkpoints
from reusable_inputs import InputCache
from PIL import Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

model, processor, _ = app.load_model()
def synthetic(color):
    stream = BytesIO()
    Image.new('RGB', (1280,960), color).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
images = ['data:image/jpeg;base64,' + base64.b64encode(p.read_bytes()).decode() for p in args.images] if args.images else [synthetic('red'), synthetic('blue')]
questions = {'scene': {'type': 'choice', 'instructions': 'Which scene appears in the supplied pictures?',
    'criteria': {'beach': 'At least one picture shows a beach.', 'portrait': 'A portrait only, with no beach.', 'other': 'Neither description matches.'}},
    'visible': {'type':'noul', 'instructions':'Is a person visible?'},
    'detail': {'type':'score', 'instructions':'Rate visible image detail.', 'criteria':['Little','Some','A lot']}}
context = 'Use only the visible evidence and evaluate the supplied questions independently. ' * 32
validation = []

def caches():
    cpu = InputCache(decode=app.decode_image)
    gpu = InferenceEngine()
    gpu.initialize_memory_budget()
    return cpu,gpu

@torch.inference_mode()
def run(record, cpu, gpu, inputs, prefix, pooled=False):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    before = cpu.stats()
    encoded,boundary,points = encode_compact(processor, record, with_checkpoints=True, input_cache=cpu, cache_enabled=inputs)
    preprocessing = (time.perf_counter()-tick)*1000
    if pooled:
        original = encoded
        encoded,boundary = pool_record(encoded,boundary, model.language_model.config.image_token_id)
        points = remap_checkpoints(points, original, encoded)
    media_key = __import__('hashlib').sha256(json.dumps([record['images'],record['media_kwargs']],sort_keys=True).encode()).hexdigest()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        logits,usage = gpu.forward(model, processor, encoded, boundary, media_key, pooled, prefix, points, feature_cache_enabled=inputs)
    torch.cuda.synchronize()
    after = cpu.stats()
    return [v.float().softmax(-1).cpu() for v in logits], {'wall_ms':(time.perf_counter()-tick)*1000,
        'preprocessing_ms':preprocessing, 'cpu_image_hits':after['image_hits']-before['image_hits'],
        'token_hits':after['token_hits']-before['token_hits'],**usage}

def compare(ref, actual, label):
    delta=max(float((a-b).abs().max()) for a,b in zip(ref,actual))
    changed=sum(int(a.argmax()!=b.argmax()) for a,b in zip(ref,actual))
    # NF4 vision matrix kernels can vary with image-batch shape.
    # Allow up to 5 pp, but require every selected option to remain unchanged.
    assert delta<.05,(label,delta)
    assert changed==0,(label,delta,changed)
    validation.append({'case':label,'max_probability_delta':delta,'choice_changes':changed})

rows=[]
for size in [256,512,1024]:
    record={'context':context,'state':'Inspect the pictures.','images':images,'questions':questions,
        'media_kwargs':{'images_kwargs':{'min_pixels':min(65536,size*size),'max_pixels':size*size}}}
    # Exact processor equivalence against the original batched path.
    cpu,gpu=caches()
    old,_,_=encode_compact(processor,record,with_checkpoints=True,input_cache=cpu,cache_enabled=False)
    new,_,_=encode_compact(processor,record,with_checkpoints=True,input_cache=cpu,cache_enabled=True)
    assert old.input_ids==new.input_ids and old.questions==new.questions
    for key,value in old.media.items():
        other=new.media[key]
        assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other,(key,size)
    validation.append({'case':f'processor exact {size}','max_probability_delta':0,'choice_changes':0})
    # Changed leading context and image order: prefix reuse deliberately disabled.
    records=[{**record,'context':f'Workflow {i}. '+context,'images':images if i%2==0 else images[::-1]} for i in range(3)]
    references=[]
    for rec in records:
        result,usage=run(rec,cpu,gpu,False,False)
        references.append(result)
    # Every mode warmed before measurement. Alternate order by resolution.
    for mode in (['uncached','input_reuse'] if size!=512 else ['input_reuse','uncached']):
        inputs=mode=='input_reuse'
        run(record,cpu,gpu,inputs,False)
        trials=[]
        for i,rec in enumerate(records):
            result,usage=run(rec,cpu,gpu,inputs,False)
            compare(references[i],result,f'{mode} {size} {i}')
            if inputs: assert usage['image_feature_cache_hits']==2 and usage['cpu_image_hits']==2,usage
            trials.append(usage)
        rows.append({'scenario':'changed_context_and_order','pixels':size,'mode':mode,'requests':3,
            'mean_ms':statistics.mean(t['wall_ms'] for t in trials),'mean_preprocessing_ms':statistics.mean(t['preprocessing_ms'] for t in trials),
            'trials':trials})
    prefix_reference,_=run(record,cpu,gpu,False,False)
    # Layer input caching on the existing exact-prefix cache, steady prefix.
    for mode in ['prefix_only','layered']:
        cpu2,gpu2=caches()
        inputs=mode=='layered'
        run(record,cpu2,gpu2,inputs,True)
        run(record,cpu2,gpu2,inputs,True)
        trials=[]
        for i in range(3):
            result,usage=run(record,cpu2,gpu2,inputs,True)
            compare(prefix_reference,result,f'{mode} {size} {i}')
            assert usage['vision_prefix_reused'],usage
            trials.append(usage)
        rows.append({'scenario':'identical_prefix','pixels':size,'mode':mode,'requests':3,
            'mean_ms':statistics.mean(t['wall_ms'] for t in trials),'mean_preprocessing_ms':statistics.mean(t['preprocessing_ms'] for t in trials),
            'trials':trials})
        gpu2.clear();cpu2.clear();del cpu2,gpu2
    # Partial replacement, duplicate images and changing pooling must preserve order.
    third=synthetic('green')
    for label,media,pooled in [('partial replacement',[images[0],third],False),('pool toggle',images,True),('16 duplicates',[images[0]]*16,True)] if size==256 else [('pool toggle',images,True)]:
        rec={**record,'context':'Another independent context.','images':media}
        ref,_=run(rec,cpu,gpu,False,False,pooled)
        result,usage=run(rec,cpu,gpu,True,False,pooled)
        compare(ref,result,f'{label} {size}')
        if label=='partial replacement': assert usage['image_feature_cache_hits']==1 and usage['image_feature_cache_misses']==1,usage
        if label=='pool toggle': assert usage['image_feature_cache_hits']==2,usage
        if label=='16 duplicates': assert usage['image_feature_cache_hits']==16,usage
    print({'size':size,'rows':rows[-4:],'cache':gpu.stats()},flush=True)
    gpu.clear();cpu.clear();del cpu,gpu
    torch.cuda.empty_cache()
# CPU retention limits, disabled mode and immutable metadata.
small=InputCache(max_mib=0,token_mib=0,decode=app.decode_image)
encode_compact(processor,record,input_cache=small)
assert not small.images and not small.text
out={'profile':args.profile,'gpu':torch.cuda.get_device_name(),'private_images':bool(args.images),'rows':rows,
    'validation':validation,'completed':True,'precision':'native model feature precision; unchanged weight quantization'}
args.output.write_text(json.dumps(out,indent=2,allow_nan=False))
print('PASS',len(validation),'comparisons; source images omitted from results',flush=True)
