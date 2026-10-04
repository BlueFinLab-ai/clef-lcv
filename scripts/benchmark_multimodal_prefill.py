"""Compare mixed image/text chunking with native single-pass inference.

Private photographs are read from --images-dir and never embedded in results.
Run on an isolated GPU; this script does not stop or modify serving processes.
"""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile', choices=['full', 'flash'], default='flash')
p.add_argument('--data-dir', type=Path, required=True)
p.add_argument('--images-dir', type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mode', choices=['reference', 'capacity'], default='reference')
p.add_argument('--reference', type=Path)
p.add_argument('--resume', action='store_true')
a = p.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, CLEF_DATA_DIR=str(a.data_dir),
    CLEF_MODEL_DIR='model-nf4' if a.profile == 'full' else 'model-nf4-compact',
    CLEF_REQUIRE_FAST_LINEAR_ATTENTION='1' if a.profile == 'full' else '0',
    CLEF_ATTENTION_BACKEND='efficient', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    PYTORCH_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='4')
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'builds' / a.profile))
app = importlib.import_module('clef_app')
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from PIL import Image, ImageOps
from chunked_prefill import forward_chunked
from optimized_inference import InferenceEngine, encode_compact, lexical_weight, pool_record, answers

model, processor, _ = app.load_model()
engine = InferenceEngine(0, feature_cache_mib=0)
engine.initialize_memory_budget()
photos = {}
for name in ('IMG_3350.jpeg', 'IMG_1220.jpeg'):
    with Image.open(a.images_dir / name) as image:
        photos[name] = ImageOps.exif_transpose(image).convert('RGB')
questions = {
 'scene': {'type': 'choice', 'instructions': 'Which setting best describes the FIRST photograph?',
    'criteria': {'beach': 'Outdoor beach with sand and water.', 'indoors': 'Indoor portrait with a wall background.', 'street': 'Outdoor city street.'}},
 'sunglasses': {'type': 'noul', 'instructions': 'In the FIRST photograph, is the foreground person wearing sunglasses?'},
 'drink': {'type': 'noul', 'instructions': 'In the FIRST photograph, is the foreground person holding a drink?'},
 'clothing': {'type': 'choice', 'instructions': 'What is the predominant clothing color in the FIRST photograph?',
    'criteria': {'red': 'Red', 'blue': 'Blue', 'other': 'Another color'}},
 'ship': {'type': 'noul', 'instructions': 'In the FIRST photograph, is a large cruise ship visible in the background?'},
 'urgency': {'type': 'score', 'instructions': 'Rate the urgency of the text incident: checkout is down and requires immediate action.',
    'criteria': ['Can wait', 'This week', 'Today']},
}
cases = []
for name in photos:
    for fidelity in (256, 512, 1024):
        cases.append({'name': name + '-' + str(fidelity), 'images': [name], 'fidelity': fidelity})
cases.extend([
 {'name':'two-images', 'images':list(photos), 'fidelity':1024},
 {'name':'reverse-images', 'images':list(reversed(photos)), 'fidelity':1024},
 {'name':'four-images-pooled', 'images':list(photos)*2, 'fidelity':1024, 'pooling':True},
 {'name':'long-10k-mid-image', 'images':list(photos), 'fidelity':1024, 'tokens':10240, 'image_offset':7400},
 {'name':'long-24k-mid-image', 'images':list(photos), 'fidelity':1024, 'tokens':24576, 'image_offset':8000},
 {'name':'sixteen-images-24k', 'images':list(photos)*8, 'fidelity':1024, 'tokens':24576},
])
if a.profile=='full':
    cases.extend([
      {'name':'long-16k-mid-image','images':list(photos),'fidelity':1024,'tokens':16384,'image_offset':8000},
      {'name':'sixteen-images-16k','images':list(photos)*8,'fidelity':1024,'tokens':16384},
    ])


def encode(case):
    record={'state':'Checkout is down; act immediately. Inspect the photographs in their supplied order.',
            'images':[photos[name] for name in case['images']], 'questions':questions,
            'media_kwargs':{'images_kwargs':{'min_pixels':min(65536,case['fidelity']**2), 'max_pixels':case['fidelity']**2}}}
    def prepare():
        encoded, boundary = encode_compact(processor, record)
        if case.get('pooling'):
            encoded, boundary = pool_record(encoded, boundary, model.language_model.config.image_token_id)
        return encoded, boundary
    if case.get('image_offset'):
        record['context']='word '
        encoded, boundary=prepare()
        offset=encoded.media['token_offset']
        record['context']='word '*(1+case['image_offset']-offset)
    encoded, boundary=prepare()
    if case.get('tokens'):
        original_state=record['state']
        words=case['tokens']-len(encoded.input_ids)
        assert words >= 0, (case['name'], 'fixture is larger than target')
        for _ in range(4):
            record['state']='word '*words + original_state
            encoded, boundary=prepare()
            if len(encoded.input_ids)==case['tokens']:
                break
            words += case['tokens']-len(encoded.input_ids)
        assert len(encoded.input_ids)==case['tokens'], (case['name'],len(encoded.input_ids))
    if case.get('image_offset'):
        assert encoded.media['token_offset']==case['image_offset']
    return record,encoded,boundary


def clean():
    engine.clear(); gc.collect(); torch.cuda.empty_cache()

@torch.inference_mode()
def run(record,encoded,boundary,policy,pooling=False):
    clean(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    tick=time.perf_counter()
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            if policy:
                logits,usage=forward_chunked(model,processor,encoded,lexical_weight(model),
                    initial_chunk_tokens=policy[0],chunk_tokens=policy[1],pooling=pooling)
            else:
                logits,usage=engine.forward(model,processor,encoded,boundary,'benchmark',
                    cache_enabled=False,feature_cache_enabled=False,pooling=pooling)
        torch.cuda.synchronize()
        result={'passed':all(torch.isfinite(x).all().item() for x in logits),
                'answers':answers(record,encoded,logits),
                'probabilities':[x.float().softmax(-1).cpu().tolist() for x in logits],
                'usage':usage}
        result['selected']=[max(range(len(v)),key=v.__getitem__) for v in result['probabilities']]
        del logits
    except torch.cuda.OutOfMemoryError as exc:
        exc.__traceback__=None
        result={'passed':False,'error':'CUDA out of memory'}
    result.update(seconds=time.perf_counter()-tick,tokens=len(encoded.input_ids),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    clean()
    return result

out={'profile':a.profile,'gpu':torch.cuda.get_device_name(),'gpu_uuid':a.gpu,
     'mode':a.mode,'base_allocated_mib':torch.cuda.memory_allocated()/2**20,
     'cases':[], 'completed':False,
     'image_sha256':{n:hashlib.sha256((a.images_dir/n).read_bytes()).hexdigest() for n in photos}}
if a.resume and a.output.exists():
    out=json.loads(a.output.read_text())
    out['completed']=False
reference = {x['case']['name']:x for x in json.loads(a.reference.read_text())['cases']} if a.reference else {}
def save():
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
def compare(trial, baseline):
    if not trial['passed'] or not baseline['passed']:
        return
    trial['selected_match']=trial['selected']==baseline['selected']
    trial['max_probability_delta_pp']=max(abs(v-r)*100 for x,y in zip(trial['probabilities'],baseline['probabilities']) for v,r in zip(x,y))
    trial['equivalent']=trial['selected_match'] and trial['max_probability_delta_pp']<=1.0

for case in cases:
    existing=next((x for x in out['cases'] if x['case']['name']==case['name']),None)
    if existing and existing['case']==case and len(existing.get('chunked',[]))==(2 if existing['tokens']<8192 else 1) and all(t.get('passed') for t in existing['chunked']):
        continue
    if existing:
        out['cases'].remove(existing)
    record,encoded,boundary=encode(case)
    entry={'case':case,'tokens':len(encoded.input_ids),
           'input_ids_sha256':hashlib.sha256(json.dumps(encoded.input_ids).encode()).hexdigest(),
           'image_tokens':sum(x==model.language_model.config.image_token_id for x in encoded.input_ids),
           'chunked':[]}
    out['cases'].append(entry)
    if a.mode=='reference':
        entry['single_pass']=run(record,encoded,boundary,None,case.get('pooling',False))
        assert entry['single_pass']['passed'],case['name']
    else:
        assert entry['input_ids_sha256']==reference[case['name']]['input_ids_sha256']
        entry['single_pass']=reference[case['name']]['single_pass']
        # Fresh native reference at a length which fits the 8 GB target.
        if len(encoded.input_ids)<=8192:
            entry['local_single_pass']=run(record,encoded,boundary,None,case.get('pooling',False))
    policies=[(64,128),(512,512)] if len(encoded.input_ids)<8192 else [(8192,4096)]
    for policy in policies:
        trial=run(record,encoded,boundary,policy,case.get('pooling',False))
        trial['policy']=policy
        compare(trial,entry['single_pass'])
        if 'local_single_pass' in entry:
            compare(trial,entry['local_single_pass'])
        entry['chunked'].append(trial)
        save()
        print(case['name'],{k:v for k,v in trial.items() if k not in ('answers','probabilities')},flush=True)
        if not trial['passed']:
            break
out['completed']=True
out['all_equivalent']=all(t.get('equivalent',False) for c in out['cases'] for t in c['chunked'])
out['mid_image_split_tested']=any(t.get('usage',{}).get('image_chunk_splits',0)>0 for c in out['cases'] for t in c['chunked'])
save();print('COMPLETE',a.profile,a.mode,'equivalent',out['all_equivalent'],flush=True)
