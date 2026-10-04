"""Check Full 27B known-answer retrieval at increasing text and image lengths.

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
p.add_argument('--profile', choices=['full'], default='full')
p.add_argument('--data-dir', type=Path, required=True)
p.add_argument('--images-dir', type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--output', type=Path, required=True)
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



# Known-answer retrieval and image checks across total encoded lengths.
questions={}
out={'profile':a.profile,'gpu_uuid':a.gpu,'completed':False,'cases':[]}
def save(): a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
def compare(t,r):
    if not t.get('passed') or not r.get('passed'): return
    t['selected_match']=t['selected']==r['selected']
    t['max_probability_delta_pp']=max(abs(x-y)*100 for xs,ys in zip(t['probabilities'],r['probabilities']) for x,y in zip(xs,ys))

def fixture(tokens,with_images,variant=0,image_count=2):
    colors=['RED','GREEN','BLUE'] if variant==0 else ['BLUE','RED','GREEN']
    criteria={c.lower():c.title() for c in ['RED','GREEN','BLUE']}
    qs={pos:{'type':'choice','instructions':f'What color is explicitly assigned to the {pos.upper()} ALERT COLOR marker? Use only that named marker.', 'criteria':criteria} for pos in ('start','middle','end')}
    truth=dict(zip(('start','middle','end'),[c.lower() for c in colors]))
    record={'questions':qs,'state':f'END ALERT COLOR: {colors[2]}. This is the final authoritative end marker.'}
    if with_images:
        order=list(photos) if variant==0 else list(reversed(photos))
        record.update(images=[photos[n] for n in (order*(image_count//2))],media_kwargs={'images_kwargs':{'min_pixels':65536,'max_pixels':1048576}})
        qs.update(scene={'type':'choice','instructions':'What is the setting of the FIRST photograph?','criteria':{'beach':'Beach with sand and water','indoors':'Indoor portrait against a wall','street':'City street'}},clothing={'type':'choice','instructions':'What is the predominant clothing color of the foreground person in the FIRST photograph?','criteria':{'red':'Red','blue':'Blue','other':'Other'}},ship={'type':'choice','instructions':'Is a large cruise ship visible in the FIRST photograph?','criteria':{'yes':'Yes','no':'No'}})
        truth.update(scene='beach' if variant==0 else 'indoors',clothing='red' if variant==0 else 'blue',ship='yes' if variant==0 else 'no')
    departments=['packing','shipping','receiving','maintenance','accounts','scheduling','inventory']
    def context(n,pad=0):
        paragraphs=[f'Log entry {i+1}: The {departments[i%7]} team reviewed station {(i*13)%97+1}. They recorded {(i*7)%31+1} routine checks, filed the report, and scheduled the next inspection. No exceptional action was requested.\n' for i in range(n)]
        half=n//2
        return f'START ALERT COLOR: {colors[0]}. This is the authoritative start marker.\n'+''.join(paragraphs[:half])+f'\nMIDDLE ALERT COLOR: {colors[1]}. This is the authoritative middle marker.\n'+''.join(paragraphs[half:])+' routine'*pad
    lo,hi=0,tokens//20
    while lo<hi:
        n=(lo+hi+1)//2;record['context']=context(n)
        enc,b=encode_compact(processor,record)
        if len(enc.input_ids)<=tokens:lo=n
        else:hi=n-1
    record['context']=context(lo);enc,b=encode_compact(processor,record);pad=tokens-len(enc.input_ids)
    assert pad>=0
    for _ in range(5):
        record['context']=context(lo,pad);enc,b=encode_compact(processor,record)
        if len(enc.input_ids)==tokens:break
        pad+=tokens-len(enc.input_ids)
    assert len(enc.input_ids)==tokens
    return record,enc,b,truth

suite=[(n,images,0,2) for n in (8192,24576,32768,45056) for images in (False,True)]
suite += [(45056,images,1,2) for images in (False,True)]
suite += [(45056,True,0,16)]
for tokens,with_images,variant,image_count in suite:
    record,encoded,boundary,truth=fixture(tokens,with_images,variant,image_count)
    entry={'tokens':tokens,'images':image_count if with_images else 0,'variant':variant,'truth':truth,'input_ids_sha256':hashlib.sha256(json.dumps(encoded.input_ids).encode()).hexdigest(),'trials':[]};out['cases'].append(entry)
    policies=[None,(8192,4096),(4096,4096)] if tokens<=32768 else [(8192,4096),(4096,4096)]
    for policy in policies:
        trial=run(record,encoded,boundary,policy)
        trial['policy']=policy
        if trial.get('passed'):
            trial['correct']={name:trial['answers'][name]['choice']==expected for name,expected in truth.items()}
            trial['all_correct']=all(trial['correct'].values())
        if entry['trials']:compare(trial,next((r for r in entry['trials'] if r.get('passed')),entry['trials'][0]))
        entry['trials'].append(trial);save()
        print('CASE',tokens,entry['images'],variant,'policy',policy,{k:v for k,v in trial.items() if k not in ('answers','probabilities','usage')},flush=True)
out['completed']=True;save();print('COMPLETE',flush=True)
