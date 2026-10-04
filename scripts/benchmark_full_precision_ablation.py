"""Isolate Full 27B vision batching and language chunking with frozen features.

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


# Controlled feature reuse separates vision batch effects from language chunking.
from dataclasses import replace
import joint_schema_model as joint
import chunked_prefill as cp
base=model.language_model.model
device=next(model.parameters()).device
real_features=cp.image_features
out={'profile':a.profile,'gpu_uuid':a.gpu,'completed':False,'cases':[]}
def save(): a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
def compare(trial,reference):
    if not trial.get('passed') or not reference.get('passed'): return
    trial['selected_match']=trial['selected']==reference['selected']
    deltas=[[abs(x-y)*100 for x,y in zip(xs,ys)] for xs,ys in zip(trial['probabilities'],reference['probabilities'])]
    trial['question_deltas_pp']=dict(zip(questions,[max(x) for x in deltas]))
    trial['max_probability_delta_pp']=max(map(max,deltas))

def manual(record,encoded,frozen,policy):
    clean(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick=time.perf_counter()
    try:
        with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            if policy:
                cp.image_features=lambda *args,**kwargs:frozen.to(device)
                logits,usage=cp.forward_chunked(model,processor,encoded,lexical_weight(model),initial_chunk_tokens=policy[0],chunk_tokens=policy[1])
            else:
                rec=replace(encoded,media={k:v for k,v in encoded.media.items() if k!='pixel_values'})
                batch=joint.collate_records([rec],processor.tokenizer.pad_token_id,device)
                ids,mask,media=batch['input_ids'],batch['attention_mask'],batch['media']
                positions,_=base.get_rope_index(ids,image_grid_thw=media['image_grid_thw'],attention_mask=mask,mm_token_type_ids=media['mm_token_type_ids'])
                embeds=base.get_input_embeddings()(ids)
                embeds=embeds.masked_scatter((ids==model.language_model.config.image_token_id).unsqueeze(-1).expand_as(embeds),frozen.to(device,dtype=embeds.dtype))
                output=base.language_model(inputs_embeds=embeds,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True)
                logits=model.head(output.last_hidden_state,ids,mask,[encoded],lexical_weight(model))[0]
                usage={'prefill_mode':'manual_single_pass'}
            torch.cuda.synchronize()
            result={'passed':all(torch.isfinite(x).all().item() for x in logits),'answers':answers(record,encoded,logits),'probabilities':[x.float().softmax(-1).cpu().tolist() for x in logits],'usage':usage}
            result['selected']=[max(range(len(v)),key=v.__getitem__) for v in result['probabilities']]
    except torch.cuda.OutOfMemoryError as exc:
        exc.__traceback__=None; result={'passed':False,'error':'CUDA out of memory'}
    finally: cp.image_features=real_features
    result.update(seconds=time.perf_counter()-tick,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    return result

case=next(c for c in cases if c['name']=='long-24k-mid-image')
record,encoded,boundary=encode(case)
entry={'case':case,'tokens':len(encoded.input_ids),'trials':[]};out['cases'].append(entry)
# Freeze both feature variants on CPU. Identical copies feed every language comparison.
with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    all_features=real_features(base,encoded,device,batch_images=0).cpu()
    one_features=real_features(base,encoded,device,batch_images=1).cpu()
    repeat_features=real_features(base,encoded,device,batch_images=0).cpu()
entry['feature_comparison']={'shape':list(all_features.shape),'batch_vs_single_max_abs':(all_features.float()-one_features.float()).abs().max().item(),'batch_vs_single_rms':(all_features.float()-one_features.float()).square().mean().sqrt().item(),'batch_repeat_max_abs':(all_features.float()-repeat_features.float()).abs().max().item(),'batch_vs_single_exact_fraction':(all_features==one_features).float().mean().item()}
print('FEATURES',entry['feature_comparison'],flush=True);save()
for name,features,policy in [('native-1',None,None),('native-2',None,None),('native-3',None,None),('batch-features-whole',all_features,None),('single-features-whole',one_features,None),('batch-features-8k4k',all_features,(8192,4096)),('single-features-8k4k',one_features,(8192,4096)),('single-features-8k4k-repeat',one_features,(8192,4096)),('batch-features-4k4k',all_features,(4096,4096)),('batch-features-8k2k',all_features,(8192,2048))]:
    trial=run(record,encoded,boundary,None) if features is None else manual(record,encoded,features,policy)
    trial['name']=name
    if entry['trials']: compare(trial,entry['trials'][0])
    entry['trials'].append(trial);save()
    print(name,{k:v for k,v in trial.items() if k not in ('answers','probabilities','usage')},flush=True)
out['completed']=True;save();print('COMPLETE',flush=True)
