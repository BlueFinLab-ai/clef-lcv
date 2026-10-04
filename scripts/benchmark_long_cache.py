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
p.add_argument('--profile', choices=['full', 'flash'], default='full')
p.add_argument('--data-dir', type=Path, required=True)
p.add_argument('--images-dir', type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--baseline-source',type=Path)
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


from context_budget import ContextBudget
budget=ContextBudget(model.language_model.config.text_config,dtype_bytes=2,calibrated=True,
    configured_limit=16384,image_limit=None,image_chunked=True)
engine=InferenceEngine('auto',feature_cache_mib=256)
engine.initialize_memory_budget()
old_forward=None
if a.baseline_source:
    import importlib.util
    spec=importlib.util.spec_from_file_location('clef_baseline_chunked',a.baseline_source)
    old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
    old_forward=old_module.forward_chunked
out={'completed':False,'trials':[]}
def save():a.output.write_text(json.dumps(out,indent=2)+'\n')
def encoded_case(case):
    record,encoded,boundary=encode(case)
    # The image byte/settings identities are fixed in this diagnostic, equivalent
    # to keys generated by the live CPU input cache. Repeated originals share keys.
    encoded.media['image_cache_keys']=tuple(case['images'])
    checkpoints=(encoded.media['token_offset']+len(encoded.media['mm_token_type_ids']),)
    return record,encoded,boundary,checkpoints

def trial(name,case,cache,reference=None,old=False):
    record,encoded,boundary,checkpoints=encoded_case(case)
    input_key=hashlib.sha256(json.dumps([case['images'],case['fidelity'],case.get('pooling',False)]).encode()).hexdigest()
    gc.collect();torch.cuda.empty_cache()
    engine.prepare_request(budget.workspace(len(encoded.input_ids)))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        if old:
            logits,usage=old_forward(model,processor,encoded,lexical_weight(model),
                initial_chunk_tokens=8192,chunk_tokens=4096,pooling=case.get('pooling',False))
        else:
            logits,usage=engine.forward(model,processor,encoded,boundary,input_key,
                cache_enabled=cache,feature_cache_enabled=cache,checkpoint_boundaries=checkpoints,
                pooling=case.get('pooling',False),prefill_initial_chunk_tokens=8192,prefill_chunk_tokens=4096,prefill_images=True)
        response=answers(record,encoded,logits)
        probs=[v.float().softmax(-1).cpu().tolist() for v in logits]
    torch.cuda.synchronize()
    result={'name':name,'tokens':len(encoded.input_ids),'seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'usage':usage,'answers':response,'probabilities':probs,'cache':engine.stats()}
    if reference:
        result['selected_match']=all(max(range(len(x)),key=x.__getitem__)==max(range(len(y)),key=y.__getitem__) for x,y in zip(probs,reference['probabilities']))
        result['max_probability_delta_pp']=max(abs(x-y)*100 for xs,ys in zip(probs,reference['probabilities']) for x,y in zip(xs,ys))
        assert result['selected_match'],name
        assert result['max_probability_delta_pp']<1.,(name,result['max_probability_delta_pp'])
    assert all(torch.isfinite(v).all().item() for v in logits)
    assert response['scene']['choice']=='beach' and response['clothing']['choice']=='red'
    assert all(response[k]['noul']>.5 for k in ('sunglasses','drink','ship'))
    assert max(response['urgency']['probabilities'],key=response['urgency']['probabilities'].get)=='2'
    del logits
    out['trials'].append(result);save();print(name,{k:v for k,v in result.items() if k not in ('answers','probabilities','cache')},flush=True)
    return result
case={'name':'eleven-high-12k','images':(list(photos)*6)[:11],'fidelity':1024,'tokens':12288}
oldref=trial('old-uncached-11',case,False,old=True) if old_forward else None
ref=trial('uncached-11',case,False,oldref)
engine.clear()
cold=trial('cold-11',case,True,ref)
warm1=trial('warm-11-1',case,True,cold)
warm2=trial('warm-11-2',case,True,cold)
assert warm1['usage']['prefix_cache']=='hit' and warm1['usage']['vision_prefix_reused']
assert warm1['usage']['reused_prefix_tokens']>8192
# Changing only questions may reuse inputs, but must compute the new answers.
questions['scene']['instructions']='Choose the setting visible in the FIRST photograph.'
saved_engine=engine
engine=InferenceEngine(0,feature_cache_mib=0);engine.initialize_memory_budget()
question_ref=trial('changed-question-uncached',case,False)
engine=saved_engine
changed=trial('changed-question',case,True,question_ref)
assert changed['usage']['prefix_cache']=='hit'
questions['scene']['instructions']='Which setting best describes the FIRST photograph?'
# A different image ordering invalidates the language prefix; independent image
# features can still be reused. First image stays the same for ground truth.
reorder={**case,'images':[case['images'][0],*reversed(case['images'][1:])]}
changed_images=trial('reordered-images',reorder,True)
assert changed_images['usage']['prefix_cache']!='hit' and changed_images['usage']['image_feature_cache_hits']>0
assert changed_images['answers']['scene']['choice']=='beach'
pooled=trial('pooled-images',{**case,'pooling':True},True)
assert pooled['usage']['prefix_cache']!='hit' and pooled['usage']['image_feature_cache_hits']>0
# The maximum-length request must evict caches by memory, not fail or shrink.
maxcase={'name':'sixteen-max','images':list(photos)*8,'fidelity':1024,'tokens':45056}
maximum=trial('maximum-16',maxcase,True)
assert maximum['answers']['scene']['choice']=='beach'
assert maximum['cache']['mib']==0
post=trial('post-max-cold',case,True,ref)
recovery=trial('post-max-warm',case,True,post)
assert recovery['usage']['prefix_cache']=='hit'
out['completed']=True;save();print('COMPLETE',flush=True)
