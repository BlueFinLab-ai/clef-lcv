"""Validate small-request hits and proactive large-prefill eviction on one GPU.

Requires the two private photo fixtures; no image bytes are saved to results.
Run with --help for isolated Full/Flash checkpoint and GPU options.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile', choices=['full', 'flash'], required=True)
p.add_argument('--data-dir', type=Path, required=True)
p.add_argument('--model-dir', required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--images-dir', type=Path, required=True)
p.add_argument('--max-tokens', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, CLEF_DATA_DIR=str(a.data_dir),
                  CLEF_MODEL_DIR=a.model_dir, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  CLEF_REQUIRE_FAST_LINEAR_ATTENTION='1', PYTORCH_ALLOC_CONF='expandable_segments:True',
                  OMP_NUM_THREADS='4')
sys.path.insert(0, str(project/'builds'/a.profile))
import clef_app as app
import torch
from PIL import Image, ImageOps
from torch.nn.attention import sdpa_kernel, SDPBackend
from optimized_inference import InferenceEngine, encode_compact, answers
from context_budget import ContextBudget
from reusable_inputs import InputCache

model, processor, _ = app.load_model()
budget = ContextBudget(model.language_model.config.text_config, dtype_bytes=2, calibrated=True,
                       configured_limit=a.max_tokens, image_limit=a.max_tokens, image_chunked=True)
engine = InferenceEngine('auto', reserve_mib=256, prefill_reserve_mib=512, utilization=1., elastic=True)
engine.initialize_memory_budget()
inputs = InputCache()
photos = []
for name in ('IMG_3350.jpeg', 'IMG_2666.jpeg'):
    with Image.open(a.images_dir/name) as im:
        photos.append(ImageOps.exif_transpose(im).convert('RGB'))
scaled = []
for im in photos:
    im = im.copy(); im.thumbnail((1024,1024)); scaled.append(im)
questions = {'action': {'type':'choice', 'instructions':'What action is appropriate?',
                        'criteria': {'act':'Act immediately', 'wait':'Wait until later'}}}
def sized(record, tokens):
    record = dict(record)
    for _ in range(4):
        n = len(encode_compact(processor,record,input_cache=inputs)[0].input_ids)
        if n == tokens:return record
        assert n < tokens
        record['state'] += 'word '*(tokens-n)
    raise AssertionError('Cannot construct token fixture')
def text(tokens, variant=0):
    return sized({'state':f'Case {variant}: Checkout is down. Act immediately. ', 'questions':questions},tokens)
image_record = {'state':'Inspect these photographs.', 'images':scaled,
                'media_kwargs':{'images_kwargs':{'do_resize':True,'min_pixels':1024,'max_pixels':1048576}},
                'questions':{'count':{'type':'choice','instructions':'How many images are supplied?',
                                     'criteria':{'one':'One image','two':'Two images','three':'Three images'}}}}
result = {'completed':False, 'profile':a.profile, 'gpu':torch.cuda.get_device_name(),
          'gpu_uuid':a.gpu, 'max_tokens':a.max_tokens, 'trials':[]}
def save():
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
@torch.inference_mode()
def trial(name, record):
    started = time.perf_counter()
    encoded,boundary,points = encode_compact(processor,record,with_checkpoints=True,input_cache=inputs)
    prepared = engine.prepare_request(budget.workspace(len(encoded.input_ids)) if len(encoded.input_ids)>8192 else None)
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            logits,usage = engine.forward(model,processor,encoded,boundary,
                'photos-'+str(len(record.get('images',[]))) if record.get('images') else 'text',
                checkpoint_boundaries=points,prefill_chunk_tokens=4096,prefill_initial_chunk_tokens=8192,prefill_images=True)
        answer = answers(record,encoded,logits)
        torch.cuda.synchronize()
        row = {'name':name,'input_tokens':len(encoded.input_ids),'seconds':time.perf_counter()-started,
               'answers':answer,'usage':{**prepared,**usage},'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
               'active_stats':engine.stats()}
    finally:
        engine.finish_request()
    row['idle_stats'] = engine.stats()
    result['trials'].append(row);save()
    print(json.dumps({'name':name,'seconds':round(row['seconds'],3),'usage':row['usage']}),flush=True)
    return row
def equal(cold, warm):
    for key, value in cold['answers'].items():
        other = warm['answers'][key]
        assert value.get('choice') == other.get('choice')
        if 'probabilities' in value:
            assert max(abs(p-other['probabilities'][k]) for k,p in value['probabilities'].items()) <= .002

cold = trial('8K-cold',text(8192));warm = trial('8K-warm',text(8192));equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit' and warm['usage']['cache_headroom_mib']==256
for i in range(8):trial('fill-'+str(i),text(4096,variant=i+1))
pressure = trial('maximum-text-pressure',text(a.max_tokens,variant=99))
assert pressure['usage']['cache_headroom_mib']==512
assert pressure['usage']['cache_evicted_mib']>0
cold = trial('8K-recovery-build',text(8192));warm = trial('8K-recovery-hit',text(8192));equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit'
cold = trial('two-images-cold',image_record);warm = trial('two-images-warm',image_record);equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit' and warm['answers']['count']['choice']=='two'
# Populate independent features, then combine sixteen image slots and text up
# to the accepted token ceiling; image positions and cache identity stay exact.
mixed = {**image_record, 'images':scaled*8, 'state':'Inspect the supplied photographs. ',
         'questions':{'scene':{'type':'choice','instructions':'Which setting describes the FIRST photograph?',
                             'criteria':{'beach':'Beach','indoor':'Indoors','street':'City street'}}}}
trial('maximum-mixed-pressure',sized(mixed,a.max_tokens))
cold = trial('two-images-recovery-build',image_record);warm = trial('two-images-recovery-hit',image_record);equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit'
if a.profile == 'full':
    native = {**image_record,'images':photos,
              'media_kwargs':{'images_kwargs':{'do_resize':True,'min_pixels':1024,'max_pixels':20000000}}}
    cold = trial('two-originals-cold',native);warm = trial('two-originals-warm',native);equal(cold,warm)
    assert warm['usage']['prefix_cache']=='hit' and warm['usage']['vision_prefix_reused']
result.update(completed=True, final_stats=engine.stats())
save();engine.clear();gc.collect();torch.cuda.empty_cache();print('COMPLETE',flush=True)
