"""Check deployed multimodal prefill using private local photos and frozen references."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
os.environ['CUDA_VISIBLE_DEVICES']=''
from PIL import Image, ImageOps
from transformers import AutoProcessor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',required=True)
p.add_argument('--project',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--images-dir',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.project/'runtime'),str(a.project/'vendor/cloudflare')]
from optimized_inference import encode_compact, pool_record
processor=AutoProcessor.from_pretrained(a.checkpoint,local_files_only=True)
image_token_id=json.loads((a.checkpoint/"config.json").read_text())["image_token_id"]
QUESTIONS={'scene': {'type': 'choice',
           'instructions': 'Which setting best describes the FIRST photograph?',
           'criteria': {'beach': 'Outdoor beach with sand and water.',
                        'indoors': 'Indoor portrait with a wall background.',
                        'street': 'Outdoor city street.'}},
 'sunglasses': {'type': 'noul',
                'instructions': 'In the FIRST photograph, is the foreground '
                                'person wearing sunglasses?'},
 'drink': {'type': 'noul',
           'instructions': 'In the FIRST photograph, is the foreground person '
                           'holding a drink?'},
 'clothing': {'type': 'choice',
              'instructions': 'What is the predominant clothing color in the '
                              'FIRST photograph?',
              'criteria': {'red': 'Red',
                           'blue': 'Blue',
                           'other': 'Another color'}},
 'ship': {'type': 'noul',
          'instructions': 'In the FIRST photograph, is a large cruise ship '
                          'visible in the background?'},
 'urgency': {'type': 'score',
             'instructions': 'Rate the urgency of the text incident: checkout '
                             'is down and requires immediate action.',
             'criteria': ['Can wait', 'This week', 'Today']}}
references={c['case']['name']:c for c in json.loads(a.reference.read_text())['cases']}
photos={};urls={}
for name in ('IMG_3350.jpeg','IMG_1220.jpeg'):
    with Image.open(a.images_dir/name) as img:
        photos[name]=ImageOps.exif_transpose(img).convert('RGB')
    urls[name]='data:image/jpeg;base64,'+base64.b64encode((a.images_dir/name).read_bytes()).decode()

def request(path,body=None):
    req=urllib.request.Request(a.url+path,data=json.dumps(body).encode() if body is not None else None,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as r:return json.load(r)

end=time.monotonic()+240
while True:
    try:
        health=request('/health')
        if health['optimizations']['chunked_prefill'].get('images_enabled'):break
    except OSError:pass
    if time.monotonic()>end:raise RuntimeError('Updated service did not become ready')
    time.sleep(3)
model='clef-flash' if 'flash' in health['model'] else 'clef'
metadata=request('/v1/models')['data'][0]
assert metadata['context_limit']['image_limit_basis'] in {'configured_chunked_ceiling','computed_memory_estimate'}

def make(case):
    original_state='Checkout is down; act immediately. Inspect the photographs in their supplied order.'
    record={'state':original_state,'images':[photos[n] for n in case['images']], 'questions':QUESTIONS,
        'media_kwargs':{'images_kwargs':{'min_pixels':min(65536,case['fidelity']**2),'max_pixels':case['fidelity']**2}}}
    def prepare():
        encoded,boundary=encode_compact(processor,record)
        if case.get('pooling'):
            encoded,boundary=pool_record(encoded,boundary,image_token_id)
        return encoded
    if case.get('image_offset'):
        record['context']='word '
        encoded=prepare()
        record['context']='word '*(1+case['image_offset']-encoded.media['token_offset'])
    encoded=prepare()
    if case.get('tokens'):
        words=case['tokens']-len(encoded.input_ids)
        assert words>=0
        for _ in range(4):
            record['state']='word '*words+original_state
            encoded=prepare()
            if len(encoded.input_ids)==case['tokens']:break
            words+=case['tokens']-len(encoded.input_ids)
        assert len(encoded.input_ids)==case['tokens']
    record.pop('media_kwargs')
    record.update(model=model,images=[urls[n] for n in case['images']],prefix_cache=False,input_cache=False,
                  image_fidelity={256:'standard',512:'medium',1024:'high'}[case['fidelity']],
                  image_pooling=case.get('pooling',False))
    return record,encoded

def selected(v):
    return v['choice'] if v['type']=='choice' else v['noul']>=.5 if v['type']=='noul' else max(v['probabilities'],key=v['probabilities'].get)
def delta(v,r):
    if v['type']=='noul':return abs(v['noul']-r['noul'])*100
    return max(abs(v['probabilities'][k]-r['probabilities'][k])*100 for k in v['probabilities'])
result={'models':metadata,'cases':[],'completed':False}
long_name='long-24k-mid-image' if metadata['max_input_tokens_with_images']>=24576 else 'long-16k-mid-image'
images_name='sixteen-images-24k' if metadata['max_input_tokens_with_images']>=24576 else 'sixteen-images-16k'
for name in ['IMG_3350.jpeg-256','IMG_1220.jpeg-256','reverse-images','four-images-pooled',
             'long-10k-mid-image',long_name,images_name]:
    ref=references[name];body,encoded=make(ref['case'])
    assert hashlib.sha256(json.dumps(encoded.input_ids).encode()).hexdigest()==ref['input_ids_sha256']
    response=request('/v1/systemone',body)
    assert response['usage']['input_tokens']==len(encoded.input_ids)
    # Automatic Full admission was promoted after controlled quality checks.
    # Detect serving regressions against the validated chunked implementation;
    # the separate ablation benchmark measures native-vs-chunked numerics.
    baseline=ref['single_pass']
    comparison='native'
    if metadata['context_limit']['image_limit_basis']=='computed_memory_estimate' and len(encoded.input_ids)>8192:
        baseline=next(t for t in ref['chunked'] if t['policy']==[8192,4096])
        comparison='validated_chunked'
    assert all(selected(v)==selected(baseline['answers'][k]) for k,v in response['answers'].items())
    drift=max(delta(v,baseline['answers'][k]) for k,v in response['answers'].items())
    assert drift<=1.,(name,drift)
    if len(encoded.input_ids)>8192:
        assert response['usage'].get('multimodal_prefill')
        assert response['usage']['image_chunk_splits']>0
    entry={'case':name,'comparison':comparison,'selected_match':True,'max_probability_delta_pp':drift,'usage':response['usage']}
    result['cases'].append(entry);print('PASS',name,response['usage']['input_tokens'],drift,flush=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
cap=metadata['max_input_tokens_with_images']
case={**references['long-24k-mid-image']['case'],'tokens':cap+1}
body,encoded=make(case)
try:request('/v1/systemone',body)
except urllib.error.HTTPError as exc:
    rejected=json.load(exc);assert exc.code==413 and rejected['max_input_tokens']==cap
    assert rejected['input_tokens']==cap+1
    result['image_one_token_over']=rejected
else:raise AssertionError('Image overlength input accepted')
body={'model':model,'state':'Checkout is down; act immediately.','questions':QUESTIONS}
request('/v1/systemone',body);recovery=request('/v1/systemone',body)
assert recovery['usage']['prefix_cache']=='hit'
result['short_cache_recovery']=recovery['usage']
result['completed']=True
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('COMPLETE',model,flush=True)
