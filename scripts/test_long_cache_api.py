"""Validate Full long-request cache reuse, admission and recovery with private photos."""
import argparse
import base64
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

import copy,statistics
photos={};urls={}
for name in ('IMG_3350.jpeg','IMG_1220.jpeg'):
    with Image.open(a.images_dir/name) as img:photos[name]=ImageOps.exif_transpose(img).convert('RGB')
    urls[name]='data:image/jpeg;base64,'+base64.b64encode((a.images_dir/name).read_bytes()).decode()
model='clef'
def request(path,body=None):
    req=urllib.request.Request(a.url+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as response:return json.load(response)
end=time.monotonic()+240
while True:
    try:
        metadata=request('/v1/models')['data'][0]
        if metadata['max_input_tokens_with_images']==45056:break
    except OSError:pass
    if time.monotonic()>end:raise RuntimeError('Full service did not become ready')
    time.sleep(3)
result={'completed':False,'initial_models':metadata,'trials':[]}
def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
def check(name,body,baseline=None):
    started=time.perf_counter();response=request('/v1/systemone',body);wall=time.perf_counter()-started
    answer=response['answers']
    assert answer['scene']['choice']=='beach' and answer['clothing']['choice']=='red'
    assert all(answer[k]['noul']>.5 for k in ('sunglasses','drink','ship'))
    assert max(answer['urgency']['probabilities'],key=answer['urgency']['probabilities'].get)=='2'
    row={'name':name,'wall_seconds':wall,**response}
    if baseline:
        delta=[]
        for k,v in answer.items():
            ref=baseline['answers'][k]
            if v['type']=='noul':delta.append(abs(v['noul']-ref['noul'])*100)
            else:delta.extend(abs(x-ref['probabilities'][key])*100 for key,x in v['probabilities'].items())
        row['max_probability_delta_pp']=max(delta)
        assert max(delta)<=.1,(name,max(delta))
    result['trials'].append(row);save()
    print('PASS',name,round(wall,3),response['usage'],flush=True)
    return response
case={'images':(list(photos)*6)[:11],'fidelity':1024,'tokens':12288}
body,encoded=make(case)
uncached=check('uncached-11',body)
body.update(prefix_cache=True,input_cache=True)
cold=check('cold-11',body,uncached)
warms=[check('warm-11-'+str(i),body,cold) for i in range(3)]
assert all(r['usage']['prefix_cache']=='hit' and r['usage']['reused_prefix_tokens']>8192 and r['usage']['vision_prefix_reused'] for r in warms)
assert statistics.median(r['usage']['latency_ms'] for r in warms)<cold['usage']['latency_ms']/2
changed=copy.deepcopy(body)
changed['questions']['scene']['instructions']='Choose the setting visible in the FIRST photograph.'
changed['prefix_cache']=False
changed_ref=check('changed-question-uncached',changed)
changed['prefix_cache']=True
changed_response=check('changed-question',changed,changed_ref)
assert changed_response['usage']['prefix_cache']=='hit'
reordered=copy.deepcopy(body);reordered['images']=[body['images'][0],*reversed(body['images'][1:])]
r=check('reordered-images',reordered)
assert r['usage']['prefix_cache']=='hit' and r['usage']['vision_images_reused']>=1
assert r['usage']['image_feature_cache_hits']+r['usage']['vision_images_reused']==11
pooled,_=make({**case,'pooling':True});pooled.update(prefix_cache=True,input_cache=True)
r=check('pooled-images',pooled)
assert r['usage']['prefix_cache']!='hit' and r['usage']['image_feature_cache_hits']==11
cap=metadata['max_input_tokens_with_images']
maximum,_=make({'images':list(photos)*8,'fidelity':1024,'tokens':cap})
maximum.update(prefix_cache=True,input_cache=True)
r=check('maximum-16',maximum)
assert r['usage']['input_tokens']==cap and r['usage']['cache_limit_reason']=='request_memory_budget'
assert request('/health')['optimizations']['prefix_cache']['entries']==0
assert request('/v1/models')['data'][0]['max_input_tokens_with_images']==cap
rejected,_=make({'images':list(photos)*8,'fidelity':1024,'tokens':cap+1})
try:request('/v1/systemone',rejected)
except urllib.error.HTTPError as exc:
    error=json.load(exc);assert exc.code==413 and error['input_tokens']==cap+1 and error['max_input_tokens']==cap
    result['over_limit']=error;save()
else:raise AssertionError('Over-limit request accepted')
post=check('post-max-cold',body,uncached)
recovery=check('post-max-warm',body,post)
assert recovery['usage']['prefix_cache']=='hit'
short={'model':'clef','state':'Checkout is down; act immediately.','questions':{'urgency':body['questions']['urgency']}}
request('/v1/systemone',short);short_reply=request('/v1/systemone',short)
assert short_reply['usage']['prefix_cache']=='hit'
result['short_recovery']=short_reply
result['final_health']=request('/health');result['final_models']=request('/v1/models')
result['completed']=True;save();print('COMPLETE',flush=True)
