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

from concurrent.futures import ThreadPoolExecutor
photos={};urls={}
for name in ('IMG_3350.jpeg','IMG_1220.jpeg'):
    with Image.open(a.images_dir/name) as img: photos[name]=ImageOps.exif_transpose(img).convert('RGB')
    urls[name]='data:image/jpeg;base64,'+base64.b64encode((a.images_dir/name).read_bytes()).decode()
def request(path,body=None):
    req=urllib.request.Request(a.url+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as response:return json.load(response)
end=time.monotonic()+240
while True:
    try:
        metadata=request('/v1/models')['data'][0]
        if metadata['context_limit']['image_limit_basis']=='computed_memory_estimate':break
    except OSError:pass
    if time.monotonic()>end:raise RuntimeError('Automatic image budget not ready')
    time.sleep(3)
cap=metadata['max_input_tokens_with_images']
assert cap==metadata['max_input_tokens']==45056
refs=json.loads(a.reference.read_text())['cases']
result={'initial_models':metadata,'cases':[],'completed':False}
def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
def body_for(record,variant=0,image_count=2):
    body={k:v for k,v in record.items() if k not in ('images','media_kwargs')}
    if 'images' in record:
        order=list(photos) if variant==0 else list(reversed(photos))
        body.update(images=[urls[n] for n in order*(image_count//2)],image_fidelity='high')
    body.update(model='clef',prefix_cache=False,input_cache=False)
    return body
for variant,count in [(1,2),(0,16)]:
    record,encoded,boundary,truth=fixture(cap,True,variant,count)
    ref=next(c for c in refs if c['tokens']==cap and c['images']==count and c['variant']==variant)
    assert hashlib.sha256(json.dumps(encoded.input_ids).encode()).hexdigest()==ref['input_ids_sha256']
    body=body_for(record,variant,count)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(request,'/v1/systemone',body)
        # Allow CPU preprocessing and inference to acquire the request lock.
        deadline=time.monotonic()+20
        while True:
            busy=request('/v1/models')
            if busy['data'][0].get('context_snapshot_status')=='last_idle_busy' or time.monotonic()>deadline or future.done():break
            time.sleep(.5)
        response=future.result()
    assert all(response['answers'][k]['choice']==v for k,v in truth.items())
    usage=response['usage'];assert usage['input_tokens']==cap and usage['accepted_max_input_tokens']==cap and usage['multimodal_prefill']
    baseline=next(t for t in ref['trials'] if t['policy']==[8192,4096])
    drift=max(abs(response['answers'][k]['probabilities'][opt]-v)*100 for k,ans in baseline['answers'].items() for opt,v in ans['probabilities'].items())
    assert drift<.1,(count,drift)
    assert busy['data'][0]['max_input_tokens_with_images']==cap
    assert busy['data'][0].get('context_snapshot_status')=='last_idle_busy',busy
    after=request('/v1/models')['data'][0]
    assert after['max_input_tokens_with_images']==cap
    result['cases'].append({'images':count,'variant':variant,'correct':True,'max_probability_delta_pp':drift,'usage':usage,'busy_discovery':busy,'after_models':after})
    save();print('PASS',count,'images',usage,flush=True)
for with_images in (True,False):
    record,encoded,boundary,truth=fixture(cap+1,with_images)
    try:request('/v1/systemone',body_for(record))
    except urllib.error.HTTPError as exc:
        rejection=json.load(exc)
        assert exc.code==413 and rejection['input_tokens']==cap+1 and rejection['max_input_tokens']==cap
        result['image_over_limit' if with_images else 'text_over_limit']=rejection
    else:raise AssertionError('Over-limit input accepted')
    save();print('PASS 413 images=',with_images,flush=True)
record,encoded,boundary,truth=fixture(8192,False)
response=request('/v1/systemone',body_for(record))
assert all(response['answers'][k]['choice']==v for k,v in truth.items())
result['text_regression']=response
short={'model':'clef','state':'START ALERT COLOR: RED. MIDDLE ALERT COLOR: GREEN. END ALERT COLOR: BLUE.','questions':record['questions']}
request('/v1/systemone',short);response=request('/v1/systemone',short)
assert response['usage']['prefix_cache']=='hit'
assert all(response['answers'][k]['choice']==v for k,v in truth.items())
result['cache_recovery']=response
result['final_models']=request('/v1/models')
result['completed']=True;save();print('COMPLETE',flush=True)
