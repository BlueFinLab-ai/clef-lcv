"""Check live adaptive cache admission, pressure eviction and recovery.

Uses CPU processor fixtures and two private photo files. Saves no input payloads.
"""
import argparse,base64,io,json,os
from pathlib import Path
import sys,time,urllib.error,urllib.request
os.environ['CUDA_VISIBLE_DEVICES']=''
from PIL import Image,ImageOps
from transformers import AutoProcessor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',required=True)
p.add_argument('--project',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--images-dir',type=Path,required=True)
p.add_argument('--expected-text-limit',type=int,required=True)
p.add_argument('--expected-image-limit',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.project/'runtime'),str(a.project/'vendor/cloudflare')]
from optimized_inference import encode_compact
from reusable_inputs import InputCache
def request(path,body=None):
    req=urllib.request.Request(a.url+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)
deadline=time.monotonic()+240
while True:
    try:
        health=request('/health');c=health['optimizations']['prefix_cache']
        if c.get('elastic') and c['reserve_mib']==256 and c.get('prefill_reserve_mib')==512:break
    except OSError:pass
    if time.monotonic()>deadline:raise RuntimeError('Adaptive cache did not become ready')
    time.sleep(3)
metadata=request('/v1/models')['data'][0];model=metadata['id']
assert metadata['max_input_tokens']==a.expected_text_limit
assert metadata['max_input_tokens_with_images']==a.expected_image_limit
processor=AutoProcessor.from_pretrained(a.checkpoint,local_files_only=True)
def decode(value):
    with Image.open(io.BytesIO(base64.b64decode(value.split(',',1)[1]))) as im:return im.convert('RGB')
fixtures=InputCache(decode=decode)
urls=[]
for name in ('IMG_3350.jpeg','IMG_2666.jpeg'):
    with Image.open(a.images_dir/name) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1024,1024));buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=95)
        urls.append('data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode())
questions={'action':{'type':'choice','instructions':'What action is appropriate?',
                     'criteria':{'act':'Act immediately','wait':'Wait until later'}}}
def sized(record,tokens):
    record=dict(record)
    for _ in range(4):
        n=len(encode_compact(processor,record,input_cache=fixtures)[0].input_ids)
        if n==tokens:return dict(record,model=model)
        assert n<tokens
        record['state']+='word '*(tokens-n)
    raise AssertionError('Cannot construct token fixture')
def text(tokens,variant=0):
    return sized({'state':f'Case {variant}: Checkout is down. Act immediately. ', 'questions':questions},tokens)
result={'completed':False,'url':a.url,'initial_models':metadata,'before':health,'trials':[]}
def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def trial(name,body):
    started=time.perf_counter();response=request('/v1/systemone',body)
    row={'name':name,'wall_seconds':time.perf_counter()-started,**response,'after_health':request('/health')}
    result['trials'].append(row);save()
    print(json.dumps({'name':name,'wall_seconds':round(row['wall_seconds'],3),'usage':row['usage']}),flush=True)
    assert row['usage']['prefix_cache']!='memory_fallback'
    assert row['after_health']['optimizations']['prefix_cache']['phase']=='idle'
    return row
def equal(cold,warm):
    for key,value in cold['answers'].items():
        other=warm['answers'][key];assert value.get('choice')==other.get('choice')
        if 'probabilities' in value:assert max(abs(v-other['probabilities'][k]) for k,v in value['probabilities'].items())<=.002
body=text(8192)
cold=trial('8K-cold',body);warm=trial('8K-warm',body);equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit' and warm['usage']['cache_headroom_mib']==256
for i in range(8):trial('fill-'+str(i),text(4096,variant=i+1))
maximum=trial('maximum-text-pressure',text(a.expected_text_limit,variant=99))
assert maximum['usage']['cache_headroom_mib']==512 and maximum['usage']['cache_evicted_mib']>0
cold=trial('8K-recovery-build',body);warm=trial('8K-recovery-hit',body);equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit'
image={'model':model,'state':'Inspect these photographs.','images':urls,
       'media_kwargs':{'images_kwargs':{'do_resize':True,'min_pixels':1024,'max_pixels':1048576}},
       'questions':{'count':{'type':'choice','instructions':'How many images are supplied?',
                            'criteria':{'one':'One image','two':'Two images','three':'Three images'}}}}
cold=trial('two-images-cold',image);warm=trial('two-images-warm',image);equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit' and warm['answers']['count']['choice']=='two'
mixed={**image,'images':urls*8,'state':'Inspect the supplied photographs. ',
       'questions':{'scene':{'type':'choice','instructions':'Which setting describes the FIRST photograph?',
                            'criteria':{'beach':'Beach','indoor':'Indoors','street':'City street'}}}}
trial('maximum-mixed-pressure',sized(mixed,a.expected_image_limit))
cold=trial('two-images-recovery-build',image);warm=trial('two-images-recovery-hit',image);equal(cold,warm)
assert warm['usage']['prefix_cache']=='hit'
try:request('/v1/systemone',text(a.expected_text_limit+1,variant=99))
except urllib.error.HTTPError as exc:
    assert exc.code==413
    error=json.load(exc);assert error['input_tokens']==a.expected_text_limit+1 and error['max_input_tokens']==a.expected_text_limit
    result['over_limit']=error
else:raise AssertionError('Over-limit request accepted')
result.update(completed=True,final_models=request('/v1/models'),final_health=request('/health'))
assert result['final_models']['data'][0]['max_input_tokens']==a.expected_text_limit
assert result['final_models']['data'][0]['max_input_tokens_with_images']==a.expected_image_limit
save();print('COMPLETE',flush=True)
