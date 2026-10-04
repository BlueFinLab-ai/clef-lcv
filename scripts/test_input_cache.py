"""CPU validation of retention bounds, namespaces and immutable caller metadata."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/cloudflare'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import torch
import joint_schema_model as joint
from reusable_inputs import InputCache

class Tokenizer:
    def __call__(self,text,**kwargs): return SimpleNamespace(input_ids=list(text.encode()))
proc=SimpleNamespace(tokenizer=Tokenizer())
old=joint._encode_media
calls=[]
def fake(processor,record):
    calls.append(record)
    count=len(record['images'])
    return [99]*count+[10],{'pixel_values':torch.ones(count*4,2),
        'image_grid_thw':torch.tensor([[1,2,2]]*count),'mm_token_type_ids':[1]*count+[0]}
def decode(value):
    if value.startswith('bad'):raise ValueError('invalid image')
    return value
joint._encode_media=fake
try:
    c=InputCache(max_entries=2,token_entries=2,decode=decode)
    rec={'images':['a','b'],'media_kwargs':{'images_kwargs':{'max_pixels':65536}}}
    ids,media=c.media(proc,rec)
    assert ids==[99,99,10] and len(calls)==2
    media['mm_token_type_ids'][0]=55
    media['image_grid_thw'][0,0]=55
    ids,media=c.media(proc,{**rec,'images':['b','a']})
    assert len(calls)==2 and media['mm_token_type_ids']==[1,1,0] and media['image_grid_thw'][0,0]==1
    c.media(proc,{**rec,'images':['c']})
    assert len(c.images)==2 and c.image_evictions==1
    c.media(proc,{**rec,'images':['a'],'media_kwargs':{'images_kwargs':{'max_pixels':262144}}})
    assert len(calls)==4, 'Fidelity must not reuse old patches'
    other=SimpleNamespace(tokenizer=Tokenizer())
    c.media(other,{**rec,'images':['a']})
    assert len(calls)==5,'Processor identity must be isolated'
    try:c.media(proc,{'images':['bad-data']})
    except ValueError:pass
    else:raise AssertionError('Validation skipped')
    tokens=c.tokens(proc.tokenizer,'hello');tokens[0]=0
    assert c.tokens(proc.tokenizer,'hello')[0]==ord('h')
    c.tokens(proc.tokenizer,'two');c.tokens(proc.tokenizer,'three')
    assert len(c.text)==2 and c.token_evictions==1
    c.clear();assert not c.images and not c.text
    zero=InputCache(max_mib=0,token_mib=0,decode=decode)
    zero.media(proc,rec);zero.tokens(proc.tokenizer,'hello')
    assert not zero.images and not zero.text
    zero.media(proc,rec,False)
    assert zero.image_hits==0
    c.media(proc,{'images':['x'*1000]})
    assert all('x'*1000 not in str(key) for key in c.images),'Raw upload leaked into key'
    assert c.stats()['image_mib']<=c.stats()['image_limit_mib']
finally:joint._encode_media=old
print('PASS: CPU LRU/byte caps, exact identities, changed fidelity, independent metadata, invalid input, token copies, bypass and no raw keys')
