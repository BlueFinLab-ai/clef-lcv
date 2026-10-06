"""Exact image ancestry, safe cuts, radix branches and tier lifecycle (CPU only)."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from media_prefix import MediaPrefix
from radix_index import PrefixRadixIndex, IndexedBank

class Grids(list):
    def tolist(self): return list(self)
processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda name: {'<|vision_start|>': 8, '<|vision_end|>': 10}[name]))

def fixture(images, count=4, options=None):
    tokens = (1, 2, 3) + (8, *([9] * count), 10) * len(images) + (4, 5, 6)
    encoded = SimpleNamespace(input_ids=tokens, media={'token_offset': 3,
        'mm_token_type_ids': [0] * ((count + 2) * len(images) + 1),
        'image_grid_thw': Grids([[1, 2, count * 2]] * len(images))})
    return encoded, MediaPrefix.build(processor, encoded, images, options)

index=PrefixRadixIndex();gpu=IndexedBank(index,'gpu');cpu=IndexedBank(index,'cpu');obs=IndexedBank(index,'observation')
def add(model, encoded, ancestry, end, bank=gpu, pool=False):
    namespace=ancestry.at(end)
    key=(model,namespace,pool if namespace!='text' else False,hashlib.sha256(str(encoded.input_ids[:end]).encode()).hexdigest())
    bank[key]={'ids': encoded.input_ids[:end], 'media_key': key[1:3], 'cache': object()}
    return key

def match(images, model=11, pool=False, options=None, count=4):
    e,a=fixture(images,count,options)
    result=index.match(model,e.input_ids,len(e.input_ids),a,pool,3,len(e.input_ids)-2)
    return result,e,a

original,a=fixture(['A','B','C','D','E'])
keys=[add(11,original,a,end) for end in a.checkpoints]
result,e,b=match(['A','B','C','D','F'])
assert result[0]==keys[3] and result[1]==a.checkpoints[3]
assert match(['A','B','C','D'])[0][0]==keys[3]
assert match(['A','B','C','D','E','F'])[0][0]==keys[4]
assert match(['A','B','X','D','E'])[0][0]==keys[1]
assert match(['A','C','B','D','E'])[0][0]==keys[0]
assert match(['X','B','C','D','E'])[0][0] is None
assert match(['A','B','C','D','F'],model=12)[0][0] is None
assert match(['A','B','C','D','F'],pool=True)[0][0] is None
assert match(['A','B','C','D','F'],options={'max_pixels': 512**2})[0][0] is None
assert match(['A','B','C','D','F'],count=8)[0][0] is None
for start,end in a.spans:
    assert a.safe(end)==end and a.safe(start+1)==start
# Changing the final grid/size preserves earlier images and their token positions.
varied_tokens = original.input_ids[:a.checkpoints[3]] + (8, *([9] * 8), 10, 4, 5, 6)
varied = SimpleNamespace(input_ids=varied_tokens, media={'token_offset':3,
    'mm_token_type_ids':[0]*(len(varied_tokens)-5),
    'image_grid_thw':Grids([[1,2,8]]*4+[[1,2,16]])})
v=MediaPrefix.build(processor,varied,['A','B','C','D','F'])
assert index.match(11,varied_tokens,len(varied_tokens),v,False,3,len(varied_tokens)-2)[0]==keys[3]
# A different text beginning invalidates every downstream image checkpoint.
different=(7, *original.input_ids[1:])
assert index.match(11,different,len(different),a,False,3,len(different)-2)[0] is None
# Both tiers retain one shared metadata terminal; eviction removes only its tier.
cpu[keys[3]]=gpu[keys[3]];del gpu[keys[3]]
assert match(['A','B','C','D','F'])[0][0]==keys[3]
del cpu[keys[3]]
assert match(['A','B','C','D','F'])[0][0]==keys[2]
# Observed-only state cannot supply a working KV snapshot.
obs[keys[3]]={'ids':original.input_ids[:a.checkpoints[3]],'media_key':keys[3][1:3]}
assert match(['A','B','C','D','F'])[0][0]==keys[2]
# Identical token IDs never authorize different images.
assert e.input_ids==original.input_ids and a.at(a.checkpoints[4])!=b.at(b.checkpoints[4])
# Malformed media fails closed.
try:MediaPrefix.build(processor,original,['A'])
except ValueError:pass
else:raise AssertionError('Malformed image metadata accepted')
gpu.clear();cpu.clear();obs.clear()
assert not index.records and not index.cached
print('PASS: replace/remove/append/reorder, exact bytes/settings/grids/model/pooling, safe image boundaries, GPU/RAM lifecycle and no token-only false hits')
