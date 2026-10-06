"""Differential checks against the previous scan, with lifecycle and media changes."""
from pathlib import Path
import hashlib,json,random,sys,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'vendor/cloudflare')]
from optimized_inference import InferenceEngine
rng=random.Random(811)
engine=InferenceEngine(0);models=[object(),object()]
banks=[engine.entries,engine.host.entries,engine.observations]
def reference(model,tokens,boundary,image,pool,start,end):
 best=None;common=0
 for bank in [engine.observations,engine.host.entries,engine.entries]:
  for key,entry in bank.items():
   if key[0]!=id(model):continue
   matched=0
   for left,right in zip(tokens[:boundary],entry['ids']):
    if left!=right:break
    matched+=1
   if start is not None and entry['media_key']!=(image,bool(pool)):matched=min(matched,start)
   if start is not None and start<matched<end:matched=start
   common=max(common,matched);length=len(entry['ids'])
   if 'cache' in entry and length<=matched and key[:3]==engine._namespace(model,image,pool,length,start):
    if best is None or length>len(engine._lookup(best)['ids']):best=key
 return best,common
for step in range(1600):
 model=rng.choice(models);image=rng.choice(['a','b','c']);pool=bool(rng.randrange(2))
 start=rng.choice([None,8,16]);end=start+8 if start is not None else 0
 tokens=tuple([1]*8+[rng.randrange(4) for _ in range(56)])
 if rng.random()<.75 and engine.prefix_index.records:
  old=rng.choice(list(engine.prefix_index.records.values()))['ids']
  n=rng.randrange(len(old)+1);tokens=old[:n]+tokens[n:]
 boundary=rng.randrange(1,65)
 if rng.random()<.65:
  cut=rng.randrange(1,65)
  if start is not None and start<cut<end:cut=start
  ids=tokens[:cut];key=engine._input_key(model,ids,len(ids),image,pool,start)
  bank=rng.choice(banks)
  bank[key]={'ids':ids,'media_key':(image,pool),**({'cache':object()} if bank is not engine.observations else {})}
 elif rng.random()<.85:
  bank=rng.choice(banks)
  if bank:bank.pop(rng.choice(list(bank)))
 else:rng.choice(banks).clear()
 expected=reference(model,tokens,boundary,image,pool,start,end)
 actual=engine._match(model,tokens,boundary,image,pool,start,end)
 # Equal-length equivalent terminals may be chosen differently, but their token
 # ancestry, namespace and reuse length must agree exactly.
 assert actual[1]==expected[1],(step,actual,expected)
 assert (actual[0] is None)==(expected[0] is None),(step,actual,expected)
 if actual[0] is not None:
  assert engine._lookup(actual[0])['ids']==engine._lookup(expected[0])['ids']
  assert actual[0][:3]==expected[0][:3]
# The index must retain metadata only, and release all dead namespaces on clear.
for bank in banks:bank.clear()
assert not engine.prefix_index.records and not engine.prefix_index.models and not engine.prefix_index.media and not engine.prefix_index.cached
print('PASS: 1600 randomized scan/radix comparisons, divergent ancestry, model/media/pooling changes, dual-tier sources, observations, deletion and clear')
