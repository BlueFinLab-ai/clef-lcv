"""Exact parent-only sharing, unique storage totals, independent GDN and eviction."""
from pathlib import Path
from types import SimpleNamespace as NS
import gc,sys,weakref
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'vendor/cloudflare')]
import torch
from host_prefix_cache import HostPrefixCache,copy_to_device
from shared_host_blocks import FrozenTensor,storages,SnapshotPlan
from optimized_inference import tensor_bytes

def entry(length,parent=None,color=0):
 keys=torch.arange(length*16,dtype=torch.float32).view(1,2,length,8)
 # Prefix values do not depend on length; each head uses the same token sequence.
 keys=torch.arange(length*8,dtype=torch.float32).view(1,1,length,8).expand(1,2,length,8).clone()
 hidden=torch.arange(length*12,dtype=torch.float32).view(1,length,12)
 return {'cache':NS(layers=[NS(keys=keys,values=keys+5),NS(conv_states=torch.ones(1,4,4)*color,recurrent_states=torch.ones(1,2,4,4)*color)]),
  'chunks':(hidden,), 'ids':tuple(range(length)), 'media_key':('text',False),'parent_key':parent,'media_start':None}
host=HostPrefixCache(2,None,shared_blocks=True)
a=(1,'text',False,'a');b=(1,'text',False,'b');c=(1,'text',False,'c')
root=entry(900);assert host.put(a,root,tensor_bytes)
parent=host.entries[a];assert isinstance(parent['cache'].layers[0].keys,FrozenTensor)
child=entry(1100,a,2);assert host.put(b,child,tensor_bytes)
assert host.shared_saves==1
saved=copy_to_device((host.entries[b]['cache'],host.entries[b]['chunks']),'cpu')
assert torch.equal(saved[0].layers[0].keys,child['cache'].layers[0].keys)
assert torch.equal(saved[1][0],child['chunks'][0])
assert torch.equal(saved[0].layers[1].recurrent_states,child['cache'].layers[1].recurrent_states)
saved[0].layers[1].conv_states.zero_();assert host.entries[b]['cache'].layers[1].conv_states.all()
assert parent['cache'].layers[1].conv_states.sum()==0
assert host.bytes==sum(storages([(v['cache'],v['chunks']) for v in host.entries.values()]).values())
assert host.bytes<sum(v['bytes'] for v in host.entries.values())
# Removing an ancestor keeps its immutable storage alive for descendants.
host.touch(b);host.evict_one();assert a not in host.entries
assert host.bytes==sum(storages((host.entries[b]['cache'],host.entries[b]['chunks'])).values())
restored=copy_to_device(host.entries[b]['cache'],'cpu');assert torch.equal(restored.layers[0].values,child['cache'].layers[0].values)
# Model/media/ancestry mismatches must not share even if tensor shapes agree.
foreign=(2,'text',False,'foreign');assert host.put(foreign,entry(1200,b,3),tensor_bytes)
assert host.shared_saves==1
changed=entry(1300,b,4);changed['ids']=(-1,)+changed['ids'][1:]
assert host.put(c,changed,tensor_bytes) and host.shared_saves==1
# Entry-count eviction during admission must recharge an unindexed ancestor.
single=HostPrefixCache(2,1,shared_blocks=True);assert single.put(a,root,tensor_bytes);assert single.put(b,child,tensor_bytes)
assert len(single.entries)==1 and single.bytes==sum(storages((single.entries[b]['cache'],single.entries[b]['chunks'])).values())
plan=SnapshotPlan(child,parent);ref=weakref.ref(plan);gc.disable();del plan
assert ref() is None,'Snapshot plans retained GPU source arrays until cyclic GC';gc.enable()
# A changed media namespace cannot share a post-image checkpoint.
media=HostPrefixCache(2,None,shared_blocks=True)
ma=(1,'image-a',False,'a');mb=(1,'image-b',False,'b')
mroot={**entry(900),'media_key':('image-a',False),'media_start':8}
assert media.put(ma,mroot,tensor_bytes)
mchild={**entry(1100,ma,2),'media_key':('image-b',False),'media_start':8}
assert media.put(mb,mchild,tensor_bytes) and media.shared_saves==0
host.clear();assert host.bytes==0 and not host.storage_refs
print('PASS: exact KV/hidden restore, parent storage sharing, independent recurrent states, model/media/ancestry guards, unique accounting, ancestor eviction and prompt release')
