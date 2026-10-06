"""Adopt append-only CPU prefix storage without copying or sharing recurrent state."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'runtime'),str(Path(__file__).resolve().parents[1]/'vendor/cloudflare')]
import torch
from host_prefix_cache import HostPrefixCache
from optimized_inference import tensor_bytes
keys=torch.arange(48).reshape(12,1,1,4).float();values=keys+10
hidden=torch.arange(96).reshape(1,12,8).float()
layer=NS(keys=keys[:8].permute(1,2,0,3),values=values[:8].permute(1,2,0,3),
         conv_states=torch.ones(1,4,2),recurrent_states=torch.ones(1,1,4,4))
entry={'cache':NS(layers=[layer]),'chunks':(hidden[:,:8],),'ids':tuple(range(8)),'media_key':('text',False)}
cache=HostPrefixCache(1,10,shared_blocks=True)
assert cache.put('adopt',entry,tensor_bytes,adopt_cpu=True)
saved=cache.entries['adopt'];k=saved['cache'].layers[0].keys
assert k.blocks[0].untyped_storage()._cdata==keys.untyped_storage()._cdata
assert saved['chunks'][0].blocks[0].untyped_storage()._cdata==hidden.untyped_storage()._cdata
keys[8:].fill_(999);values[8:].fill_(999);hidden[:,8:].fill_(999)
layer.conv_states.zero_();layer.recurrent_states.zero_()
assert torch.equal(k.materialize('cpu'),torch.arange(32).reshape(1,1,8,4).float())
assert saved['cache'].layers[0].conv_states.all() and saved['cache'].layers[0].recurrent_states.all()
assert saved['chunks'][0].materialize('cpu').max()==63
print('PASS: zero-copy prefix adoption; suffix-only appends; recurrent-state isolation; physical storage accounting')
