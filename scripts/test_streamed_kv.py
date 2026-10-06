"""CUDA numerical test: GQA, causal new chunk, CPU history, odd block tails."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runtime'))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from streamed_kv import StreamedKVOffload

if not torch.cuda.is_available() or torch.version.hip:
    raise SystemExit('This numerical test requires an NVIDIA CUDA GPU')
torch.manual_seed(7319)
rows=[]
for storage in ('cpu','cuda'):
 for dtype in (torch.float16,torch.bfloat16):
     cases=[(0,17,5,4,2,64),(13,7,5,4,2,64),(61,17,16,4,2,64),
            (257,63,64,4,2,64),(33,1,16,4,2,64),
            (257,63,64,16,4,256),(513,128,128,16,4,256)]
     for old_length,chunk,block,heads,kv_heads,dim in cases:
         controller=StreamedKVOffload.__new__(StreamedKVOffload)
         controller.device='cuda';controller.capacity=old_length+chunk
         controller.cpu={};controller.buffers=None;controller.block_tokens=block;controller.storage=storage;controller.prefix=None;controller.host_buffers=None;controller.tail_buffers=None
         controller.copy_stream=torch.cuda.Stream();controller.upload_bytes=controller.blocks=0
         q=torch.randn(1,heads,chunk,dim,device='cuda',dtype=dtype)
         k=torch.randn(1,kv_heads,old_length+chunk,dim,device='cuda',dtype=dtype)
         v=torch.randn_like(k)
         controller._allocate(0,k[:,:,-chunk:],v[:,:,-chunk:])
         state=controller.cpu[0]
         state['keys'][:old_length].copy_(k[:,:,:old_length].permute(2,0,1,3))
         state['values'][:old_length].copy_(v[:,:,:old_length].permute(2,0,1,3))
         state['length']=old_length
         assert state['keys'][:min(block,old_length)].is_contiguous()
         groups=heads//kv_heads
         module=SimpleNamespace(num_key_value_groups=groups,scaling=dim**-0.5)
         actual=controller.attend(module,q,k[:,:,-chunk:],v[:,:,-chunk:],state)
         causal=(torch.arange(old_length+chunk,device='cuda')[None,:]
                 <= old_length+torch.arange(chunk,device='cuda')[:,None])
         with sdpa_kernel(SDPBackend.MATH):
             expected=F.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(groups,dim=1),
                 v.float().repeat_interleave(groups,dim=1),attn_mask=causal).to(dtype)
         error=(actual-expected).abs().max().item()
         torch.testing.assert_close(actual,expected,atol=0.01,rtol=0.04)
         assert controller.upload_bytes==(old_length*2*kv_heads*dim*k.element_size() if storage=='cpu' else 0)
         rows.append((str(dtype),old_length,chunk,block,heads,kv_heads,dim,error))
         controller.copy_stream.synchronize();torch.cuda.synchronize()
print('PASS: blockwise CPU-history attention agrees with FP32 dense causal attention')
for row in rows:print(row)

# A failed request must not leave the model's attention methods or mask policy
# patched. Test this without allocating/loading another language model.
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
config=Qwen3_5TextConfig(hidden_size=128,num_attention_heads=2,
    num_key_value_heads=1,head_dim=64,num_hidden_layers=2,
    layer_types=['full_attention','full_attention'])
layers=torch.nn.ModuleList([torch.nn.Module(),torch.nn.Module()])
for index,layer in enumerate(layers):layer.self_attn=Qwen3_5Attention(config,index).eval()
language=SimpleNamespace(config=config,layers=layers)
originals=[layer.self_attn.forward for layer in layers]
controller=StreamedKVOffload(language,'cuda',128)
try:
    try:
        layers[0].self_attn(hidden_states=torch.zeros(2,1,128),
            position_embeddings=(None,None),past_key_values=controller.cache)
    except ValueError:pass
    else:raise AssertionError('Batched streamed request was not rejected')
finally:controller.close()
controller.close()
assert language.config is config and controller.cache is None
assert all(layer.self_attn.forward==old for layer,old in zip(layers,originals))
print('PASS: rejected batched request, original attention/config restored, idempotent cleanup')

from transformers.cache_utils import DynamicCache
from shared_host_blocks import FrozenTensor
for layer in layers:layer.self_attn.to(device='cuda',dtype=torch.float16)
source=DynamicCache(config=config)
keys=torch.randn(1,1,8,64,dtype=torch.float16)
values=torch.randn_like(keys)
for index in range(2):
    source.update(keys.clone(),values.clone(),index)
    for name in ('keys','values'):
        tensor=getattr(source.layers[index],name)
        setattr(source.layers[index],name,FrozenTensor(tensor.shape,tensor.dtype,2,tensor.chunk(2,dim=2),tensor.element_size()))
prefix={'cache':source,'ids':tuple(range(8))}
for storage in ('cpu','cuda'):
    controller=StreamedKVOffload(language,'cuda',12,storage=storage,prefix=prefix)
    try:
        assert controller.cache.get_seq_length()==8
        for state in controller.cpu.values():
            if storage=='cpu':
                restored=torch.empty(8,1,1,64,dtype=keys.dtype)
                controller._read_history(state,'keys',0,8,restored)
                assert state['keys'].shape[0]==4
            else:restored=state['keys'][:8].cpu()
            torch.testing.assert_close(restored,keys.permute(2,0,1,3),atol=0,rtol=0)
        with torch.inference_mode():
            hidden=torch.randn(1,2,128,device='cuda',dtype=torch.float16)
            positions=(torch.ones(1,2,64,device='cuda',dtype=torch.float16),torch.zeros(1,2,64,device='cuda',dtype=torch.float16))
            for layer in layers:layer.self_attn(hidden,positions,past_key_values=controller.cache)
        assert controller.cache.get_seq_length()==10
        for layer in source.layers:
            torch.testing.assert_close(layer.keys.materialize('cpu'),keys,atol=0,rtol=0)
            torch.testing.assert_close(layer.values.materialize('cpu'),values,atol=0,rtol=0)
    finally:controller.close()
print('PASS: immutable segmented CPU prefix restores into both tiers; new suffix extends owned state only')
