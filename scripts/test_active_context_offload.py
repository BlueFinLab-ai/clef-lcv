"""CPU lifecycle and incremental KV preservation with mocked CUDA timing."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import torch
from active_context_offload import ActiveKVOffload
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/cloudflare'))
from optimized_inference import InferenceEngine
for mode in ['none','auto','hidden','kv_hidden','kv_stream','kv_gpu']:
 assert InferenceEngine(0,active_context_offload=mode).active_context_offload==mode
class Config:
 layer_types=['full_attention','full_attention']
 num_hidden_layers=2
 def get_text_config(self,decoder=True):return self
class Layer(torch.nn.Module):
 def __init__(self,index):super().__init__();self.index=index
 def forward(self,values,past_key_values):
  past_key_values.update(values,values+10,self.index)
  return values
language=NS(config=Config(),layers=torch.nn.ModuleList([Layer(0),Layer(1)]))
old=torch.cuda.current_stream
try:
 torch.cuda.current_stream=lambda *a:NS(synchronize=lambda:None)
 controller=ActiveKVOffload(language,'cpu')
 for values in [torch.arange(12).view(1,1,3,4).float(),torch.arange(8).view(1,1,2,4).float()+20]:
  for layer in language.layers:layer(values,past_key_values=controller.cache)
 expected=torch.cat([torch.arange(12).view(1,1,3,4).float(),torch.arange(8).view(1,1,2,4).float()+20],dim=2)
 for state in controller.cpu.values():
  assert torch.equal(state['keys'],expected)
  assert torch.equal(state['values'],expected+10)
 assert controller.offload_bytes==expected.numel()*expected.element_size()*4
 controller.close();controller.close()
 assert not controller.hooks and controller.cache is None
 assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in language.layers)
finally:torch.cuda.current_stream=old
print('PASS: exact incremental KV append, all prior tokens retained, new-tail transfers, and idempotent hook cleanup')
