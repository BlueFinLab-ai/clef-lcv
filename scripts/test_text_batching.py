"""CPU guards for independent hybrid branches and memory/shape admission."""
import sys,copy
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from text_batching import clone_hybrid_cache,TextBatchAdmission,BatchUnavailable
layer=SimpleNamespace(keys=torch.randn(1,4,17,8),values=torch.randn(1,4,17,8),
                      conv_states=torch.randn(1,32,4),recurrent_states=torch.randn(1,8,16,16),max_batch_size=1)
cache=SimpleNamespace(layers=[layer]);before=copy.deepcopy(cache)
branch=clone_hybrid_cache(cache,3)
for name in ['keys','values','conv_states','recurrent_states']:
    tensor=getattr(branch.layers[0],name)
    assert tensor.shape[0]==3
    tensor[0].zero_()
    torch.testing.assert_close(tensor[1],getattr(before.layers[0],name)[0])
    torch.testing.assert_close(getattr(cache.layers[0],name),getattr(before.layers[0],name))
assert branch.layers[0].max_batch_size==3 and layer.max_batch_size==1
cfg=SimpleNamespace(layer_types=['full_attention']*8+['linear_attention']*24,
 num_key_value_heads=4,head_dim=256,linear_num_value_heads=32,
 linear_key_head_dim=128,linear_value_head_dim=128,hidden_size=4096,num_attention_heads=16)
a=TextBatchAdmission(cfg)
assert a.reason([2444,2665]) is None
assert a.reason([2444,2665,2636,2443])=='padded_token_budget'
assert a.reason([1000,2800])=='length_mismatch'
assert a.reason([5000,5000])=='long_request'
assert a.workspace([2665]*4,1544)>a.workspace([2665]*2,1544)
a.oom([2444,2665]);assert a.reason([2444,2665])=='previous_batch_oom'
print('PASS: independent KV/conv/recurrent rows, unchanged source cache, padding/length bounds and learned OOM guard')
