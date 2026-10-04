import argparse,json,statistics,time,torch
from pathlib import Path
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule
p=argparse.ArgumentParser(description='Compare optimized kernels with native PyTorch math.')
p.add_argument('--dtype',choices=['bfloat16','float16'],default='bfloat16')
p.add_argument('--output',type=Path)
p.add_argument('--heads',type=int,default=4)
p.add_argument('--lengths',default='128,577')
p.add_argument('--benchmark',action='store_true')
a=p.parse_args();dtype=getattr(torch,a.dtype)
lengths=[int(x) for x in a.lengths.split(',')]
if a.heads<=0 or not lengths or any(n<2 for n in lengths):p.error('Heads must be positive and lengths at least two.')
out=[];torch.manual_seed(123)
with torch.inference_mode():
 for length in lengths:
  q,k,v=[torch.randn(1,length,a.heads,128,device='cuda',dtype=dtype) for _ in range(3)]
  g=-torch.rand(1,length,a.heads,device='cuda',dtype=torch.float32);beta=torch.rand_like(g).to(dtype)
  kwargs=dict(output_final_state=True,use_qk_l2norm_in_kernel=True)
  actual,state=chunk_gated_delta_rule(q,k,v,g,beta,**kwargs)
  reference,reference_state=torch_chunk_gated_delta_rule(q,k,v,g,beta,**kwargs)
  torch.cuda.synchronize();diff=(actual.float()-reference.float()).abs()
  assert torch.isfinite(actual).all()
  torch.testing.assert_close(actual.float(),reference.float(),rtol=.08,atol=.015)
  torch.testing.assert_close(state.float(),reference_state.float(),rtol=.08,atol=.015)
  split=length//2
  _,initial=chunk_gated_delta_rule(q[:,:split],k[:,:split],v[:,:split],g[:,:split],beta[:,:split],**kwargs)
  continued,final=chunk_gated_delta_rule(q[:,split:],k[:,split:],v[:,split:],g[:,split:],beta[:,split:],initial_state=initial,**kwargs)
  torch.testing.assert_close(continued.float(),actual[:,split:].float(),rtol=.08,atol=.015)
  torch.testing.assert_close(final.float(),state.float(),rtol=.08,atol=.015)
  step_args=dict(g=g[:,-1:],beta=beta[:,-1:],initial_state=state,
      output_final_state=True,use_qk_l2norm_in_kernel=True)
  step,step_state=fused_recurrent_gated_delta_rule(q[:,-1:],k[:,-1:],v[:,-1:],**step_args)
  reference_step,reference_step_state=torch_recurrent_gated_delta_rule(q[:,-1:],k[:,-1:],v[:,-1:],**step_args)
  torch.testing.assert_close(step.float(),reference_step.float(),rtol=.08,atol=.015)
  torch.testing.assert_close(step_state.float(),reference_step_state.float(),rtol=.08,atol=.015)
  timings={}
  if a.benchmark:
   for name,kernel in [('torch',torch_chunk_gated_delta_rule),('fla',chunk_gated_delta_rule)]:
    measured=[]
    for i in range(12):
     torch.cuda.synchronize();tick=time.perf_counter()
     kernel(q,k,v,g,beta,**kwargs);torch.cuda.synchronize()
     if i>=2:measured.append((time.perf_counter()-tick)*1000)
    timings[name]=statistics.median(measured)
  row={'heads':a.heads,'benchmark_median_ms':timings,'kind':'gated_delta_rule','dtype':a.dtype,'length':length,'max_abs_difference':diff.max().item(),'mean_abs_difference':diff.mean().item(),'state_max_abs_difference':(state-reference_state).abs().max().item(),'continuation_passed':True,'single_token_passed':True,'passed':True};out.append(row);print(row,flush=True)
 try:from causal_conv1d import causal_conv1d_fn
 except ImportError:causal_conv1d_fn=None
 if causal_conv1d_fn:
  for length in lengths:
   x=torch.randn(1,128,length,device='cuda',dtype=dtype);w=torch.randn(128,4,device='cuda',dtype=dtype)
   actual=causal_conv1d_fn(x,w,activation='silu')
   reference=torch.nn.functional.silu(torch.nn.functional.conv1d(x,w[:,None],padding=3,groups=128)[...,:length])
   torch.testing.assert_close(actual.float(),reference.float(),rtol=.08,atol=.04)
   row={'kind':'causal_conv1d','dtype':a.dtype,'length':length,'max_abs_difference':(actual-reference).abs().max().item(),'passed':True};out.append(row);print(row,flush=True)
output = a.output or Path(__file__).resolve().parents[1] / 'build' / 'kernel-smoke.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(out, indent=2) + '\n')
