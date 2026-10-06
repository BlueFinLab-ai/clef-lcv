"""Isolated real-checkpoint test of the existing native-layout GEMV-M kernel."""
import ctypes,json,os,time
from pathlib import Path
import torch
import bitsandbytes as bnb
from safetensors import safe_open
from bitsandbytes.functional import QuantState

torch.set_num_threads(4)
assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx803'
torch.cuda.tunable.set_filename('/app/tuning/gfx803.csv')
assert torch.cuda.tunable.read_file('/app/tuning/gfx803.csv')
torch.cuda.tunable.tuning_enable(False);torch.cuda.tunable.enable(True)
path=os.environ.get('CLEF_GFX803_GEMV_LIBRARY','/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/libgfx803gemv_m.so')
lib=ctypes.CDLL(path)
launch=lib.gfx803_gemv_m_launch
launch.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p];launch.restype=None
hip=ctypes.CDLL('libamdhip64.so');hip.hipGetLastError.argtypes=[];hip.hipGetLastError.restype=ctypes.c_int
prefix='model.language_model.layers.0.mlp.gate_proj.weight'
with safe_open('/checkpoint/model.safetensors',framework='pt',device='cpu') as f:
    packed=f.get_tensor(prefix).cuda()
    stats={k[len(prefix)+1:]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix+'.')}
state=QuantState.from_dict(stats,device=torch.device('cuda'))
weight=bnb.functional.dequantize_4bit(packed,state)
report={'completed':False,'shape':list(weight.shape),'source':'Pinned community ROCm image, existing libgfx803gemv_m.so; native N×K weights, tiles of at most 16 input rows. No persistent FP16 cache.','rows':[]}
def save():Path('/bench/gfx803-native-layout-results.json').write_text(json.dumps(report,indent=2,allow_nan=False))

def native(x,w):
    m,k=x.shape;n=w.shape[0]
    out=torch.full((m,n),float('nan'),device=x.device,dtype=x.dtype)
    for offset in range(0,m,16):
        rows=min(16,m-offset)
        if rows==1:out[offset:]=torch.nn.functional.linear(x[offset:],w)
        else:
            launch(x.data_ptr()+offset*k*2,w.data_ptr(),out.data_ptr()+offset*n*2,rows,n,k,
                   torch.cuda.current_stream().cuda_stream)
            status=hip.hipGetLastError()
            if status:raise RuntimeError(f'HIP native-layout launch failed: {status}')
    return out

def nf4(x):
    w=bnb.functional.dequantize_4bit(packed,state)
    return native(x,w)

def measured(fn):
    fn();torch.cuda.synchronize();started=time.perf_counter()
    for _ in range(3):out=fn()
    torch.cuda.synchronize();return (time.perf_counter()-started)*1000/3,out

torch.manual_seed(20261004)
with torch.inference_mode():
    for m in [2,8,16,128,512,1024]:
        x=torch.randn(m,weight.shape[1],device='cuda',dtype=torch.float16)*.02
        reference=torch.nn.functional.linear(x,weight)
        for name,fn in [('bnb_nf4',lambda:bnb.matmul_4bit(x,packed,quant_state=state)),
                        ('fp16_rocblas',lambda:torch.nn.functional.linear(x,weight)),
                        ('native_layout_fp16',lambda:native(x,weight)),('native_layout_nf4',lambda:nf4(x))]:
            ms,out=measured(fn);delta=(out.float()-reference.float()).abs()
            row={'tokens':m,'method':name,'ms':ms,'max_abs_error':float(delta.max()),
                 'relative_l2_error':float(delta.norm()/reference.float().norm()),'finite':bool(out.isfinite().all())}
            row['passed']=row['finite'] and row['relative_l2_error']<.002
            report['rows'].append(row);save();print('NATIVE',row,flush=True);assert row['passed']
report['completed']=True;save()
