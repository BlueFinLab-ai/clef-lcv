"""Run the actual single-request handler around mock GPU calls after CPU splitting."""
import ast,contextlib,gc,hashlib,json,logging,sys,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
from fastapi import HTTPException
from fastapi.responses import JSONResponse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from cpu_preparation import PreparedDecision
source=Path(__file__).resolve().parents[1]/'clef_service/app.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'prepare_decision','run_decision'}]
class OOM(RuntimeError):pass
class Engine:
 base_bytes=0
 context_policy=None
 def __init__(self):self.calls=[];self.finishes=0
 def prepare_request(self,*a,**kw):return {}
 def forward(self,*a,**kw):self.calls.append(kw);return [[0]],{'prefill_mode':'single_pass','reused_prefix_tokens':0}
 def finish_request(self):self.finishes+=1
 def _bytes(self):return 0
 def clear(self):pass
engine=Engine();cache=NS(usage=lambda:dict(image_hits=0,image_misses=0,token_hits=0,token_misses=0))
namespace={'DecisionRequest':NS,'PreparedDecision':PreparedDecision,'time':time,'lock':threading.Lock(),
 'request_queue':NS(nesting=False),'offer_short_request':lambda *a:None,'gpu_profiles':threading.local(),
 'request_record':lambda r:{'state':r.state,'images':r.images,'questions':{}},'input_cache':cache,'INPUT_CACHE_ENABLED':True,
 'processor':NS(image_processor=NS(patch_size=16)),
 'encode_compact':lambda *a,**kw:(NS(input_ids=(1,)*64,media=None),40,()),
 'MAX_VISION_PATCH_TOKENS':0,'refresh_context_budget':lambda:{'max_input_tokens':8192,'max_input_tokens_with_images':8192,'context_limit':{'mode':'configured'}},
 'context_budget':NS(computed=False,image_computed=False),'engine':engine,
 'torch':NS(cuda=NS(reset_peak_memory_stats=lambda:None,OutOfMemoryError=OOM,synchronize=lambda:None,max_memory_allocated=lambda:1024,empty_cache=lambda:None)),
 'STARTUP_STRATEGY':{'runtime_backend':'cuda'},'PREFILL_INITIAL_CHUNK_TOKENS':8192,'PREFILL_CHUNK_TOKENS':4096,
 'PREFIX_CACHE_ENABLED':True,'GPU_CACHE_SUPPORTED':True,'ATTENTION_BACKEND':'efficient','IMAGE_PREFILL_ENABLED':True,
 'sdpa_kernel':lambda *a:contextlib.nullcontext(),'SDPBackend':NS(EFFICIENT_ATTENTION=1,MATH=2),'nullcontext':contextlib.nullcontext,
 'hashlib':hashlib,'json':json,'HTTPException':HTTPException,'JSONResponse':JSONResponse,'model':None,
 'answers':lambda *a:{'ok':True},'ROCM_LONG_PROMPT_THRESHOLD':0,'gc':gc,'logging':logging}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
for enabled in [True,False]:
 request=NS(state='input',images=[],input_cache=enabled,prefix_cache=enabled,image_pooling=False,model='test')
 row=namespace['run_decision'](request)
 assert row['answers']=={'ok':True} and engine.calls[-1]['feature_cache_enabled']==enabled
 prepared=namespace['prepare_decision'](request,True)
 row=namespace['run_decision'](prepared)
 assert row['usage']['cpu_preparation_overlapped'] and engine.calls[-1]['cache_enabled']==enabled
assert engine.finishes==4
print('PASS: actual inline/prepared single handler, input/prefix cache opt-outs, usage preservation and worker cleanup')
