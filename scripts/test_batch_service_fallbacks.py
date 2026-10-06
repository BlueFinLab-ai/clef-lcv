"""CPU fault injection into the production batch worker, without GPU imports."""
import ast,contextlib,gc,hashlib,json,logging,sys,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
from fastapi import HTTPException
from fastapi.responses import JSONResponse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from text_batching import BatchUnavailable,TextBatchAdmission,shared_tokens
from cpu_preparation import PreparedDecision
source=Path(__file__).resolve().parents[1]/'clef_service/app.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_decision_batch')
code=compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec')
class OOM(RuntimeError):pass
cfg=NS(layer_types=['full_attention']*8+['linear_attention']*24,num_key_value_heads=4,
 head_dim=256,linear_num_value_heads=32,linear_key_head_dim=128,linear_value_head_dim=128,
 hidden_size=4096,num_attention_heads=16)
class Engine:
    base_bytes=0;prefill_reserve_bytes=512*2**20
    def __init__(self):self.calls=[];self.clears=0;self.finishes=0
    def _bytes(self):return 0
    def prepare_request(self,*a,**kw):return {}
    def plan_text_batch(self,*a,**kw):return {"enabled":True,"key":None,"reason":None}
    def clear(self):self.clears+=1
    def finish_request(self):self.finishes+=1
    def forward_text_batch(self,m,p,records,*a,**kw):
        self.calls.append(len(records))
        if len(records)==4:raise OOM('forced aggregate OOM')
        return [[0]]*len(records),{'reused_prefix_tokens':0,'prefix_cache':'disabled'}

def run(requests):
    engine=Engine();admission=TextBatchAdmission(cfg,padded_tokens=12000)
    singles=[];emitted=[];cache=NS(stats=lambda:{'token_hits':0,'token_misses':0})
    def record(r):
        if r.bad:raise HTTPException(400,'bad member')
        return {'state':r.value}
    def encode(p,r,**kw):return NS(input_ids=tuple([1]*r['state']),media=None),r['state']-1,()
    def single(r):singles.append(r.value);return {'model':'test','answers':{'ok':True},'usage':{}}
    def prepare(r):
        rec=record(r);e,b,p=encode(None,rec)
        return PreparedDecision(r,rec,e,b,p,{'token_cache_hits':0,'token_cache_misses':0},'text')
    namespace={'time':time,'app':NS(state=NS(text_batching=admission)),'lock':threading.Lock(),
      'request_record':record,'input_cache':cache,'INPUT_CACHE_ENABLED':True,'processor':None,
      'encode_compact':encode,'refresh_context_budget':lambda:{'max_input_tokens':8192},
      'context_budget':NS(mode='configured'),'HTTPException':HTTPException,'JSONResponse':JSONResponse,
      'shared_tokens':shared_tokens,'available_cuda_memory':lambda:16*2**30,'engine':engine,
      'torch':NS(cuda=NS(reset_peak_memory_stats=lambda:None,OutOfMemoryError=OOM,
          empty_cache=lambda:None,synchronize=lambda:None,max_memory_allocated=lambda:2**30)),
      'sdpa_kernel':lambda *a:contextlib.nullcontext(),'SDPBackend':NS(EFFICIENT_ATTENTION=1),
      'PREFIX_CACHE_ENABLED':True,'GPU_CACHE_SUPPORTED':True,'hashlib':hashlib,'json':json,
      'model':None,'answers':lambda *a:{'ok':True},'BatchUnavailable':BatchUnavailable,
      'gc':gc,'log':logging.getLogger('batch-test'),'run_decision':single}
    namespace.update(PreparedDecision=PreparedDecision,prepare_decision=prepare)
    exec(code,namespace)
    namespace['run_decision_batch'](requests,lambda i,r:emitted.append((i,r)))
    assert sorted(i for i,r in emitted)==list(range(len(requests)))
    return engine,admission,singles,emitted

def request(n=2000,bad=False):return NS(value=n,bad=bad,prefix_cache=True,input_cache=True,model='test')
e,a,s,rows=run([request() for _ in range(4)])
assert e.calls==[4,2,2] and e.clears==1 and not s and a.oom_fallbacks==1
assert all(r['usage']['batch_size']==2 for _,r in rows)
assert all(r['usage']['accepted_max_input_tokens']==8192 for _,r in rows)
e,a,s,rows=run([request(bad=True),request(),request(9000),request()])
assert isinstance(rows[0][1],HTTPException) and rows[0][1].status_code==400
assert any(isinstance(r,JSONResponse) and r.status_code==413 for _,r in rows)
assert e.calls==[2] and a.requests==2
print('PASS: aggregate OOM splits 4->2 without single context shrink; invalid/oversized members isolated')
