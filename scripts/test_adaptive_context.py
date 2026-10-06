"""CPU policy tests: total context, model/GPU budgets, learning and contention."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from adaptive_context import AdaptiveContext,MIB
from context_budget import ContextBudget

def policy(full=False,**kw):
    config=NS(max_position_embeddings=262144,num_hidden_layers=64 if full else 32,
        layer_types=['full_attention']*(16 if full else 8),num_key_value_heads=4,
        num_attention_heads=24 if full else 16,head_dim=256,hidden_size=5120 if full else 4096)
    native=ContextBudget(config,dtype_bytes=2,calibrated=True,configured_limit=8192,image_limit=None)
    return AdaptiveContext(config,1024,native_budget=native,fingerprint={'profile':full},**kw)

p=policy();host=80*1024*MIB
assert p.choose(8192,2600*MIB,host)['mode']=='none'
assert p.choose(32768,2600*MIB,host)['mode']=='kv_gpu'
assert p.choose(65536,2600*MIB,host)['mode']=='kv_stream'
assert p.choose(131072,2600*MIB,host)['mode']=='kv_stream'
assert p.choose(131072,2600*MIB,None) is None
assert p.choose(131072,2600*MIB,1024*MIB) is None
assert p.choose(131072,2600*MIB,5120*MIB,cached_tokens=130800)['mode']=='kv_stream'
assert p.limits(2600*MIB,5120*MIB,cached_tokens=130800)['kv_stream']>p.limits(2600*MIB,5120*MIB)['kv_stream']
assert p.choose(32768,20000*MIB,host)['mode']=='none'
# Larger model needs a lower resident limit with the same free working budget.
assert p.limits(6400*MIB,host)['kv_gpu']>policy(True).limits(6400*MIB,host)['kv_gpu']
assert p.limits(20000*MIB,host)['kv_gpu']==262144
# Warm hits must not lower cold-workspace estimates.
original=p.workspace('kv_gpu',32768)
p.observe('kv_gpu',32768,1500*MIB,reused_tokens=32000)
assert p.workspace('kv_gpu',32768)==original
p.observe('kv_gpu',32768,2100*MIB)
assert p.samples['kv_gpu']['max_cold_tokens']==32768
assert p.workspace('kv_gpu',32768)<original
p.record_oom('kv_gpu',49152,2600*MIB)
assert p.choose(49152,2600*MIB,host)['mode']=='kv_stream'
assert p.choose(49152,5000*MIB,host)['mode'] in {'none','kv_gpu'}
assert policy(stream_enabled=False).choose(65536,2600*MIB,host) is None
with tempfile.TemporaryDirectory() as d:
    file=Path(d)/'calibration.json'
    p=policy(calibration_path=file);p.observe('kv_gpu',32768,2100*MIB)
    q=policy(calibration_path=file);assert q.samples==p.samples
    assert not policy(True,calibration_path=file).samples
print('PASS: adaptive budgets, all-context accounting, cold-only learning, OOM contention backoff and fingerprinted persistence')
