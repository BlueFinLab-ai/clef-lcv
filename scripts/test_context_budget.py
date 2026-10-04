"""CPU tests for estimator accounting, policy fallbacks, and boundary arithmetic."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from context_budget import ContextBudget, MIB


def budget(full=False, calibrated=True, **kw):
    layers = 64 if full else 32
    config = SimpleNamespace(max_position_embeddings=262144,
        num_hidden_layers=layers, layer_types=['full_attention']*(layers//4),
        num_key_value_heads=4, num_attention_heads=24 if full else 16,
        head_dim=256, hidden_size=5120 if full else 4096)
    return ContextBudget(config, dtype_bytes=2, calibrated=calibrated,
        configured_limit=16384 if full else 24576, image_limit=kw.pop('image_limit',8192), **kw)


def snapshot(b, available_mib, cache_mib=0, reserved_mib=0):
    return b.snapshot(free_bytes=(available_mib-cache_mib-reserved_mib)*MIB,
        reserved_bytes=(5000+reserved_mib)*MIB, allocated_bytes=5000*MIB,
        cache_bytes=cache_mib*MIB)

f, big = budget(), budget(True)
assert (f.kv_bytes, f.hidden_bytes) == (32768, 8192)
assert (big.kv_bytes, big.hidden_bytes) == (65536, 10240)
# Allocator slack and reclaimable entries must not reduce the request budget.
a = snapshot(f, 5600)
assert a['max_input_tokens'] == snapshot(f, 5600, 100, 1500)['max_input_tokens']
assert a['max_input_tokens'] > snapshot(f, 2600)['max_input_tokens']
assert snapshot(big, 6400)['max_input_tokens'] > snapshot(f, 2600)['max_input_tokens']
assert a['max_input_tokens'] > f.configured_limit
assert a['context_limit']['configured_cap_applied'] is False
assert a['max_input_tokens_with_images'] == 8192
assert snapshot(budget(image_chunked=True),5600)['context_limit']['image_limit_basis']=='configured_chunked_ceiling'
assert snapshot(f, 512)['max_input_tokens'] == 0
limit = a['max_input_tokens']
assert f.workspace(limit) <= (5600-512)*MIB
assert f.workspace(limit+f.quantum) > (5600-512)*MIB
assert snapshot(f, 100000)['max_input_tokens'] == 262144
f.observe(24576, 2100*MIB)
assert f.reference_workspace == 2100*MIB
f.observe(24576, 100*MIB)
assert f.reference_workspace == 2100*MIB
f.record_oom(45056)
assert snapshot(f, 100000)['max_input_tokens'] == 40960
assert snapshot(budget(mode='fixed'), 5600)['max_input_tokens'] == 24576
f = budget(calibrated=False)
assert snapshot(f, 5600)['context_limit']['computed_max_input_tokens'] is None
print('Context budget arithmetic, allocator accounting, observed cost, OOM ceiling and fallbacks passed')

# Automatic image budgets follow available memory only on calibrated chunked builds.
image_auto=budget(True,image_chunked=True,image_limit=None)
a=snapshot(image_auto,6471)
assert a['max_input_tokens_with_images']==a['max_input_tokens']==45056
assert a['context_limit']['image_limit_basis']=='computed_memory_estimate'
assert a['context_limit']['configured_max_input_tokens_with_images'] is None
assert snapshot(image_auto,5000)['max_input_tokens_with_images']<45056
image_auto.record_oom(45056)
assert snapshot(image_auto,100000)['max_input_tokens_with_images']==40960
for b in (budget(True,image_chunked=False,image_limit=None),
          budget(True,calibrated=False,image_chunked=True,image_limit=None),
          budget(True,mode='fixed',image_chunked=True,image_limit=None)):
    assert snapshot(b,6471)['max_input_tokens_with_images']==16384
assert snapshot(budget(True,image_chunked=True,image_limit=32768),6471)['max_input_tokens_with_images']==32768
print('Automatic image budgets, explicit caps, memory changes, OOM backoff and safe fallbacks passed')
