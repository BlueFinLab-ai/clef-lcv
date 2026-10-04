"""CPU checks for multi-image span remapping and bounded cache accounting."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/cloudflare'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import torch
from optimized_inference import pool_record, tensor_bytes, InferenceEngine, available_cuda_memory
from joint_schema_model import EncodedRecord, EncodedQuestion

image_id = 99
prefix = [10,11]
media_ids = [image_id]*64 + [12,13] + [image_id]*24 + [14]
state = [15,16,17]
schema = [20,21,22,23,24,25]
ids = prefix + media_ids + state + schema
boundary = len(prefix + media_ids + state)
question = EncodedQuestion('q', 1, (boundary,boundary+2), ((boundary+2,boundary+4),(boundary+4,boundary+6)), ('a','b'))
media = {'token_offset':2, 'image_grid_thw':torch.tensor([[1,16,16],[1,8,12]]),
         'mm_token_type_ids':[1]*64+[0,0]+[1]*24+[0]}
encoded = EncodedRecord(tuple(ids),(question,),'test',media)
pooled,new_boundary = pool_record(encoded,boundary,image_id)
assert len(encoded.input_ids)-len(pooled.input_ids) == 66
assert pooled.input_ids.count(image_id) == 22
assert new_boundary == boundary-66
assert pooled.media['image_grid_thw'].tolist() == [[1,8,8],[1,4,6]]
assert pooled.media['mm_token_type_ids'].count(1) == 22
assert pooled.media['original_grid_thw'].tolist() == [[1,16,16],[1,8,12]]
assert encoded.media['image_grid_thw'].tolist() == [[1,16,16],[1,8,12]]
for old,new in zip(question.option_spans,pooled.questions[0].option_spans):
    assert encoded.input_ids[old[0]:old[1]] == pooled.input_ids[new[0]:new[1]]
assert pooled.input_ids[new_boundary:] == encoded.input_ids[boundary:]
storage = torch.zeros(100,dtype=torch.float32)
assert tensor_bytes({'whole':storage,'view':storage[:10]}) == 400
engine=InferenceEngine(1,1)
engine.entries['a']={'cache':{'keys':storage},'chunks':(),'ids':tuple(range(100)),'bytes':400}
assert engine._bytes()==400 and engine.stats()['mib']<1
engine.clear();assert engine.stats()['entries']==0
original = (torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated)
try:
    torch.cuda.mem_get_info = lambda: (100,1000)
    torch.cuda.memory_reserved = lambda: 400
    torch.cuda.memory_allocated = lambda: 250
    assert available_cuda_memory() == 250
finally:
    torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated = original
print('PASS: independent image grids, schema spans, boundary, immutable metadata, shared-storage accounting')
