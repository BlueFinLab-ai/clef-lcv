"""CPU checks for ordered feature insertion, split images, positions and pooling."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'runtime'),str(root/'vendor/cloudflare')]
import torch
from chunked_prefill import forward_chunked, image_features
from joint_schema_model import EncodedRecord

class Visual:
    dtype=torch.float32
    def __call__(self,pixels,grid_thw,return_dict):
        pieces=[];cursor=0
        for grid in grid_thw.tolist():
            count=grid[0]*grid[1]*grid[2]
            marker=pixels[cursor,0]
            pieces.append(marker+torch.arange(count//4,dtype=torch.float32).view(-1,1).expand(-1,4))
            cursor+=count
        return SimpleNamespace(pooler_output=torch.cat(pieces))

class Language:
    def __init__(self,base): self.base=base;self.positions=[]
    def __call__(self,*,attention_mask,position_ids,past_key_values,use_cache,return_dict,
                 inputs_embeds=None,input_ids=None):
        start=past_key_values or 0
        hidden=inputs_embeds if inputs_embeds is not None else self.base.embedding(input_ids)
        end=start+hidden.shape[1]
        assert attention_mask.shape[1]==end
        torch.testing.assert_close(position_ids,self.base.expected_positions[:,:,start:end])
        self.positions.append(position_ids.clone())
        return SimpleNamespace(last_hidden_state=hidden+position_ids[1].unsqueeze(-1),past_key_values=end)

class Base:
    def __init__(self):
        self.visual=Visual();self.embedding=torch.nn.Embedding(10,4)
        self.language_model=Language(self)
    def get_input_embeddings(self): return self.embedding
    def get_rope_index(self,ids,**kwargs):
        assert kwargs['mm_token_type_ids'].shape==ids.shape
        return self.expected_positions,None

class Model(torch.nn.Module):
    def __init__(self,base,encoded):
        super().__init__();self.embedding=base.embedding;self.encoded=encoded
        self.language_model=SimpleNamespace(model=base,config=SimpleNamespace(image_token_id=9))
        self.captured=None
    def head(self,hidden,ids,mask,records,weight):
        assert records==[self.encoded]
        assert ids.tolist()[0]==list(self.encoded.input_ids)
        self.captured=hidden.clone()
        return [[hidden.sum().reshape(1)]]

ids=(1,2,9,9,9,9,3,9,9,4)
media={'image_grid_thw':torch.tensor([[1,4,4],[1,2,4]]),
       'pixel_values':torch.cat([torch.full((16,1),100.),torch.full((8,1),200.)]),
       'mm_token_type_ids':[0,0,1,1,1,1,0,1,1,0], 'token_offset':0}
encoded=EncodedRecord(ids,(),'fixture',media)
base=Base();base.expected_positions=torch.arange(len(ids)).view(1,1,-1).expand(3,1,-1).clone()
# Spatial coordinates differ from sequential positions; detect accidental reset.
base.expected_positions[1,0,2:6]=torch.tensor([20,20,21,21])
base.expected_positions[2,0,2:6]=torch.tensor([30,31,30,31])
model=Model(base,encoded);processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
features=image_features(base,encoded,torch.device('cpu'))
all_at_once=image_features(base,encoded,torch.device('cpu'),batch_images=0)
torch.testing.assert_close(features,all_at_once)
assert features[:,0].tolist()==[100.,101.,102.,103.,200.,201.]
full_ids=torch.tensor([ids]);expected=base.embedding(full_ids)
expected=expected.masked_scatter((full_ids==9).unsqueeze(-1).expand_as(expected),features)
expected+=base.expected_positions[1].unsqueeze(-1)
with patch('torch.cuda.synchronize'):
    _,usage=forward_chunked(model,processor,encoded,None,initial_chunk_tokens=3,chunk_tokens=2)
torch.testing.assert_close(model.captured,expected)
assert usage['image_chunk_splits']==2 and usage['prefill_chunks']==5
assert torch.cat(base.language_model.positions,dim=2).equal(base.expected_positions)
pooled_media={**media,'original_grid_thw':media['image_grid_thw'],
              'image_grid_thw':torch.tensor([[1,2,2],[1,2,2]])}
pooled=image_features(base,EncodedRecord(ids,(),'pooled',pooled_media),torch.device('cpu'),pooling=True)
assert pooled[:,0].tolist()==[101.5,200.5]
print('PASS: ordered images, bounded batches, pooled grids, mid-image chunk splits, global positions, and complete head input')

# Resume across image boundaries with independent state and all head inputs.
snapshot={}
def capture(end,past,hidden):
    snapshot.update(cache=past,chunks=(hidden.clone(),),ids=ids[:end])
base.language_model.positions=[]
with patch('torch.cuda.synchronize'):
    forward_chunked(model,processor,encoded,None,initial_chunk_tokens=3,chunk_tokens=2,
                    checkpoint_positions=(6,),checkpoint_callback=capture)
assert snapshot['ids']==ids[:6]
with patch('torch.cuda.synchronize'):
    forward_chunked(model,processor,encoded,None,initial_chunk_tokens=3,chunk_tokens=2,
                    prefix_entry=snapshot)
torch.testing.assert_close(model.captured,expected)
# After the complete image region, warm requests must skip vision entirely.
from dataclasses import replace
last=replace(encoded,media={**media,'mm_token_type_ids':media['mm_token_type_ids'][:9]})
model.encoded=last
entry={'cache':9,'chunks':(expected[:,:9].clone(),),'ids':ids[:9]}
def forbidden(*args):raise AssertionError('Cached image prefix re-encoded vision')
with patch('torch.cuda.synchronize'):
    _,usage=forward_chunked(model,processor,last,None,initial_chunk_tokens=3,chunk_tokens=2,
        prefix_entry=entry,features_provider=forbidden)
assert usage['vision_prefix_reused'] and usage['prefill_chunks']==1
torch.testing.assert_close(model.captured,expected)
print('PASS: prefix resume, independent state, retained head input and skipped vision')
