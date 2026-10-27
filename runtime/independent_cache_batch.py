"""Independent-history prefill; never write padded final state back.

Full-attention history is left padded and explicitly masked. Convolution and
recurrent states are stacked as independent rows. Query RoPE positions retain
each row's logical history length. Right-padded suffix outputs/states are dropped;
native cached GDN does not preserve a shorter row's state through padding.
"""
import copy
import math
import time
import torch


class IncompatibleHistory(ValueError):
    pass


def tensor_like(value):
    return isinstance(value,torch.Tensor) or type(value).__name__=='FrozenTensor'


def copy_tensor(destination, source):
    if isinstance(source,torch.Tensor):destination.copy_(source)
    else:
        for block,target in source.segments(destination):target.copy_(block)


def pack_states(entries, device):
    if not entries:
        raise IncompatibleHistory('empty_history_batch')
    caches=[e['cache'] for e in entries]
    lengths=[len(e['ids']) for e in entries]
    if min(lengths)<1 or any(getattr(c,'offloading',False) for c in caches):
        raise IncompatibleHistory('missing_or_offloading_history')
    layers=[getattr(c,'layers',None) for c in caches]
    if any(v is None or len(v)!=len(layers[0]) for v in layers):
        raise IncompatibleHistory('cache_layout_mismatch')
    result=copy.copy(caches[0]);result.layers=[]
    maximum=max(lengths);batch=len(entries);bytes_used=0
    for index in range(len(layers[0])):
        sources=[v[index] for v in layers]
        if any(type(s) is not type(sources[0]) for s in sources):
            raise IncompatibleHistory('layer_type_mismatch')
        layer=copy.copy(sources[0])
        kind=type(layer).__name__
        if kind not in {'DynamicLayer','LinearAttentionLayer'}:
            raise IncompatibleHistory('unsupported_layer:'+kind)
        flags=('is_initialized',) if kind=='DynamicLayer' else (
            'has_previous_state','is_conv_states_initialized','is_recurrent_states_initialized')
        if any(getattr(s,flag,False) is not True for s in sources for flag in flags):
            raise IncompatibleHistory('uninitialized_history_layer')
        names=('keys','values') if kind=='DynamicLayer' else ('conv_states','recurrent_states')
        for name in names:
            values=[getattr(s,name,None) for s in sources]
            if any(not tensor_like(t) or t.shape[0]!=1 or t.dtype!=values[0].dtype for t in values):
                raise IncompatibleHistory('uninitialized_or_dtype_mismatch:'+name)
            if kind=='DynamicLayer':
                if any(t.ndim!=4 or t.shape[-2]!=n or t.shape[1]!=values[0].shape[1]
                       or t.shape[-1]!=values[0].shape[-1] for t,n in zip(values,lengths)):
                    raise IncompatibleHistory('kv_shape_mismatch')
                merged=torch.zeros((batch,values[0].shape[1],maximum,values[0].shape[-1]),dtype=values[0].dtype,device=device)
                for row,(tensor,n) in enumerate(zip(values,lengths)):
                    copy_tensor(merged[row:row+1,:,maximum-n:,:],tensor)
            else:
                if any(t.shape[1:]!=values[0].shape[1:] for t in values):
                    raise IncompatibleHistory('linear_state_shape_mismatch')
                merged=torch.empty((batch,*values[0].shape[1:]),dtype=values[0].dtype,device=device)
                for row,tensor in enumerate(values):copy_tensor(merged[row:row+1],tensor)
            setattr(layer,name,merged);bytes_used+=merged.numel()*merged.element_size()
        layer.device=torch.device(device)
        if hasattr(layer,'max_batch_size'):layer.max_batch_size=batch
        result.layers.append(layer)
    if hasattr(result,'_seen_tokens'):result._seen_tokens=maximum
    return result,lengths,bytes_used


def continuation_inputs(records, entries, pad_token_id, device, dtype):
    if not records or len(records)!=len(entries):
        raise IncompatibleHistory('empty_or_mismatched_rows')
    prefixes=[len(e['ids']) for e in entries]
    suffixes=[len(r.input_ids)-p for r,p in zip(records,prefixes)]
    if len(records)!=len(entries) or min(suffixes)<1:
        raise IncompatibleHistory('invalid_suffix_lengths')
    for record,entry,p in zip(records,entries,prefixes):
        if getattr(record,'media',None) is not None or tuple(record.input_ids[:p])!=tuple(entry['ids']):
            raise IncompatibleHistory('media_or_token_ancestry_mismatch')
    batch=len(records);past=max(prefixes);query=max(suffixes);keys=past+query
    ids=torch.full((batch,query),pad_token_id,dtype=torch.long,device=device)
    for row,(r,p,n) in enumerate(zip(records,prefixes,suffixes)):
        ids[row,:n]=torch.tensor(r.input_ids[p:],dtype=torch.long,device=device)
    logical=torch.tensor(prefixes,device=device)[:,None]+torch.arange(query,device=device)[None,:]
    positions=logical[None].expand(3,-1,-1)
    # Efficient SDPA requires aligned bias row strides even when the logical
    # key width is not a multiple of eight. The view keeps physical alignment.
    aligned=math.ceil(keys/8)*8
    bias=torch.full((batch,1,query,aligned),float('-inf'),dtype=dtype,device=device)
    key_index=torch.arange(keys,device=device)[None,None,:]
    query_index=torch.arange(query,device=device)[None,:,None]
    p=torch.tensor(prefixes,device=device)[:,None,None]
    n=torch.tensor(suffixes,device=device)[:,None,None]
    history=(key_index>=past-p)&(key_index<past)
    new=(key_index>=past)&(key_index<past+n)&(key_index<=past+query_index)
    bias[:,:,:,:keys].masked_fill_((history|new)[:,None],0)
    return ids,positions,bias[:,:,:,:keys],prefixes,suffixes


@torch.inference_mode()
def continue_backbone(backbone,records,entries,pad_token_id,device,*,native_single_mask=False):
    started=time.perf_counter()
    if not entries or any(not e.get('chunks') for e in entries):
        raise IncompatibleHistory('missing_hidden_history')
    dtype=entries[0]['chunks'][0].dtype
    ids,positions,bias,prefixes,suffixes=continuation_inputs(records,entries,pad_token_id,device,dtype)
    for entry,p in zip(entries,prefixes):
        if sum(c.shape[1] for c in entry['chunks'])!=p:
            raise IncompatibleHistory('missing_prefix_hidden')
        if any(c.shape[0]!=1 or c.dtype!=dtype for c in entry['chunks']):
            raise IncompatibleHistory('hidden_layout_mismatch')
    past,prefixes,state_bytes=pack_states(entries,device)
    if native_single_mask:
        if len(records)!=1:raise IncompatibleHistory('single_native_mask_requires_one_row')
        attention=torch.ones((1,prefixes[0]+suffixes[0]),dtype=torch.long,device=device)
    else:attention=bias
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
    assembly_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter()
    output=backbone(input_ids=ids,position_ids=positions,attention_mask=attention,
                    past_key_values=past,use_cache=True,return_dict=True)
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
    language_ms=(time.perf_counter()-started)*1000
    batch=len(records);maximum=max(len(r.input_ids) for r in records)
    hidden=torch.zeros((batch,maximum,output.last_hidden_state.shape[-1]),dtype=dtype,device=device)
    full_ids=torch.full((batch,maximum),pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros((batch,maximum),dtype=torch.long,device=device)
    for row,(record,entry,p,n) in enumerate(zip(records,entries,prefixes,suffixes)):
        cursor=0
        for chunk in entry['chunks']:
            if chunk.shape[0]!=1:raise IncompatibleHistory('hidden_batch_axis_mismatch')
            stop=cursor+chunk.shape[1];copy_tensor(hidden[row:row+1,cursor:stop],chunk);cursor=stop
        if cursor!=p:raise IncompatibleHistory('missing_prefix_hidden')
        hidden[row,p:p+n].copy_(output.last_hidden_state[row,:n])
        full_ids[row,:p+n]=torch.tensor(record.input_ids,dtype=torch.long,device=device)
        mask[row,:p+n]=1
    # Padded final recurrence/conv states cannot be safely retained. Only the
    # original snapshots remain cacheable; this combined working cache dies here.
    del output,past
    return hidden,full_ids,mask,{'cache_assembly_ms':assembly_ms,'language_ms':language_ms,
        'working_state_mib':state_bytes/2**20,'reused_tokens':prefixes,
        'suffix_tokens':suffixes,'physical_history_tokens':max(prefixes),
        'physical_suffix_tokens':max(suffixes),
        'history_padding_tokens':sum(max(prefixes)-p for p in prefixes),
        'suffix_padding_tokens':sum(max(suffixes)-n for n in suffixes)}


from contextlib import contextmanager
from collections import Counter


def plan_histories(engine, model, records, boundaries, checkpoints, input_keys, admission, enabled):
    """Metadata only. Preserve every native checkpoint/canonical forward cut."""
    def reject(reason):return {'reason':reason}
    if not enabled or engine.max_entries <= 0:return reject('independent_cache_disabled')
    if len(records)<2 or any(getattr(r,'media',None) is not None for r in records):
        return reject('independent_text_only')
    if not (len(records)==len(boundaries)==len(checkpoints)==len(input_keys)):
        return reject('independent_row_metadata')
    keys=[];prefixes=[];suffixes=[];width=0
    for record,boundary,cuts,input_key in zip(records,boundaries,checkpoints,input_keys):
        key,common=engine._match(model,record.input_ids,boundary,input_key,False,None,0)
        entry=engine._lookup(key) if key is not None else None
        if entry is None:return reject('independent_cache_miss')
        if not entry.get('chunks'):return reject('independent_missing_hidden')
        width=max(width,torch.finfo(entry['chunks'][0].dtype).bits//8)
        p=len(entry['ids']);n=len(record.input_ids)-p
        if n<1 or tuple(record.input_ids[:p])!=tuple(entry['ids']):return reject('independent_ancestry')
        # The native path may build a promoted boundary or an explicit context
        # checkpoint before the question suffix. Do not elide those passes.
        whole=engine._input_key(model,record.input_ids,boundary,input_key,False,None)
        points=set(cuts)
        if whole in engine.observations or common<engine.min_prefix_tokens:points.add(boundary)
        if common>=p+engine.min_prefix_tokens:points.add(common)
        if engine.checkpoint_tokens:points.update(range(engine.checkpoint_tokens,boundary,engine.checkpoint_tokens))
        points={x for x in points if p<x<=boundary and (x>=engine.min_prefix_tokens or x==boundary or x in cuts)}
        prior=max([p,*[x for x in points if x<boundary]])
        if boundary in points and prior and boundary-prior<engine.min_prefix_tokens:points.remove(boundary)
        if points:return reject('independent_checkpoint_required')
        keys.append(key);prefixes.append(p);suffixes.append(n)
    if admission.reason([len(r.input_ids) for r in records]):return reject(admission.reason([len(r.input_ids) for r in records]))
    if min(prefixes)/max(prefixes)<.65 or min(suffixes)/max(suffixes)<admission.length_ratio:
        return reject('independent_padding_ratio')
    physical=len(records)*(max(prefixes)+max(suffixes))
    if physical>admission.padded_tokens:return reject('independent_padded_token_budget')
    # Cover packed KV, GDN temporary state, compact hidden vectors and the
    # explicit bias. Protected source storage remains charged to the banks.
    projected=admission.workspace([max(prefixes)+max(suffixes)]*len(records),max(prefixes))
    bias=len(records)*max(suffixes)*math.ceil((max(prefixes)+max(suffixes))/8)*8*width
    projected+=2*bias  # mask plus construction/output overlap
    return {'reason':None,'keys':keys,'prefixes':prefixes,'suffixes':suffixes,'physical_tokens':physical,'workspace':projected}


@contextmanager
def lease_histories(engine, keys):
    """Pin retained GPU/CPU sources until packing/head work releases its refs."""
    if not hasattr(engine,'_leased_keys'):engine._leased_keys=Counter()
    with engine.host.lock:
        entries=[engine._lookup(k) for k in keys]
        if any(e is None for e in entries):raise IncompatibleHistory('independent_lease_miss')
        for key in keys:
            engine._leased_keys[key]+=1
            engine.host.pins[key]+=1
        tiers=['gpu' if k in engine.entries else 'cpu' for k in keys]
    try:
        yield entries,tiers
    finally:
        entries.clear()
        with engine.host.lock:
            for key in keys:
                engine._leased_keys[key]-=1
                engine.host.pins[key]-=1
                if not engine._leased_keys[key]:del engine._leased_keys[key]
                if not engine.host.pins[key]:del engine.host.pins[key]


@torch.inference_mode()
def forward_histories(engine, model, processor, records, boundaries, input_keys, plan, entries, tiers):
    hidden,ids,mask,stats=continue_backbone(model.language_model.model.language_model,
        records,entries,processor.tokenizer.pad_token_id,next(model.parameters()).device)
    from optimized_inference import lexical_weight
    started=time.perf_counter()
    logits=model.head(hidden,ids,mask,records,lexical_weight(model))
    torch.cuda.current_stream().synchronize()
    head_ms=(time.perf_counter()-started)*1000
    if any(not torch.isfinite(v).all() for row in logits for v in row):raise IncompatibleHistory('nonfinite_independent_head')
    usage=[]
    for record,boundary,input_key,key,entry,tier,p in zip(records,boundaries,input_keys,plan['keys'],entries,tiers,plan['prefixes']):
        engine.frequency.touch(entry)
        if key in engine.entries:engine.entries.move_to_end(key)
        # GPU and CPU snapshots can share a key but are distinct metadata.
        if tier=='gpu':engine.host.touch(key)
        else:engine.host.hits+=1
        engine.telemetry.record('hit',tier,key,entry=entry,independent_batch=True)
        engine.hits+=1;engine.reused_tokens+=p
        whole=engine._input_key(model,record.input_ids,boundary,input_key,False,None)
        engine._remember(whole,record.input_ids,boundary,input_key,False)
        usage.append({'prefix_cache':'hit','prefix_cache_tier':tier,'reused_prefix_tokens':p,
            'prefix_checkpoints_saved':0,'prefix_build_ms':0.,'prefill_mode':'independent_cache_batch',
            'prefix_restore_ms':0.,'prefix_offload_ms':0.,'image_feature_cache_hits':0,
            'image_feature_cache_misses':0,'vision_ms':0.,'language_ms':round(stats['language_ms'],1),
            'head_ms':round(head_ms,1),'cache_assembly_ms':round(stats['cache_assembly_ms'],1),
            'independent_history_padding_tokens':stats['history_padding_tokens'],
            'independent_suffix_padding_tokens':stats['suffix_padding_tokens'],
            'independent_working_state_mib':round(stats['working_state_mib'],1)})
    return logits,usage
