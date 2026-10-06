"""Incremental mixed text/image prefill with request-local hybrid state.

Images are fully vision-encoded in bounded batches before language prefill. Global
multimodal positions are calculated once, then sliced alongside chunk embeddings.
All token outputs are retained for the unchanged native head. Optional prefix
checkpoints and independent image-feature reuse are supplied by the memory-budgeted
inference engine.
"""
from dataclasses import replace
from media_prefix import remaining_images
import copy
import math
import os
import logging
import time

import torch
import torch.nn.functional as F
import joint_schema_model as joint
from active_context_offload import ActiveKVOffload, check_host_capacity


@torch.inference_mode()
def image_features(base, encoded, device, *, pooling=False, batch_images=1, feature_getter=None, cpu_output=False):
    """Transfer and encode bounded image batches; preserve original image order."""
    media = encoded.media
    if any(k in media for k in ('pixel_values_videos', 'video_grid_thw')):
        raise ValueError('Incremental media prefill supports still images only')
    original = media['original_grid_thw'] if pooling else media['image_grid_thw']
    grids = original.tolist()
    if any(t != 1 for t, _, _ in grids):
        raise ValueError('Incremental media prefill supports still images only')
    pixels = media['pixel_values'].split([math.prod(g) for g in grids])
    size = batch_images if batch_images > 0 else len(grids)
    pieces = []
    new_grids=media['image_grid_thw'].tolist()
    def keep(feature,index):
        if cpu_output and pooling:
            h,w=grids[index][1]//2,grids[index][2]//2
            ph,pw=new_grids[index][1]//2,new_grids[index][2]//2
            feature=F.adaptive_avg_pool2d(feature.reshape(h,w,-1).permute(2,0,1).unsqueeze(0),(ph,pw))[0].permute(1,2,0).reshape(ph*pw,-1)
        return feature.to('cpu',copy=True) if cpu_output else feature
    for start in range(0, len(grids), size):
        end = min(len(grids), start + size)
        if feature_getter is not None:
            pieces.extend(keep(feature_getter(i, grids[i], pixels[i]),i) for i in range(start, end))
            continue
        current = torch.cat(pixels[start:end]).to(device=device, dtype=base.visual.dtype)
        features = base.visual(current, grid_thw=original[start:end].to(device),
                               return_dict=True).pooler_output
        pieces.extend(keep(piece,start+i) for i,piece in enumerate(features.split([math.prod(g) // 4 for g in grids[start:end]])))
        del current, features
    vision = torch.cat(pieces)
    if not pooling or cpu_output:
        return vision
    pooled, offset = [], 0
    for old, new in zip(grids, media['image_grid_thw'].tolist()):
        h, w = old[1] // 2, old[2] // 2
        ph, pw = new[1] // 2, new[2] // 2
        features = vision[offset:offset + h*w].reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
        pooled.append(F.adaptive_avg_pool2d(features, (ph, pw))[0].permute(1, 2, 0).reshape(ph*pw, -1))
        offset += h*w
    return torch.cat(pooled)


@torch.inference_mode()
def _forward_chunked_impl(model, processor, encoded, lexical_weight, *,
                    initial_chunk_tokens=8192, chunk_tokens=4096, pooling=False,
                    vision_batch_images=1, features_provider=None, prefix_entry=None,
                    checkpoint_positions=(), checkpoint_callback=None, prefix_boundary=None, prefix_entry_is_working=False,
                    on_chunk=None, active_offload='none', _offload_holder=None, _diagnostic_state=None,
                    cpu_head_prepare=None):
    if initial_chunk_tokens < 1 or chunk_tokens < 1:
        raise ValueError('Prefill chunk sizes must be positive')
    base = model.language_model.model
    device = next(model.parameters()).device
    has_images = bool(encoded.media)
    tiled=active_offload in {'kv_gpu','kv_stream'}
    if active_offload not in {'none','hidden','kv_hidden','kv_stream','kv_gpu'}:raise ValueError('Invalid active context offload mode')
    if active_offload!='none' and not tiled and (prefix_entry is not None or checkpoint_callback is not None or on_chunk is not None):
        raise ValueError('Active offloading requires isolated uncached execution')
    prepare_head=active_offload!='none' and (os.environ.get('CLEF_HEAD_CPU_PREPARE','0')=='1' if cpu_head_prepare is None else cpu_head_prepare)
    if active_offload!='none':check_host_capacity(len(encoded.input_ids),base.language_model.config,active_offload,head_cpu=prepare_head,
        cached_tokens=len(prefix_entry['ids']) if prefix_entry is not None and active_offload=='kv_stream' else 0)
    if _diagnostic_state is not None:
        _diagnostic_state.update(stage='backbone', tokens=len(encoded.input_ids))
    # Keep large pixel tensors on CPU until each vision batch needs them.
    collated_record = replace(encoded, media={k: v for k, v in encoded.media.items()
                              if k != 'pixel_values'}) if has_images else encoded
    batch = joint.collate_records([collated_record], processor.tokenizer.pad_token_id, device)
    ids, mask = batch['input_ids'], batch['attention_mask']
    features = None
    image_token = model.language_model.config.image_token_id if has_images else None
    image_counts = None
    vision_ms = 0.0
    cursor = len(prefix_entry['ids']) if prefix_entry is not None else 0
    vision_reused = False
    vision_images_reused = 0
    feature_offset = 0
    if has_images:
        media = batch['media']
        positions, _ = base.get_rope_index(ids, image_grid_thw=media['image_grid_thw'],
            attention_mask=mask, mm_token_type_ids=media['mm_token_type_ids'])
        if device.type == 'cuda':
            torch.cuda.current_stream(device).synchronize()
        vision_started = time.perf_counter()
        media_end = encoded.media['token_offset'] + len(encoded.media['mm_token_type_ids'])
        remaining, vision_images_reused = remaining_images(encoded, image_token, cursor)
        vision_reused = vision_images_reused == len(encoded.media['image_grid_thw'])
        if not vision_reused:
            features = (features_provider(base, remaining, device, pooling) if features_provider else
                        image_features(base, remaining, device, pooling=pooling, batch_images=vision_batch_images,cpu_output=tiled))
        if device.type == 'cuda':
            torch.cuda.current_stream(device).synchronize()
        vision_ms = (time.perf_counter() - vision_started) * 1000
        image_counts = [0]
        for token in encoded.input_ids:
            image_counts.append(image_counts[-1] + int(token == image_token))
        feature_offset = sum(math.prod(g) // 4 for g in encoded.media['image_grid_thw'][:vision_images_reused].tolist())
        if features is not None and image_counts[-1] - feature_offset != features.shape[0]:
            raise ValueError('Image features and placeholder count disagree')
    else:
        positions = torch.arange(ids.shape[1], device=device).view(1, 1, -1).expand(3, 1, -1)
    controller=ActiveKVOffload(base.language_model,device) if active_offload=='kv_hidden' else None
    if tiled:
        if ids.shape[0]!=1 or not bool(torch.all(mask==1)):
            raise ValueError('Streamed KV experiment requires a single unpadded request')
        from streamed_kv import StreamedKVOffload
        seed_started=time.perf_counter()
        controller=StreamedKVOffload(base.language_model,device,len(encoded.input_ids),
            storage='cuda' if active_offload=='kv_gpu' else 'cpu',prefix=prefix_entry)
        torch.cuda.current_stream(device).synchronize()
        prefix_seed_ms=(time.perf_counter()-seed_started)*1000 if prefix_entry is not None else 0.
    if controller is not None and _offload_holder is not None:_offload_holder.append(controller)
    past = controller.cache if controller is not None else None
    hidden = None
    hidden_offload_ms=hidden_restore_ms=0.
    if prefix_entry is not None:
        if not tiled:past = prefix_entry['cache'] if prefix_entry_is_working else copy.deepcopy(prefix_entry['cache'])
        cached = prefix_entry['chunks']
        hidden = torch.empty((ids.shape[0], ids.shape[1], cached[0].shape[-1]),
                             dtype=cached[0].dtype, device='cpu' if tiled else device)
        offset = 0
        for piece in cached:
            from shared_host_blocks import FrozenTensor
            if isinstance(piece,FrozenTensor):piece=piece.materialize('cpu' if tiled else device)
            hidden[:, offset:offset + piece.shape[1]].copy_(piece)
            offset += piece.shape[1]
        if offset != cursor:
            raise ValueError('Cached hidden states and prefix length disagree')
        del cached, prefix_entry, piece
    chunks = splits = 0
    points = sorted(set(checkpoint_positions))
    if device.type == 'cuda':
        torch.cuda.current_stream(device).synchronize()
    started = time.perf_counter()
    while cursor < ids.shape[1]:
        size = initial_chunk_tokens if cursor == 0 else chunk_tokens
        end = min(cursor + size, ids.shape[1])
        # The input/schema boundary is stable across changing questions. Use the
        # same partition for cold, uncached and warm calls, independent of budget.
        cuts = [p for p in (*points, prefix_boundary) if p is not None and cursor < p < end]
        if cuts:
            end = min(cuts)
        chunk_ids = ids[:, cursor:end]
        if has_images:
            embeddings = base.get_input_embeddings()(chunk_ids)
            start_feature, end_feature = image_counts[cursor], image_counts[end]
            if end_feature > start_feature:
                image_mask = (chunk_ids == image_token).unsqueeze(-1).expand_as(embeddings)
                embeddings = embeddings.masked_scatter(image_mask,
                    features[start_feature-feature_offset:end_feature-feature_offset].to(device=device,dtype=embeddings.dtype))
            inputs = {'inputs_embeds': embeddings}
            if end < len(encoded.input_ids) and encoded.input_ids[end-1] == image_token == encoded.input_ids[end]:
                splits += 1
        else:
            inputs = {'input_ids': chunk_ids}
        output = base.language_model(**inputs, attention_mask=mask[:, :end],
            position_ids=positions[:, :, cursor:end], past_key_values=past,
            use_cache=True, return_dict=True)
        past = output.past_key_values
        if hidden is None:
            hidden = torch.empty((ids.shape[0], ids.shape[1], output.last_hidden_state.shape[-1]),
                                 dtype=output.last_hidden_state.dtype, device='cpu' if active_offload!='none' else device)
        transfer_started=time.perf_counter()
        hidden[:, cursor:end].copy_(output.last_hidden_state)
        if active_offload!='none':hidden_offload_ms+=(time.perf_counter()-transfer_started)*1000
        del output, inputs
        if has_images:
            del embeddings
        cursor = end
        chunks += 1
        if checkpoint_callback is not None and end in points:
            checkpoint_callback(end, past, hidden[:, :end])
        if on_chunk is not None and cursor<ids.shape[1]:
            on_chunk(cursor,ids.shape[1])
    if device.type == 'cuda':
        torch.cuda.current_stream(device).synchronize()
    language_ms = (time.perf_counter() - started) * 1000
    offload_usage=controller.stats() if controller is not None else {}
    if controller is not None:controller.close()
    del past, features
    cpu_head = prepare_head
    if _diagnostic_state is not None:_diagnostic_state['stage']='hidden_restore'
    if active_offload!='none' and not cpu_head:
        transfer_started=time.perf_counter()
        hidden=hidden.to(device=device,copy=True)
        if device.type == 'cuda':
            torch.cuda.current_stream(device).synchronize()
        hidden_restore_ms=(time.perf_counter()-transfer_started)*1000
    started = time.perf_counter()
    if _diagnostic_state is not None:_diagnostic_state['stage']='decision_head'
    # Original schema spans and IDs, including all image placeholders, are intact.
    head_usage = {}
    if cpu_head:
        from cpu_head_prepare import head_from_cpu
        results, head_usage = head_from_cpu(model.head, hidden, ids, mask, [encoded], lexical_weight,
            chunk_tokens=int(os.environ.get('CLEF_HEAD_PREPARE_CHUNK_TOKENS','4096')))
        logits = results[0]
    else:
        logits = model.head(hidden, ids, mask, [encoded], lexical_weight)[0]
    if device.type == 'cuda':
        torch.cuda.current_stream(device).synchronize()
    usage = {'prefill_mode': 'chunked', 'prefill_chunks': chunks,
             'prefill_initial_chunk_tokens': initial_chunk_tokens,
             'prefill_chunk_tokens': chunk_tokens,
             'language_ms': round(language_ms, 1),
             'head_ms': round((time.perf_counter() - started) * 1000, 1), **head_usage}
    if active_offload!='none':usage.update(active_context_offload=active_offload,
        hidden_offload_ms=round(hidden_offload_ms,1),hidden_restore_ms=round(hidden_restore_ms,1),
        active_hidden_mib=round(hidden.numel()*hidden.element_size()/2**20,1),**offload_usage)
    if tiled:usage['prefix_seed_ms']=round(prefix_seed_ms,1)
    if has_images:
        usage.update(multimodal_prefill=True, vision_ms=round(vision_ms, 1),
                     vision_batch_images=vision_batch_images, image_chunk_splits=splits,
                     vision_prefix_reused=vision_reused, vision_images_reused=vision_images_reused)
    return logits, usage


@torch.inference_mode()
def forward_chunked(*args, **kwargs):
    holders=[]
    diagnostic={}
    try:
        return _forward_chunked_impl(*args,_offload_holder=holders,_diagnostic_state=diagnostic,**kwargs)
    except torch.cuda.OutOfMemoryError:
        if os.environ.get('CLEF_OFFLOAD_TRACE','0')=='1':
            logging.getLogger(__name__).exception('Active context GPU OOM: %s',diagnostic)
        raise
    finally:
        for controller in holders:controller.close()
