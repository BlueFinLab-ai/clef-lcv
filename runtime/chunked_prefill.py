"""Incremental mixed text/image prefill with request-local hybrid state.

Images are fully vision-encoded in bounded batches before language prefill. Global
multimodal positions are calculated once, then sliced alongside chunk embeddings.
All token outputs are retained for the unchanged native head. Optional prefix
checkpoints and independent image-feature reuse are supplied by the memory-budgeted
inference engine.
"""
from dataclasses import replace
import copy
import math
import time

import torch
import torch.nn.functional as F
import joint_schema_model as joint


@torch.inference_mode()
def image_features(base, encoded, device, *, pooling=False, batch_images=1, feature_getter=None):
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
    for start in range(0, len(grids), size):
        end = min(len(grids), start + size)
        if feature_getter is not None:
            pieces.extend(feature_getter(i, grids[i], pixels[i]) for i in range(start, end))
            continue
        current = torch.cat(pixels[start:end]).to(device=device, dtype=base.visual.dtype)
        features = base.visual(current, grid_thw=original[start:end].to(device),
                               return_dict=True).pooler_output
        pieces.extend(features.split([math.prod(g) // 4 for g in grids[start:end]]))
        del current, features
    vision = torch.cat(pieces)
    if not pooling:
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
def forward_chunked(model, processor, encoded, lexical_weight, *,
                    initial_chunk_tokens=8192, chunk_tokens=4096, pooling=False,
                    vision_batch_images=1, features_provider=None, prefix_entry=None,
                    checkpoint_positions=(), checkpoint_callback=None, prefix_boundary=None):
    if initial_chunk_tokens < 1 or chunk_tokens < 1:
        raise ValueError('Prefill chunk sizes must be positive')
    base = model.language_model.model
    device = next(model.parameters()).device
    has_images = bool(encoded.media)
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
    if has_images:
        media = batch['media']
        positions, _ = base.get_rope_index(ids, image_grid_thw=media['image_grid_thw'],
            attention_mask=mask, mm_token_type_ids=media['mm_token_type_ids'])
        torch.cuda.synchronize()
        vision_started = time.perf_counter()
        media_end = encoded.media['token_offset'] + len(encoded.media['mm_token_type_ids'])
        vision_reused = cursor >= media_end
        if not vision_reused:
            features = (features_provider(base, encoded, device, pooling) if features_provider else
                        image_features(base, encoded, device, pooling=pooling, batch_images=vision_batch_images))
        torch.cuda.synchronize()
        vision_ms = (time.perf_counter() - vision_started) * 1000
        image_counts = [0]
        for token in encoded.input_ids:
            image_counts.append(image_counts[-1] + int(token == image_token))
        if features is not None and image_counts[-1] != features.shape[0]:
            raise ValueError('Image features and placeholder count disagree')
    else:
        positions = torch.arange(ids.shape[1], device=device).view(1, 1, -1).expand(3, 1, -1)
    past = hidden = None
    if prefix_entry is not None:
        past = copy.deepcopy(prefix_entry['cache'])
        cached = prefix_entry['chunks']
        hidden = torch.empty((ids.shape[0], ids.shape[1], cached[0].shape[-1]),
                             dtype=cached[0].dtype, device=device)
        offset = 0
        for piece in cached:
            hidden[:, offset:offset + piece.shape[1]].copy_(piece)
            offset += piece.shape[1]
        if offset != cursor:
            raise ValueError('Cached hidden states and prefix length disagree')
        del cached, prefix_entry
    chunks = splits = 0
    points = sorted(set(checkpoint_positions))
    torch.cuda.synchronize()
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
                    features[start_feature:end_feature].to(embeddings.dtype))
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
                                 dtype=output.last_hidden_state.dtype, device=device)
        hidden[:, cursor:end].copy_(output.last_hidden_state)
        del output, inputs
        if has_images:
            del embeddings
        cursor = end
        chunks += 1
        if checkpoint_callback is not None and end in points:
            checkpoint_callback(end, past, hidden[:, :end])
    torch.cuda.synchronize()
    language_ms = (time.perf_counter() - started) * 1000
    del past, features
    started = time.perf_counter()
    # Original schema spans and IDs, including all image placeholders, are intact.
    logits = model.head(hidden, ids, mask, [encoded], lexical_weight)[0]
    torch.cuda.synchronize()
    usage = {'prefill_mode': 'chunked', 'prefill_chunks': chunks,
             'prefill_initial_chunk_tokens': initial_chunk_tokens,
             'prefill_chunk_tokens': chunk_tokens,
             'language_ms': round(language_ms, 1),
             'head_ms': round((time.perf_counter() - started) * 1000, 1)}
    if has_images:
        usage.update(multimodal_prefill=True, vision_ms=round(vision_ms, 1),
                     vision_batch_images=vision_batch_images, image_chunk_splits=splits,
                     vision_prefix_reused=vision_reused)
    return logits, usage
