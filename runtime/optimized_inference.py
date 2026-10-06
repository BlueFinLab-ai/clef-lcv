"""Single-pass encoding, bounded exact-prefix reuse, and optional spatial pooling.

The official joint decision head remains unchanged. Cache entries live only in
this process and are cloned before use because hybrid attention states mutate.
"""
import copy
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import math
import time

import torch
import torch.nn.functional as F
import joint_schema_model as joint
from host_prefix_cache import HostPrefixCache, copy_to_device
from radix_index import PrefixRadixIndex, IndexedBank
from async_restore import AsyncRestoreRuntime
from chunked_prefill import forward_chunked, image_features


def encode_compact(processor, record, *, with_checkpoints=False, input_cache=None, cache_enabled=True):
    tokenizer = processor.tokenizer
    tokens = (lambda text: input_cache.tokens(tokenizer, text, cache_enabled)) if input_cache else (lambda text: joint._tokens(tokenizer, text))
    ids = tokens(f'<|im_start|>system\n{joint.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n')
    checkpoints = []
    if record.get('context') is not None:
        ids += tokens('CONTEXT:\n' + joint.render(record['context']) + '\nINPUT:\n')
        checkpoints.append(len(ids))
    media_ids, media = input_cache.media(processor, record, cache_enabled) if input_cache else joint._encode_media(processor, record)
    if media is not None:
        media['token_offset'] = len(ids)
        ids += media_ids
        checkpoints.append(len(ids))
    ids += tokens(joint.render(record['state']))
    boundary = len(ids)
    ids += tokens('\nSCHEMA:\n')
    questions = []
    for qid, question in record['questions'].items():
        ids += tokens(f'{qid} ({question["type"]}): ')
        start = len(ids)
        ids += tokens(question.get('instructions') or qid)
        end = len(ids)
        spans, options = [], []
        for oid, description in joint.question_options(question):
            ids += tokens('\n')
            option_start = len(ids)
            ids += tokens(joint.render({'option_id': oid, 'description': description}))
            spans.append((option_start, len(ids)))
            options.append(oid)
        ids += tokens('\n')
        questions.append(joint.EncodedQuestion(qid, joint.QUESTION_TYPES[question['type']], (start, end), tuple(spans), tuple(options)))
    ids += tokens('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:')
    # Reject overlength requests; never silently truncate the user's context.
    result = joint.EncodedRecord(tuple(ids), tuple(questions), 'request', media)
    return (result, boundary, tuple(checkpoints)) if with_checkpoints else (result, boundary)


def pool_record(encoded, boundary, image_token_id):
    """Compress each image's merged spatial grid independently, preserving order."""
    if not encoded.media:
        return encoded, boundary
    media = encoded.media.copy()
    original = media['image_grid_thw'].clone()
    grid = original.clone()
    keep = [True] * len(encoded.input_ids)
    cursor = 0
    for index, (temporal, height, width) in enumerate(original.tolist()):
        if temporal != 1:
            raise ValueError('Pooling supports still images only')
        old_count = height * width // 4
        ph, pw = math.ceil(height / 4), math.ceil(width / 4)
        grid[index, 1], grid[index, 2] = ph * 2, pw * 2
        while cursor < len(keep) and encoded.input_ids[cursor] != image_token_id:
            cursor += 1
        if encoded.input_ids[cursor:cursor + old_count] != (image_token_id,) * old_count:
            raise ValueError('Image grid and placeholder count disagree')
        for position in range(cursor + ph * pw, cursor + old_count):
            keep[position] = False
        cursor += old_count
    offsets = [0]
    for retained in keep:
        offsets.append(offsets[-1] + retained)
    questions = tuple(replace(q, question_span=tuple(offsets[p] for p in q.question_span),
        option_spans=tuple(tuple(offsets[p] for p in span) for span in q.option_spans)) for q in encoded.questions)
    offset = media['token_offset']
    media['mm_token_type_ids'] = [v for i, v in enumerate(media['mm_token_type_ids']) if keep[offset + i]]
    media.update(image_grid_thw=grid, original_grid_thw=original)
    return replace(encoded, input_ids=tuple(v for i, v in enumerate(encoded.input_ids) if keep[i]), questions=questions, media=media), offsets[boundary]


def remap_checkpoints(checkpoints, original, pooled):
    """Encoder checkpoints are before all media or after all media, never inside."""
    removed = len(original.input_ids) - len(pooled.input_ids)
    start = original.media['token_offset'] if original.media else len(original.input_ids)
    return tuple(p if p <= start else p - removed for p in checkpoints)


def tensor_bytes(value, seen=None):
    """Account for cache tensor storages without counting shared views twice."""
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        key = ('storage', storage.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(tensor_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v, seen) for v in value)
    if hasattr(value, '__dict__'):
        return tensor_bytes(vars(value), seen)
    return 0


def available_cuda_memory():
    # mem_get_info excludes blocks reserved by PyTorch but currently unused.
    # Those blocks can satisfy cache clones without new device allocations.
    free = torch.cuda.mem_get_info()[0]
    return free + max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())


class LexicalRows:
    """Keep compact NF4 embeddings on their row-lookup path."""
    def __init__(self, embedding):
        self.embedding = embedding

    def __getitem__(self, token_ids):
        return self.embedding(token_ids)


def lexical_weight(model):
    embedding = model.language_model.get_output_embeddings()
    return embedding.weight if embedding.weight.is_floating_point() else LexicalRows(embedding)


class InferenceEngine:
    """Exact prefix checkpoints, with independent mutable hybrid state per branch.

    Checkpoints are indexed by their complete ancestry, not just the last chunk.
    Hidden chunks share immutable storage; hybrid cache snapshots are independent.
    This is checkpoint reuse, not a paged-attention implementation.
    """
    def __init__(self, max_cache_mib='auto', max_entries=32, reserve_mib=1024,
                 utilization=.90, checkpoint_tokens=0, min_prefix_tokens=128, feature_cache_mib=256, feature_cache_entries=64, elastic=False,
                 prefill_reserve_mib=512, promote_repeated_boundary=False, host_cache_mib=0, host_cache_entries=128,
                 host_cache_reserve_fraction=.25, ram_budget=None, prefix_index="radix", async_restore=False, restore_staging_mib=128,
                 shared_host_blocks=False, active_context_offload='none'):
        if active_context_offload not in {'none','auto','hidden','kv_hidden','kv_stream','kv_gpu'}:raise ValueError('Invalid active context offload mode')
        self.active_context_offload=active_context_offload
        self.host = HostPrefixCache(host_cache_mib, host_cache_entries, host_cache_reserve_fraction, budget=ram_budget, shared_blocks=shared_host_blocks)
        if prefix_index not in {'radix', 'scan'}:
            raise ValueError('Prefix index must be radix or scan')
        self.prefix_index = PrefixRadixIndex() if prefix_index == 'radix' else None
        if self.prefix_index is not None:
            self.host.entries = IndexedBank(self.prefix_index, 'cpu')
        self.restore_ms = 0.0
        self.restore_rejections = 0
        self.async_restore = AsyncRestoreRuntime(restore_staging_mib) if async_restore and not torch.version.hip else None
        self.entries = IndexedBank(self.prefix_index, 'gpu') if self.prefix_index is not None else OrderedDict()
        self.observations = IndexedBank(self.prefix_index, 'seen') if self.prefix_index is not None else OrderedDict()
        self.features = OrderedDict()
        self.feature_limit = max(0, int(feature_cache_mib)) * 2**20
        self.feature_entries = max(0, int(feature_cache_entries))
        self.feature_hits = self.feature_misses = self.feature_evictions = 0
        self.elastic = bool(elastic)
        self.retention_rejections = {}
        self.last_retention_rejection = None
        self.last_rejected_checkpoint_bytes = 0
        self.auto_budget = str(max_cache_mib).lower() == 'auto'
        self.max_bytes = 0 if self.auto_budget else max(0, int(max_cache_mib)) * 2**20
        self.max_entries = max(0, max_entries)
        self.reserve_bytes = max(0, reserve_mib) * 2**20
        self.prefill_reserve_bytes = max(self.reserve_bytes, max(0, prefill_reserve_mib) * 2**20)
        if not 0 < utilization <= 1:
            raise ValueError('Cache GPU utilization must be in (0, 1]')
        self.utilization = utilization
        self.checkpoint_tokens = max(0, checkpoint_tokens)
        self.min_prefix_tokens = max(1, min_prefix_tokens)
        self.promote_repeated_boundary = bool(promote_repeated_boundary)
        self.base_bytes = None
        self.workspace_bytes = 0
        self.chunked_workspace_bytes = 0
        self.request_workspace_bytes = 0 if self.elastic else None
        self.hits = self.misses = self.evictions = self.reused_tokens = 0
        self.progress_hook = None
        self.context_policy = None

    def _bytes(self):
        # Token ancestry, media keys and byte-count metadata contain no tensors.
        # Walking them on every budget check made long retained prompts a CPU
        # bottleneck. Use one traversal of tensor-bearing fields so immutable
        # hidden chunks shared by checkpoints/features still count only once.
        snapshots = [(entry['cache'], entry['chunks']) for entry in self.entries.values()]
        return tensor_bytes([snapshots, list(self.features.values())])

    def initialize_memory_budget(self):
        """Call after model loading, before request tensors are allocated."""
        self.base_bytes = torch.cuda.memory_allocated() - self._bytes()
        self._budget()

    def _drop(self, protected=None):
        # Prefer reclaiming large language snapshots over cheap reusable image
        # features. A protected hit survives unless the caller rejects its clone.
        for key in list(self.entries):
            if key != protected:
                self.host.put(key, self.entries[key], tensor_bytes)
                del self.entries[key]
                self.evictions += 1
                return True
        if self.features:
            self.features.popitem(last=False)
            self.feature_evictions += 1
            return True
        return False

    def _workspace(self):
        return self.workspace_bytes if self.request_workspace_bytes is None else self.request_workspace_bytes

    def _reserve(self):
        # Explicit projected workspace is used for incremental prefill. Idle
        # and short calls keep the smaller margin, so an 8K hit can survive.
        return self.prefill_reserve_bytes if self.elastic and self.request_workspace_bytes else self.reserve_bytes

    def prepare_request(self, workspace_bytes=None, *, observed_chunked=False, protected=None):
        """Evict by projected request workspace, without a token-count cutoff."""
        started = time.perf_counter()
        host_offload_before = self.host.offload_ms
        self.host.trim()
        before = self._bytes()
        if observed_chunked:
            # Fixed-limit deployments lack a calibrated token-based projection.
            # Use their measured chunked peak rather than the much smaller peak
            # of a short, unchunked call. Explicit projections can only grow.
            workspace_bytes = max(workspace_bytes or 0, self.chunked_workspace_bytes) or None
        self.request_workspace_bytes = workspace_bytes
        self._trim(protected=protected)
        released = max(0, before - self._bytes())
        release_ms = 0.0
        if self.elastic and workspace_bytes and released:
            # Return unused allocator segments after eviction before starting
            # prefill; this leaves cached live tensors intact and avoids doing
            # failed GPU work merely to discover releasable memory.
            release_started = time.perf_counter()
            torch.cuda.empty_cache()
            release_ms = (time.perf_counter() - release_started) * 1000
        return {'cache_prepare_ms': round((time.perf_counter()-started)*1000, 1),
                'cache_offload_ms': round(self.host.offload_ms-host_offload_before, 1),
                'cache_evicted_mib': round(released/2**20, 1),
                'cache_headroom_mib': self._reserve()/2**20,
                'cache_allocator_release_ms': round(release_ms, 1)}

    def finish_request(self):
        """Release active workspace reservation; cached tensors stay process-local."""
        if self.async_restore is not None:
            self.async_restore.finish()
        if self.elastic:
            self.request_workspace_bytes = 0
            self._trim()
        self.host.trim()

    def _required_free(self, allocation=0, *, request_clone=False):
        if not self.elastic:
            return max(self.reserve_bytes, self._workspace()) + allocation
        # Working tensors already occupy VRAM. Reserve only the remaining peak,
        # plus fixed headroom. A branch clone is part of request workspace;
        # a retained snapshot/feature clone is an extra persistent allocation.
        live = max(0, torch.cuda.memory_allocated() - (self.base_bytes or 0) - self._bytes())
        remaining = max(0, self._workspace() - live)
        remaining = max(remaining, allocation) if request_clone else remaining + allocation
        return self._reserve() + remaining

    def _reject_retention(self, reason, size):
        self.last_retention_rejection = reason
        self.last_rejected_checkpoint_bytes = size
        self.retention_rejections[reason] = self.retention_rejections.get(reason, 0) + 1
        return False

    def _budget(self):
        if self.auto_budget:
            free, total = torch.cuda.mem_get_info()
            other = max(0, total - free - torch.cuda.memory_reserved())
            self.max_bytes = max(0, int(total * self.utilization) - other
                                 - (self.base_bytes or 0)
                                 - (self._reserve() + self._workspace() if self.elastic
                                    else max(self.reserve_bytes, self._workspace())))
        return self.max_bytes

    def _trim(self, protected=None, clone_bytes=0, request_clone=False):
        self._budget()
        needed = self._required_free(clone_bytes, request_clone=request_clone)
        while (self.entries or self.features) and (self._bytes() > self.max_bytes
                or len(self.entries) > self.max_entries
                or available_cuda_memory() < needed):
            if not self._drop(protected):
                break

    @staticmethod
    def _namespace(model, input_key, pooling, position, media_start):
        # Prefixes before the first image are independent of all image bytes.
        return (id(model), input_key if media_start is not None and position > media_start else 'text',
                bool(pooling) if media_start is not None and position > media_start else False)

    def _match(self, model, tokens, boundary, input_key, pooling, media_start, media_end):
        if self.prefix_index is None:
            return self._match_scan(model, tokens, boundary, input_key, pooling, media_start, media_end)
        return self.prefix_index.match(id(model), tokens, boundary, input_key,
                                      pooling, media_start, media_end)

    def _match_scan(self, model, tokens, boundary, input_key, pooling, media_start, media_end):
        best_key = None
        common = 0
        for key, entry in [*list(self.observations.items()), *list(self.host.entries.items()), *list(self.entries.items())]:
            if key[0] != id(model):
                continue
            n = min(boundary, len(entry['ids']))
            matched = 0
            for left, right in zip(tokens[:n], entry['ids'][:n]):
                if left != right:
                    break
                matched += 1
            if media_start is not None and entry['media_key'] != (input_key, bool(pooling)):
                matched = min(matched, media_start)
            if media_start is not None and media_start < matched < media_end:
                matched = media_start
            common = max(common, matched)
            length = len(entry['ids'])
            if 'cache' in entry and length <= matched and key[:3] == self._namespace(model, input_key, pooling, length, media_start):
                if best_key is None or length > len(self._lookup(best_key)['ids']):
                    best_key = key
        return best_key, common

    def _lookup(self, key):
        return self.entries.get(key) or self.host.entries.get(key)

    def _acquire(self, key, device):
        """Return an immutable GPU hit or a request-owned restored host snapshot.

        Restore under the existing single-GPU worker, not at HTTP admission.
        Host hits are working copies: callers must not clone them a second time.
        """
        if key is None:
            return None, 'miss'
        self.host.trim()
        if key in self.entries:
            entry = self.entries[key]
            self._trim(protected=key, clone_bytes=tensor_bytes(entry['cache']), request_clone=True)
            if available_cuda_memory() < self._required_free(tensor_bytes(entry['cache']), request_clone=True):
                self.restore_rejections += 1
                self._acquire_cpu(key,count_hit=False)
                del entry
                if key in self.host.entries:return self._acquire(key,device)
                return None, 'recompute'
            self.entries.move_to_end(key)
            entry['uses'] = entry.get('uses', 0) + 1
            entry['last_use'] = time.monotonic_ns()
            self.host.touch(key)
            return entry, 'gpu'
        source = self.host.entries.get(key)
        if source is None:
            return None, 'miss'
        self._trim(clone_bytes=source['bytes'], request_clone=True)
        if available_cuda_memory() < self._required_free(source['bytes'], request_clone=True):
            self.restore_rejections += 1
            return None, 'recompute'
        started = time.perf_counter()
        if self.async_restore is not None:
            cache, chunks = self.async_restore.start(source['cache'], source['chunks'], device)
        else:
            cache, chunks = copy_to_device((source['cache'], source['chunks']), device)
        entry = {**source, 'cache': cache, 'chunks': chunks}
        self.restore_ms += (time.perf_counter() - started) * 1000
        self.host.hits += 1
        # Budget trimming can evict the source from RAM while spilling GPU entries;
        # the local reference is still valid, but it may no longer be indexed.
        self.host.touch(key)
        return entry, 'cpu'

    def _acquire_cpu(self,key,*,count_hit=True):
        """Borrow immutable CPU state; never restore full KV into CUDA here."""
        if key is None:return None,'miss'
        source=self.entries.get(key)
        if source is not None:
            if self.host.put(key,source,tensor_bytes):
                entry=self.host.entries[key]
            else:
                cache,chunks=copy_to_device((source['cache'],source['chunks']),'cpu')
                entry={**source,'cache':cache,'chunks':chunks}
            del self.entries[key]
            self.evictions+=1
        else:entry=self.host.entries.get(key)
        if entry is None:return None,'miss'
        if count_hit:self.host.hits+=1
        self.host.touch(key,uses=1 if count_hit else 0)
        return entry,'cpu'

    def spill_gpu_cache(self):
        """Reclaim GPU snapshots through the RAM tier, preserving metadata."""
        while self.entries or self.features:self._drop()
        torch.cuda.empty_cache()

    def reclaim_host_workspace(self,tokens,mode,model,input_key,pooling,boundary,encoded):
        from host_memory import available_memory
        media_start=encoded.media['token_offset'] if encoded.media else None
        media_end=media_start+len(encoded.media['mm_token_type_ids']) if encoded.media else 0
        key,_=self._match(model,encoded.input_ids,boundary,input_key,pooling,media_start,media_end)
        prefix=self.host.entries.get(key)
        cached=len(prefix['ids']) if prefix is not None and mode=='kv_stream' else 0
        del prefix
        c=model.language_model.config.text_config
        hidden=2*tokens*c.hidden_size*2
        kv=max(0,tokens-cached)*c.layer_types.count('full_attention')*2*c.num_key_value_heads*c.head_dim*2 if mode=='kv_stream' else 0
        needed=hidden+kv+2*kv//max(1,c.layer_types.count('full_attention'))
        released=0
        while True:
            sample=available_memory()
            if sample and sample['available_bytes']//2>=needed:return released
            before=self.host.bytes
            if not self.host.evict_one(protected=key):return released
            released+=max(0,before-self.host.bytes)

    def _input_key(self, model, tokens, end, input_key, pooling, media_start):
        return (*self._namespace(model, input_key, pooling, end, media_start),
                hashlib.sha256(json.dumps(tokens[:end]).encode()).hexdigest())

    def _remember(self, key, tokens, boundary, input_key, pooling):
        # CPU token metadata identifies repeated branches without spending VRAM
        # or introducing a separate language forward for every unique suffix.
        self.observations[key] = {'ids': tuple(tokens[:boundary]), 'media_key': (input_key, bool(pooling))}
        self.observations.move_to_end(key)
        while len(self.observations) > self.max_entries:
            self.observations.popitem(last=False)

    def _retain(self, model, tokens, end, input_key, pooling, media_start, past, chunks, copy_chunks=False, parent_key=None):
        key = self._input_key(model, tokens, end, input_key, pooling, media_start)
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        # Snapshot copies KV and both linear-state types; never share mutable state.
        size = tensor_bytes(past) + (sum(c.numel()*c.element_size() for c in chunks) if copy_chunks else tensor_bytes(chunks))
        if self.max_entries == 0:
            return self._reject_retention('entry_limit_disabled', size)
        if size > self._budget():
            if self.host.put(key, {'cache': past, 'chunks': tuple(chunks), 'ids': tuple(tokens[:end]), 'media_key': (input_key, bool(pooling)), 'parent_key':parent_key,'media_start':media_start}, tensor_bytes):
                return True
            return self._reject_retention('checkpoint_exceeds_memory_budget', size)
        while (self.entries or self.features) and (len(self.entries) >= self.max_entries
                or self._bytes() + size > self.max_bytes
                or available_cuda_memory() < self._required_free(size)):
            self._drop()
        if available_cuda_memory() < self._required_free(size):
            if self.host.put(key, {'cache': past, 'chunks': tuple(chunks), 'ids': tuple(tokens[:end]), 'media_key': (input_key, bool(pooling)), 'parent_key':parent_key,'media_start':media_start}, tensor_bytes):
                return True
            return self._reject_retention('insufficient_snapshot_headroom', size)
        snapshot = {'cache': copy.deepcopy(past), 'chunks': tuple(c.clone() for c in chunks) if copy_chunks else tuple(chunks), 'ids': tuple(tokens[:end]),
                    'media_key': (input_key, bool(pooling)), 'uses': 0,'parent_key':parent_key,'media_start':media_start}
        snapshot['bytes'] = tensor_bytes((snapshot['cache'], snapshot['chunks']))
        self.entries[key] = snapshot
        return True

    def clear(self):
        if self.async_restore is not None:
            self.async_restore.finish(suppress=True)
        self.entries.clear()
        self.host.clear()
        self.observations.clear()
        self.features.clear()

    def stats(self):
        return {'entries': len(self.entries), 'mib': round(self._bytes() / 2**20, 1),
                'limit_mib': round(self.max_bytes / 2**20, 1), 'budget_mode': 'auto' if self.auto_budget else 'fixed',
                'elastic': self.elastic, 'phase': 'idle' if self.elastic and self.request_workspace_bytes == 0 else 'request',
                'request_workspace_mib': round(self._workspace()/2**20, 1),
                'retention_rejections': dict(self.retention_rejections),
                'last_retention_rejection': self.last_retention_rejection,
                'last_rejected_checkpoint_mib': round(self.last_rejected_checkpoint_bytes/2**20, 1),
                'max_entries': self.max_entries, 'reserve_mib': self.reserve_bytes / 2**20,
                'prefill_reserve_mib': self.prefill_reserve_bytes / 2**20,
                'effective_reserve_mib': self._reserve() / 2**20,
                'observed_prefixes': len(self.observations), 'index': self.prefix_index.stats() if self.prefix_index is not None else {'kind': 'scan'},
                'observed_workspace_mib': round(self.workspace_bytes / 2**20, 1),
                'observed_chunked_workspace_mib': round(self.chunked_workspace_bytes / 2**20, 1),
                'gpu_utilization_target': self.utilization, 'checkpoint_tokens': self.checkpoint_tokens,
                'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
                'reused_tokens': self.reused_tokens,
                'host': {**self.host.stats(), 'restore_ms': round(self.restore_ms, 1), 'restore_rejections': self.restore_rejections,
                         'async_restore': self.async_restore.stats() if self.async_restore is not None else {'enabled':False}},
                'image_features': {'entries': len(self.features), 'mib': round(tensor_bytes(list(self.features.values())) / 2**20, 2),
                    'limit_mib': self.feature_limit / 2**20, 'max_entries': self.feature_entries,
                    'hits': self.feature_hits, 'misses': self.feature_misses, 'evictions': self.feature_evictions}}

    def _vision(self, base, encoded, media, grid, enabled):
        """Reuse unpooled image features; vision positions are image-local.

        Misses run together to preserve GPU throughput. Cache keys include exact
        source identity, processor settings, grid, model instance and dtype/device.
        The ordered concatenation is always rebuilt for this request.
        """
        image_keys = encoded.media.get('image_cache_keys', ())
        if not enabled or not image_keys or not self.feature_limit or not self.feature_entries:
            return base.visual(media['pixel_values'].to(base.visual.dtype), grid_thw=grid, return_dict=True).pooler_output, 0, len(grid)
        grids = grid.tolist()
        pixels = torch.split(media['pixel_values'], [math.prod(g) for g in grids])
        keys = [(id(base.visual), str(grid.device), str(base.visual.dtype), k, tuple(g)) for k,g in zip(image_keys, grids)]
        if len(keys) != len(grids):
            raise ValueError('Image cache metadata and grids disagree')
        found, missing = {}, {}
        hits = 0
        for i,key in enumerate(keys):
            if key in self.features:
                found[key] = self.features[key]
                self.features.move_to_end(key)
                hits += 1
            elif key not in missing:
                missing[key] = i
        misses = len(missing)
        self.feature_hits += hits
        self.feature_misses += misses
        if missing:
            indices = list(missing.values())
            computed = base.visual(torch.cat([pixels[i] for i in indices]).to(base.visual.dtype),
                grid_thw=grid[indices], return_dict=True).pooler_output
            chunks = computed.split([math.prod(grids[i]) // 4 for i in indices])
            for key,chunk in zip(missing, chunks):
                found[key] = chunk
                size = tensor_bytes(chunk)
                # clone isolates each image from the complete miss-batch storage
                size = chunk.numel() * chunk.element_size()
                while self.features and (len(self.features) >= self.feature_entries
                        or tensor_bytes(list(self.features.values())) + size > self.feature_limit):
                    self.features.popitem(last=False)
                    self.feature_evictions += 1
                self._trim(clone_bytes=size)
                if size <= self.feature_limit and self._bytes() + size <= self._budget() and available_cuda_memory() >= self._required_free(size):
                    self.features[key] = chunk.clone()
        return torch.cat([found[key] for key in keys]), hits, misses

    @torch.inference_mode()
    def forward(self, model, processor, encoded, boundary, input_key, pooling=False, cache_enabled=True,
                checkpoint_boundaries=(), feature_cache_enabled=True,
                prefill_chunk_tokens=0, prefill_initial_chunk_tokens=8192, prefill_images=False,
                execution_plan=None):
        self.last_retention_rejection = None
        mode=execution_plan['mode'] if execution_plan is not None else self.active_context_offload
        if mode=='auto':mode='none'
        tiled=mode in {'kv_gpu','kv_stream'}
        if mode!='none':
            if not prefill_chunk_tokens:raise ValueError('Active offloading requires incremental prefill')
            if torch.version.hip:raise ValueError('Active offloading has not been validated on ROCm')
            if not tiled:cache_enabled=feature_cache_enabled=False
        if self.async_restore is not None and mode=='none':
            self.async_restore.install(model.language_model.model.language_model)
        self.last_rejected_checkpoint_bytes = 0
        offload_started = self.host.offload_ms
        if prefill_chunk_tokens and (encoded.media is None or prefill_images) and (len(encoded.input_ids) > prefill_initial_chunk_tokens or mode!='none'):
            if self.base_bytes is None:
                self.initialize_memory_budget()
            self._trim()
            enabled = cache_enabled and self.max_entries > 0 and (self._budget() > 0 or self.host.enabled)
            media_start = encoded.media['token_offset'] if encoded.media else None
            media_end = media_start + len(encoded.media['mm_token_type_ids']) if encoded.media else 0
            key, _ = self._match(model, encoded.input_ids, boundary, input_key, pooling,
                                 media_start, media_end) if enabled else (None, 0)
            restore_started = self.restore_ms
            entry, cache_tier = self._acquire_cpu(key) if tiled else self._acquire(key, next(model.parameters()).device)
            if entry is None:
                key = None
            reused = len(entry['ids']) if entry is not None else 0
            if enabled:
                if key is None: self.misses += 1
                else:
                    self.hits += 1
                    self.reused_tokens += reused
            # Stable partitions preserve cold/warm numerical behavior. Never
            # checkpoint a partly consumed image group, and avoid tiny snapshots.
            points = sorted(set(p for p in checkpoint_boundaries if p >= self.min_prefix_tokens))
            whole_key = self._input_key(model, encoded.input_ids, boundary, input_key, pooling, media_start)
            # A repeated exact state can justify a second snapshot even when
            # its tail after context/media is short. The boundary is already
            # a stable forward cut, so this changes retention, not partitioning.
            promote = self.promote_repeated_boundary and whole_key in self.observations
            if not points or boundary-points[-1] >= self.min_prefix_tokens or promote:
                points.append(boundary)
            saved = feature_hits = feature_misses = 0
            retained_parent=key
            def retain(end, past, hidden):
                nonlocal saved,retained_parent
                if enabled:
                    if tiled:
                        checkpoint_key=self._input_key(model,encoded.input_ids,end,input_key,pooling,media_start)
                        retained=self.host.put(checkpoint_key,{'cache':past,'chunks':(hidden,),
                            'ids':tuple(encoded.input_ids[:end]),'media_key':(input_key,bool(pooling)),
                            'parent_key':retained_parent,'media_start':media_start},tensor_bytes,
                            adopt_cpu=mode=='kv_stream' and end==boundary)
                        saved+=retained
                        if retained:retained_parent=checkpoint_key
                    else:
                        saved += self._retain(model, encoded.input_ids, end, input_key,
                            pooling, media_start, past, (hidden,), copy_chunks=True, parent_key=key)
            def vision(base, record, device, pooled):
                def one(i, grid, pixels):
                    nonlocal feature_hits, feature_misses
                    image_keys = record.media.get('image_cache_keys', ())
                    feature_key = (id(base.visual), str(device), str(base.visual.dtype),
                        image_keys[i], tuple(grid), 'single_image') if image_keys else None
                    caching_features = feature_cache_enabled and feature_key is not None and self.feature_limit and self.feature_entries
                    if caching_features and feature_key in self.features:
                        self.features.move_to_end(feature_key)
                        feature_hits += 1; self.feature_hits += 1
                        return self.features[feature_key]
                    feature_misses += 1; self.feature_misses += 1
                    result = base.visual(pixels.to(device=device,dtype=base.visual.dtype),
                        grid_thw=torch.tensor([grid],device=device),return_dict=True).pooler_output
                    size = result.numel()*result.element_size()
                    if caching_features:
                        while self.features and (len(self.features)>=self.feature_entries or
                                tensor_bytes(list(self.features.values()))+size>self.feature_limit):
                            self.features.popitem(last=False); self.feature_evictions += 1
                        self._trim(clone_bytes=size)
                        if size<=self.feature_limit and self._bytes()+size<=self._budget() and available_cuda_memory()>=self._required_free(size):
                            self.features[feature_key]=result.clone()
                    return result
                return image_features(base, record, device, pooling=pooled, feature_getter=one, cpu_output=tiled)
            retained_start = self._bytes()
            torch.cuda.reset_peak_memory_stats()
            logits, usage = forward_chunked(model, processor, encoded, lexical_weight(model),
                initial_chunk_tokens=prefill_initial_chunk_tokens, chunk_tokens=prefill_chunk_tokens,
                pooling=pooling, features_provider=vision,
                prefix_entry=entry, prefix_entry_is_working=cache_tier == 'cpu',
                checkpoint_positions=points, checkpoint_callback=retain if enabled else None, prefix_boundary=boundary,
                on_chunk=self.progress_hook if mode=='none' else None,
                active_offload=mode, cpu_head_prepare=execution_plan['head_cpu'] if execution_plan else None)
            self.restore_ms+=usage.get('prefix_seed_ms',0.)
            self.chunked_workspace_bytes = max(self.chunked_workspace_bytes,
                torch.cuda.max_memory_allocated() - (self.base_bytes or 0) - min(retained_start,self._bytes()))
            whole_key = self._input_key(model,encoded.input_ids,boundary,input_key,pooling,media_start)
            self._remember(whole_key,encoded.input_ids,boundary,input_key,pooling)
            return logits, {**usage, 'prefix_cache_tier': cache_tier, 'prefix_restore_ms': round(self.restore_ms-restore_started, 1), 'prefix_offload_ms': round(self.host.offload_ms-offload_started, 1), 'prefix_cache': 'hit' if key is not None else ('miss' if saved else 'not_retained') if enabled else 'disabled',
                'prefix_tokens': boundary, 'reused_prefix_tokens': reused,
                'cache_memory_budget_mib': round(self._budget()/2**20,1),
                'cache_limit_reason': self.last_retention_rejection if enabled and key is None and not saved else None if enabled else ('disabled_by_request' if not cache_enabled else 'request_memory_budget'),
                'new_prefix_tokens': boundary-reused, 'prefix_build_ms': usage['language_ms'] if saved else 0.,
                'prefix_checkpoints_saved': saved, 'image_feature_cache_hits': feature_hits,
                'image_feature_cache_misses': feature_misses}

        base = model.language_model.model
        ids_device = next(model.parameters()).device
        batch = joint.collate_records([encoded], processor.tokenizer.pad_token_id, ids_device)
        ids, mask, media = batch['input_ids'], batch['attention_mask'], batch['media']
        if media:
            positions, _ = base.get_rope_index(ids, image_grid_thw=media['image_grid_thw'],
                attention_mask=mask, mm_token_type_ids=media['mm_token_type_ids'])
        else:
            positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, 1, -1)
        if self.base_bytes is None:
            self.initialize_memory_budget()
        self._trim()
        retained_start = self._bytes()
        torch.cuda.reset_peak_memory_stats()
        enabled = cache_enabled and self.max_entries > 0 and (self._budget() > 0 or self.host.enabled)
        media_start = encoded.media['token_offset'] if encoded.media else None
        media_end = media_start + len(encoded.media['mm_token_type_ids']) if encoded.media else 0
        whole_key = self._input_key(model, encoded.input_ids, boundary, input_key, pooling, media_start)
        seen_input = whole_key in self.observations
        key, common = self._match(model, encoded.input_ids, boundary, input_key, pooling, media_start, media_end) if enabled else (None, 0)
        restore_started = self.restore_ms
        entry, cache_tier = self._acquire(key, ids_device)
        if entry is None:
            key = None
        hit = entry is not None
        reused = len(entry['ids']) if hit else 0
        build_ms = 0.0
        saved = 0
        vision_reused = bool(hit and media and reused >= media_end)
        if hit:
            self.hits += 1
            self.reused_tokens += reused
        elif enabled:
            self.misses += 1
        feature_hits = feature_misses = 0
        manual_vision = bool(media and not vision_reused and (enabled or pooling or feature_cache_enabled))
        inputs = {'input_ids': ids}
        if manual_vision:
            original = encoded.media['original_grid_thw'].to(ids.device) if pooling else media['image_grid_thw']
            vision, feature_hits, feature_misses = self._vision(base, encoded, media, original, feature_cache_enabled)
            if pooling:
                chunks, offset = [], 0
                for old, new in zip(original.tolist(), media['image_grid_thw'].tolist()):
                    h, w = old[1] // 2, old[2] // 2
                    ph, pw = new[1] // 2, new[2] // 2
                    features = vision[offset:offset + h*w].reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
                    chunks.append(F.adaptive_avg_pool2d(features, (ph, pw))[0].permute(1, 2, 0).reshape(ph*pw, -1))
                    offset += h*w
                features = torch.cat(chunks)
            else:
                features = vision
            embeddings = base.get_input_embeddings()(ids)
            image_mask, _ = base.get_placeholder_mask(ids, inputs_embeds=embeddings, image_features=features)
            inputs = {'inputs_embeds': embeddings.masked_scatter(image_mask, features.to(embeddings.dtype))}
        if not enabled:
            if manual_vision:
                output = base.language_model(**inputs, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
                logits = model.head(output.last_hidden_state, ids, mask, batch['records'], lexical_weight(model))[0]
            else:
                logits = model(batch)[0]
        if enabled:
            past = (entry['cache'] if cache_tier == 'cpu' else copy.deepcopy(entry['cache'])) if hit else None
            hidden_chunks = list(entry['chunks']) if hit else []
            points = set(checkpoint_boundaries)
            # A novel unrelated prefix seeds discovery. Shared-guide hits process
            # unique state plus questions in one forward; repeated states are
            # promoted to their own GPU checkpoints when useful.
            if seen_input or common < self.min_prefix_tokens:
                points.add(boundary)
            # Tiny extensions would duplicate the large recurrent snapshot for
            # very little saved work. Explicit boundaries remain exact.
            if common >= reused + self.min_prefix_tokens:
                points.add(common)
            if self.checkpoint_tokens:
                points.update(range(self.checkpoint_tokens, boundary, self.checkpoint_tokens))
            # Never checkpoint midway through an image: its rotary positions and
            # embeddings must be consumed as one media region.
            points = sorted(p for p in points if reused < p <= boundary
                            and (p >= self.min_prefix_tokens or p == boundary or p in checkpoint_boundaries)
                            and not (media_start is not None and media_start < p < media_end))
            prior = max([reused, *[p for p in points if p < boundary]])
            if boundary in points and prior and boundary - prior < self.min_prefix_tokens:
                points.remove(boundary)
            cursor = reused
            torch.cuda.current_stream().synchronize()
            started = time.perf_counter()
            for end in points:
                output = base.language_model(**{k: v[:, cursor:end] for k,v in inputs.items()},
                    attention_mask=mask[:, :end], position_ids=positions[:, :, cursor:end],
                    past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                hidden_chunks.append(output.last_hidden_state.clone())
                saved += self._retain(model, encoded.input_ids, end, input_key, pooling, media_start, past, hidden_chunks, parent_key=key)
                cursor = end
            torch.cuda.current_stream().synchronize()
            build_ms = (time.perf_counter() - started) * 1000
            suffix_inputs = {k: v[:, cursor:] for k,v in inputs.items()}
            suffix = base.language_model(**suffix_inputs, attention_mask=mask, position_ids=positions[:, :, cursor:],
                past_key_values=past, use_cache=True, return_dict=True)
            hidden = torch.cat([*hidden_chunks, suffix.last_hidden_state], dim=1)
            logits = model.head(hidden, ids, mask, batch['records'], lexical_weight(model))[0]
            self._remember(whole_key, encoded.input_ids, boundary, input_key, pooling)
        # Keep a high-water reserve for workload growth. Retained checkpoints are
        # subtracted conservatively so request tensors, clones and temporaries count.
        peak = torch.cuda.max_memory_allocated()
        self.workspace_bytes = max(self.workspace_bytes, peak - (self.base_bytes or 0)
                                   - min(retained_start, self._bytes()))
        status = 'hit' if hit else ('miss' if saved else 'not_retained') if enabled else 'disabled'
        return logits, {'prefix_cache_tier': cache_tier, 'prefix_restore_ms': round(self.restore_ms-restore_started, 1), 'prefix_offload_ms': round(self.host.offload_ms-offload_started, 1), 'prefill_mode': 'prefix_cache' if enabled else 'single_pass',
                        'prefix_cache': status, 'prefix_tokens': boundary,
                        'reused_prefix_tokens': reused, 'new_prefix_tokens': boundary - reused if enabled else boundary,
                        'prefix_build_ms': round(build_ms, 1), 'prefix_checkpoints_saved': saved,
                        'vision_prefix_reused': vision_reused, 'image_feature_cache_hits': feature_hits,
                        'image_feature_cache_misses': feature_misses}

    def plan_text_batch(self, model, records, boundaries, input_key, cache_enabled=True):
        from text_batching import shared_tokens
        common = shared_tokens(records, boundaries)
        enabled = cache_enabled and self.max_entries > 0 and (self._budget() > 0 or self.host.enabled)
        key, _ = self._match(model, records[0].input_ids, common, input_key, False, None, 0) if enabled else (None, 0)
        reused = len(self._lookup(key)['ids']) if key is not None else 0
        reason = 'host_prefix_requires_single_request' if key is not None and key not in self.entries else None
        if enabled:
            for e, boundary in zip(records, boundaries):
                best, _ = self._match(model, e.input_ids, boundary, input_key, False, None, 0)
                if best is not None and len(self._lookup(best)['ids']) > reused:
                    reason = 'better_individual_prefix'
                    break
            if reason is None and common > reused and all(boundary == common for boundary in boundaries):
                reason = 'exact_state_promotion'
        return {'common':common,'key':key,'enabled':enabled,'reason':reason}

    @torch.inference_mode()
    def forward_text_batch(self, model, processor, records, boundaries, input_key, *,
                           cache_enabled=True, checkpoint_boundaries=(), cache_plan=None):
        """Reuse a shared exact branch with independent hybrid state for each row.

        Only the common prefix is retained. Padded end states must not be saved
        as individual checkpoints: recurrent padding cannot be cropped away.
        """
        from text_batching import BatchUnavailable, clone_hybrid_cache, shared_tokens
        if any(e.media is not None for e in records):
            raise BatchUnavailable('images_require_single_request')
        offload_started = self.host.offload_ms
        plan = cache_plan if cache_plan is not None else self.plan_text_batch(model, records, boundaries, input_key, cache_enabled)
        if plan['reason']:
            raise BatchUnavailable(plan['reason'])
        common, enabled = plan['common'], plan['enabled']
        key = plan['key'] if plan['key'] in self.entries else None
        entry = self.entries.get(key) if key is not None else None
        reused = len(entry['ids']) if entry else 0
        device = next(model.parameters()).device
        batch = joint.collate_records(records, processor.tokenizer.pad_token_id, device)
        ids, mask = batch['input_ids'], batch['attention_mask']
        b = len(records)
        positions = torch.arange(ids.shape[1], device=device).view(1, 1, -1).expand(3, b, -1)
        saved = 0
        build_ms = 0.0
        language_started = time.perf_counter()
        if enabled and entry is None:
            # Prefer the same explicit context cut used by single requests.
            explicit = set(checkpoint_boundaries[0]) if checkpoint_boundaries else set()
            for points in checkpoint_boundaries[1:]:
                explicit.intersection_update(points)
            cuts = [p for p in explicit if p <= common]
            cut = max(cuts) if cuts else common
            if cut >= self.min_prefix_tokens or (cut and all(boundary == cut for boundary in boundaries)):
                tick = time.perf_counter()
                prefix = model.language_model.model.language_model(
                    input_ids=ids[:1, :cut], attention_mask=mask[:1, :cut],
                    position_ids=positions[:, :1, :cut], use_cache=True, return_dict=True)
                # A fresh prefix is immutable for the duration of this batch.
                entry = {'cache': prefix.past_key_values, 'chunks': (prefix.last_hidden_state,), 'ids': tuple(records[0].input_ids[:cut])}
                saved += self._retain(model, records[0].input_ids, cut, input_key, False, None,
                                      entry['cache'], entry['chunks'], copy_chunks=True)
                reused = cut
                torch.cuda.current_stream().synchronize()
                build_ms = (time.perf_counter()-tick)*1000
                del prefix
        if entry is not None:
            past = clone_hybrid_cache(entry['cache'], b)
            prefix_hidden = torch.cat(entry['chunks'], dim=1)
            output = model.language_model.model.language_model(
                input_ids=ids[:, reused:], attention_mask=mask, position_ids=positions[:, :, reused:],
                past_key_values=past, use_cache=True, return_dict=True)
            hidden = torch.cat([prefix_hidden.expand(b, -1, -1), output.last_hidden_state], dim=1)
        else:
            output = model.language_model.model.language_model(
                input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            hidden = output.last_hidden_state
        torch.cuda.current_stream().synchronize()
        language_ms = (time.perf_counter()-language_started)*1000
        head_started = time.perf_counter()
        logits = model.head(hidden, ids, mask, records, lexical_weight(model))
        torch.cuda.current_stream().synchronize()
        head_ms = (time.perf_counter()-head_started)*1000
        if key is not None:
            self.entries[key]['uses'] = self.entries[key].get('uses', 0) + b
            self.entries[key]['last_use'] = time.monotonic_ns()
            self.host.touch(key, b)
        if any(not torch.isfinite(v).all() for row in logits for v in row):
            raise ValueError('Non-finite batch decision output')
        if enabled:
            if key is not None:
                self.entries.move_to_end(key)
                self.hits += b
            else:
                self.misses += b
            self.reused_tokens += b*reused if key is not None else 0
            for e, boundary in zip(records, boundaries):
                whole = self._input_key(model, e.input_ids, boundary, input_key, False, None)
                self._remember(whole, e.input_ids, boundary, input_key, False)
        return logits, {'prefix_offload_ms': round(self.host.offload_ms-offload_started, 1), 'language_ms':round(language_ms,1),'head_ms':round(head_ms,1),'vision_ms':0.,'prefill_mode': 'text_batch', 'prefix_cache': 'hit' if key is not None else 'miss' if saved else 'not_retained' if enabled else 'disabled',
                        'reused_prefix_tokens': reused if key is not None else 0,
                        'shared_batch_prefix_tokens': reused, 'prefix_checkpoints_saved': saved,
                        'prefix_build_ms': round(build_ms, 1), 'image_feature_cache_hits': 0,
                        'image_feature_cache_misses': 0}


def answers(record, encoded, logits):
    return {q.question_id: joint.systemone_answer(record['questions'][q.question_id],
        dict(zip(q.option_ids, values.float().softmax(-1).tolist()))) for q, values in zip(encoded.questions, logits)}
