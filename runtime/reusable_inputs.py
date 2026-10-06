"""Bounded process-local CPU reuse of exact text and independently processed images.

No files or responses are cached. Processor/tokenizer objects must remain immutable
for their process lifetime; creating a replacement object gets a new namespace.
"""
from collections import OrderedDict
import hashlib
import json
import sys
import time
from threading import local

import torch
import joint_schema_model as joint
from ram_cache_budget import RAMCacheBudget


def cpu_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(cpu_bytes(k) + cpu_bytes(v) for k,v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(cpu_bytes(v) for v in value)
    return sys.getsizeof(value)


class InputCache:
    def __init__(self, max_mib=256, max_entries=128, token_mib=8, token_entries=1024, decode=None,
                 ram_budget=None, image_fraction=.25):
        self.images = OrderedDict()
        self.text = OrderedDict()
        self.auto = str(max_mib).lower() == 'auto'
        self.configured_bytes = None if self.auto else max(0, int(max_mib)) * 2**20
        self.limit = self.configured_bytes or 0
        self.max_entries = None if max_entries is None else max(0, int(max_entries))
        self.image_bytes = 0
        self.image_uses = {}
        self.ram_budget = ram_budget or RAMCacheBudget()
        self.ram_budget.register('images', lambda: self.image_bytes, self._evict_image, image_fraction)
        self.token_limit = max(0, int(token_mib)) * 2**20
        self.token_entries = max(0, int(token_entries))
        self.decode = decode
        self.local = local()
        self.image_hits = self.image_misses = self.token_hits = self.token_misses = 0
        self.image_evictions = self.token_evictions = 0

    @staticmethod
    def _put(bank, key, value, limit, count):
        size = cpu_bytes((key, value))
        evicted = 0
        if size > limit or count == 0:
            return evicted
        while bank and (len(bank) >= count or sum(v[1] for v in bank.values()) + size > limit):
            bank.popitem(last=False)
            evicted += 1
        bank[key] = (value, size)
        return evicted

    def _count(self, name):
        setattr(self, name, getattr(self, name)+1)
        counts = getattr(self.local, 'counts', None)
        if counts is None:
            self.local.counts = counts = {}
        counts[name] = counts.get(name, 0)+1

    def usage(self):
        counts = getattr(self.local, 'counts', {})
        return {name: counts.get(name, 0) for name in
                ('image_hits', 'image_misses', 'token_hits', 'token_misses')}

    def tokens(self, tokenizer, text, enabled=True):
        if not enabled:
            return joint._tokens(tokenizer, text)
        key = (id(tokenizer), hashlib.sha256(text.encode()).digest())
        with self.ram_budget.lock:
            if key in self.text:
                self._count('token_hits')
                self.text.move_to_end(key)
                value = self.text[key][0]
                return list(value)
            self._count('token_misses')
        ids = tuple(joint._tokens(tokenizer, text))
        with self.ram_budget.lock:
            if key not in self.text:
                self.token_evictions += self._put(self.text, key, ids, self.token_limit, self.token_entries)
        return list(ids)

    def _image_key(self, processor, image, kwargs):
        if isinstance(image, str):
            # Exact upload identity; changing encoding/EXIF forces validation again.
            identity = hashlib.sha256(image.encode()).hexdigest()
        else:
            identity = hashlib.sha256(str((image.mode, image.size)).encode() + image.tobytes()).hexdigest()
        return (id(processor), identity, json.dumps(kwargs, sort_keys=True, separators=(',', ':')))

    def _decode(self, image):
        if isinstance(image, str):
            if self.decode is None:
                raise ValueError('Image data URLs require a decoder')
            return self.decode(image)
        return image

    def _evict_image(self):
        if not self.images: return False
        key = min(self.images, key=lambda k: self.image_uses[k])
        self.image_bytes -= self.images.pop(key)[1]
        del self.image_uses[key]
        self.image_evictions += 1
        return True

    def _put_image(self, key, value):
        size = cpu_bytes((key, value))
        with self.ram_budget.lock:
            if key in self.images: return
            self.limit, _, sample = self.ram_budget.estimate('images')
            if self.configured_bytes is not None:
                self.limit = min(self.limit, self.configured_bytes) if sample is not None else self.configured_bytes
            if self.max_entries == 0 or size > self.limit:
                return
            while self.images and self.max_entries is not None and len(self.images) >= self.max_entries:
                self._evict_image()
            if not self.ram_budget.admit('images', size, self.configured_bytes):
                return
            self.images[key] = (value, size)
            self.image_uses[key] = (0, time.monotonic_ns())
            self.image_bytes += size

    def trim(self):
        with self.ram_budget.lock:
            self.ram_budget.admit('images', maximum=self.configured_bytes)

    def image_snapshots(self, processor, record):
        """Hold immutable cached inputs for bounded lookahead despite eviction."""
        snapshots = []
        for image in record.get('images') or []:
            key = self._image_key(processor, image, record.get('media_kwargs') or {})
            with self.ram_budget.lock:
                cached = self.images.get(key)
                if cached is None: return None
                snapshots.append((key, cached[0], cached[1]))
        return tuple(snapshots)

    def media(self, processor, record, enabled=True):
        self.trim()
        images = list(record.get('images') or [])
        if not images:
            return joint._encode_media(processor, record)
        kwargs = record.get('media_kwargs') or {}
        if not enabled or record.get('videos'):
            return joint._encode_media(processor, {**record, 'images': [self._decode(v) for v in images]})
        newline = joint._tokens(processor.tokenizer, '\n')
        ids, payloads, keys = [], [], []
        snapshots = record.get('_cache_image_snapshots')
        for index, image in enumerate(images):
            key = snapshots[index][0] if snapshots is not None else self._image_key(processor, image, kwargs)
            with self.ram_budget.lock:
                cached = (snapshots[index][1], 0) if snapshots is not None else self.images.get(key)
                if cached is not None:
                    self._count('image_hits')
                    if key in self.images:
                        self.images.move_to_end(key)
                        self.image_uses[key] = (self.image_uses[key][0]+1, time.monotonic_ns())
                    item_ids, item_media = cached[0]
                else:
                    self._count('image_misses')
            if cached is None:
                item_ids, item_media = joint._encode_media(processor,
                    {'images': [self._decode(image)], 'media_kwargs': kwargs})
                if not newline or item_ids[-len(newline):] != newline:
                    raise ValueError('Unsupported processor media terminator')
                item_ids = tuple(item_ids[:-len(newline)])
                item_media = {k: (v[:-len(newline)] if k in joint.MEDIA_TOKEN_KEYS else v)
                              for k,v in item_media.items()}
                self._put_image(key, (item_ids, item_media))
            ids.extend(item_ids)
            payloads.append(item_media)
            keys.append(key)
        ids.extend(newline)
        media = {k: torch.cat([p[k] for p in payloads], dim=0) for k in joint.MEDIA_BATCH_KEYS if k in payloads[0]}
        for key in joint.MEDIA_TOKEN_KEYS:
            if key in payloads[0]:
                media[key] = [v for p in payloads for v in p[key]] + [0] * len(newline)
        media['image_cache_keys'] = tuple(keys)
        return ids, media

    def clear(self):
        with self.ram_budget.lock:
            self.images.clear()
            self.text.clear()
            self.image_bytes = 0
            self.image_uses.clear()

    def stats(self):
        with self.ram_budget.lock:
            return self._stats()

    def _stats(self):
        limit, _, sample = self.ram_budget.estimate('images')
        if self.configured_bytes is not None:
            limit = min(limit, self.configured_bytes) if sample is not None else self.configured_bytes
        return {'image_entries': len(self.images), 'image_mib': round(self.image_bytes / 2**20, 2),
                'image_limit_mib': limit / 2**20, 'image_max_entries': self.max_entries,
                'image_budget_mode': 'auto' if self.auto else 'fixed', 'image_eviction_policy': 'lfu_lru_tiebreak',
                'image_hits': self.image_hits, 'image_misses': self.image_misses, 'image_evictions': self.image_evictions,
                'token_entries': len(self.text), 'token_mib': round(sum(v[1] for v in self.text.values()) / 2**20, 2),
                'token_limit_mib': self.token_limit / 2**20, 'token_max_entries': self.token_entries,
                'token_hits': self.token_hits, 'token_misses': self.token_misses, 'token_evictions': self.token_evictions}
