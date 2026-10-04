"""Bounded process-local CPU reuse of exact text and independently processed images.

No files or responses are cached. Processor/tokenizer objects must remain immutable
for their process lifetime; creating a replacement object gets a new namespace.
"""
from collections import OrderedDict
import hashlib
import json
import sys

import torch
import joint_schema_model as joint


def cpu_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(cpu_bytes(k) + cpu_bytes(v) for k,v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(cpu_bytes(v) for v in value)
    return sys.getsizeof(value)


class InputCache:
    def __init__(self, max_mib=256, max_entries=128, token_mib=8, token_entries=1024, decode=None):
        self.images = OrderedDict()
        self.text = OrderedDict()
        self.limit = max(0, int(max_mib)) * 2**20
        self.max_entries = max(0, int(max_entries))
        self.token_limit = max(0, int(token_mib)) * 2**20
        self.token_entries = max(0, int(token_entries))
        self.decode = decode
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

    def tokens(self, tokenizer, text, enabled=True):
        if not enabled:
            return joint._tokens(tokenizer, text)
        key = (id(tokenizer), hashlib.sha256(text.encode()).digest())
        if key in self.text:
            self.token_hits += 1
            self.text.move_to_end(key)
            return list(self.text[key][0])
        self.token_misses += 1
        ids = tuple(joint._tokens(tokenizer, text))
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

    def media(self, processor, record, enabled=True):
        images = list(record.get('images') or [])
        if not images:
            return joint._encode_media(processor, record)
        kwargs = record.get('media_kwargs') or {}
        if not enabled or record.get('videos'):
            return joint._encode_media(processor, {**record, 'images': [self._decode(v) for v in images]})
        newline = joint._tokens(processor.tokenizer, '\n')
        ids, payloads, keys = [], [], []
        for image in images:
            key = self._image_key(processor, image, kwargs)
            if key in self.images:
                self.image_hits += 1
                self.images.move_to_end(key)
                item_ids, item_media = self.images[key][0]
            else:
                self.image_misses += 1
                item_ids, item_media = joint._encode_media(processor,
                    {'images': [self._decode(image)], 'media_kwargs': kwargs})
                if not newline or item_ids[-len(newline):] != newline:
                    raise ValueError('Unsupported processor media terminator')
                item_ids = tuple(item_ids[:-len(newline)])
                item_media = {k: (v[:-len(newline)] if k in joint.MEDIA_TOKEN_KEYS else v)
                              for k,v in item_media.items()}
                self.image_evictions += self._put(self.images, key, (item_ids, item_media), self.limit, self.max_entries)
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
        self.images.clear()
        self.text.clear()

    def stats(self):
        return {'image_entries': len(self.images), 'image_mib': round(sum(v[1] for v in self.images.values()) / 2**20, 2),
                'image_limit_mib': self.limit / 2**20, 'image_max_entries': self.max_entries,
                'image_hits': self.image_hits, 'image_misses': self.image_misses, 'image_evictions': self.image_evictions,
                'token_entries': len(self.text), 'token_mib': round(sum(v[1] for v in self.text.values()) / 2**20, 2),
                'token_limit_mib': self.token_limit / 2**20, 'token_max_entries': self.token_entries,
                'token_hits': self.token_hits, 'token_misses': self.token_misses, 'token_evictions': self.token_evictions}
