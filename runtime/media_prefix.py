"""Exact still-image ancestry for position-specific prefix namespaces."""
from dataclasses import dataclass, replace
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class MediaPrefix:
    identities: tuple
    spans: tuple
    namespaces: tuple

    @classmethod
    def build(cls, processor, encoded, images, options=None):
        identities = tuple(digest([image, options]) for image in images)
        return cls(identities, (), ()).rebuild(processor, encoded)

    def rebuild(self, processor, encoded):
        # Raises for video or unexpected markers; callers then fall back to the
        # conservative whole-media key.
        media = encoded.media
        if media is None or 'video_grid_thw' in media:
            raise ValueError('Image prefix checkpoints require still images')
        start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        offset = media['token_offset']
        stop = offset + len(media['mm_token_type_ids'])
        spans, beginning = [], None
        for position in range(offset, stop):
            token = encoded.input_ids[position]
            if token == start_id:
                if beginning is not None: raise ValueError('Nested image marker')
                beginning = position
            elif token == end_id:
                if beginning is None: raise ValueError('Unpaired image marker')
                spans.append((beginning, position + 1)); beginning = None
        grids = media['image_grid_thw'].tolist()
        if beginning is not None or len(spans) != len(self.identities) or len(grids) != len(spans):
            raise ValueError('Image markers, identities and grids disagree')
        ancestry, namespaces = [], []
        for identity, grid in zip(self.identities, grids):
            ancestry.append((identity, grid))
            namespaces.append(digest(['image-prefix-v1', ancestry]))
        return MediaPrefix(self.identities, tuple(spans), tuple(namespaces))

    def at(self, position):
        count = sum(end <= position for _, end in self.spans)
        return self.namespaces[count - 1] if count else 'text'

    def safe(self, position):
        for start, end in self.spans:
            if start < position < end: return start
        return position

    def intervals(self, boundary):
        # A namespace can cover text between images, but never part of an image.
        rows = [('text', False, 0, self.spans[0][0])]
        for index, (_, end) in enumerate(self.spans):
            stop = self.spans[index + 1][0] if index + 1 < len(self.spans) else boundary
            rows.append((self.namespaces[index], True, end, stop))
        return rows

    @property
    def checkpoints(self):
        return tuple(end for _, end in self.spans)


def remaining_images(encoded, image_token_id, cursor):
    """Vision-only view excluding complete images already in the language state."""
    media = encoded.media
    if media is None: return encoded, 0
    consumed = sum(token == image_token_id for token in encoded.input_ids[:cursor])
    skip, count = 0, 0
    for grid in media['image_grid_thw'].tolist():
        size = math.prod(grid) // 4
        if count + size > consumed: break
        skip += 1; count += size
    if not skip: return encoded, 0
    original = media.get('original_grid_thw', media['image_grid_thw'])
    pixels = sum(math.prod(grid) for grid in original[:skip].tolist())
    view = {**media, 'pixel_values': media['pixel_values'][pixels:],
            'image_grid_thw': media['image_grid_thw'][skip:]}
    if 'original_grid_thw' in media: view['original_grid_thw'] = original[skip:]
    if 'image_cache_keys' in media: view['image_cache_keys'] = media['image_cache_keys'][skip:]
    return replace(encoded, media=view), skip
