"""Bounded text-batch admission; no GPU work, timers or email-specific rules."""
import copy
import math
import torch

MIB = 2**20


class BatchUnavailable(ValueError):
    pass


def clone_hybrid_cache(cache, count):
    """Clone KV plus convolution/recurrent state; never mutate a retained branch."""
    result = copy.deepcopy(cache)
    if not hasattr(result, 'layers'):
        raise BatchUnavailable('unsupported_cache_layout')
    for layer in result.layers:
        for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != 1:
                    raise BatchUnavailable('unsupported_cache_batch_axis')
                setattr(layer, name, value.repeat_interleave(count, dim=0))
        if hasattr(layer, 'max_batch_size'):
            layer.max_batch_size = count
    return result


def shared_tokens(records, boundaries):
    common = 0
    for values in zip(*(e.input_ids[:boundary] for e, boundary in zip(records, boundaries))):
        if len(set(values)) != 1:
            break
        common += 1
    return common


class TextBatchAdmission:
    def __init__(self, config, *, padded_tokens=6000, record_tokens=4096, length_ratio=.75):
        if padded_tokens < 1 or record_tokens < 1 or not 0 < length_ratio <= 1:
            raise ValueError('Invalid text batch limits')
        self.config, self.padded_tokens, self.record_tokens, self.length_ratio = config, padded_tokens, record_tokens, length_ratio
        self.observations = {}
        self.sizes = {}
        self.oom_limits = {}
        self.batches = self.requests = self.oom_fallbacks = 0
        self.fallbacks = {}

    def reason(self, lengths):
        maximum = max(lengths)
        padded = len(lengths) * maximum
        if maximum > self.record_tokens:
            return 'long_request'
        if padded > self.padded_tokens:
            return 'padded_token_budget'
        if min(lengths) / maximum < self.length_ratio:
            return 'length_mismatch'
        if padded >= self.oom_limits.get(len(lengths), math.inf):
            return 'previous_batch_oom'
        return None

    def workspace(self, lengths, prefix_tokens=0):
        c = self.config
        b, n = len(lengths), max(lengths)
        tail = max(1, n-prefix_tokens)
        types = c.layer_types
        kv = types.count('full_attention') * 2 * c.num_key_value_heads * c.head_dim * 2 * n * b
        linear = types.count('linear_attention') * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4 * b
        hidden = c.hidden_size * 2 * n * b * 4
        expanded = 2 * c.num_attention_heads * c.head_dim * 2 * n * b
        projections = tail * c.linear_num_value_heads * max(c.linear_key_head_dim, c.linear_value_head_dim) * 4 * b * 12
        block_states = math.ceil(tail/64) * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4 * b * 2
        estimate = 256*MIB + math.ceil((kv+linear+hidden+expanded+projections+block_states)*1.5)
        observed = self.observations.get(b)
        if observed:
            estimate = max(estimate, math.ceil(observed['bytes'] * max(1, b*n/observed['padded_tokens'])))
        return estimate

    def observe(self, lengths, workspace):
        b = len(lengths)
        old = self.observations.get(b)
        if old is None or workspace > old['bytes']:
            self.observations[b] = {'bytes': int(workspace), 'padded_tokens': b*max(lengths)}
        self.sizes[b] = self.sizes.get(b,0)+1
        self.batches += 1
        self.requests += b

    def fallback(self, reason):
        self.fallbacks[reason] = self.fallbacks.get(reason, 0)+1

    def oom(self, lengths):
        b = len(lengths)
        self.oom_limits[b] = min(self.oom_limits.get(b, math.inf), b*max(lengths))
        self.oom_fallbacks += 1
        self.fallback('gpu_oom')

    def stats(self):
        return {'text_only': True, 'padded_token_budget': self.padded_tokens,
                'max_record_tokens': self.record_tokens, 'minimum_length_ratio': self.length_ratio,
                'gpu_batch_sizes': dict(self.sizes), 'gpu_batches': self.batches, 'batched_requests': self.requests,
                'fallbacks': dict(self.fallbacks), 'oom_fallbacks': self.oom_fallbacks,
                'oom_padded_token_limits': dict(self.oom_limits),
                'observed_workspace': {k:{'mib':round(v['bytes']/MIB,1),'padded_tokens':v['padded_tokens']} for k,v in self.observations.items()}}
