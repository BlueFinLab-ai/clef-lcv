"""Conservative memory projection for the measured NF4/efficient-attention builds.

Calibration anchors are request workspace above model weights, measured with
8K initial + 4K continuation prefill and four typed questions. Growth accounts
for KV, retained hidden states, an expanded full-attention KV pair and a KV
concatenation copy, with 25% additional slack. This is an estimate, not a model
accuracy guarantee or a CUDA allocation reservation.
"""
import math
import time

MIB = 2**20


class ContextBudget:
    def __init__(self, config, *, dtype_bytes, calibrated, configured_limit,
                 image_limit, reserve_mib=512, mode='auto', quantum=4096, image_chunked=False):
        if mode not in {'auto', 'fixed'} or reserve_mib < 0 or (image_limit is not None and image_limit < 1):
            raise ValueError('Invalid context budget mode or reserve')
        self.configured_limit = configured_limit
        self.image_limit = image_limit
        self.image_chunked = image_chunked
        self.reserve_bytes = reserve_mib * MIB
        self.quantum = quantum
        self.architecture_limit = config.max_position_embeddings
        types = getattr(config, 'layer_types', [])
        full_layers = types.count('full_attention')
        self.kv_bytes = full_layers * 2 * config.num_key_value_heads * config.head_dim * dtype_bytes
        self.hidden_bytes = config.hidden_size * dtype_bytes
        # SDPA currently expands GQA K/V; concatenating the current layer also
        # temporarily holds its old KV. Include both, even if peaks do not overlap.
        expanded = 2 * config.num_attention_heads * config.head_dim * dtype_bytes
        concat = 2 * config.num_key_value_heads * config.head_dim * dtype_bytes
        self.growth_bytes = math.ceil((self.kv_bytes + self.hidden_bytes + expanded + concat + 32) * 1.25)
        anchors = {32: (24576, 2000 * MIB), 64: (16384, 2210 * MIB)}
        self.reference_tokens, self.reference_workspace = anchors.get(config.num_hidden_layers, (0, 0))
        self.computed = mode == 'auto' and calibrated and bool(self.reference_tokens)
        self.image_computed = self.computed and image_chunked and image_limit is None
        self.mode = 'memory_estimate' if self.computed else 'configured'
        self.fallback_reason = None if self.computed else ('explicit fixed mode' if mode == 'fixed' else 'uncalibrated build or prefill policy')
        self.oom_ceiling = self.architecture_limit
        self.last = None

    def workspace(self, tokens):
        # Preserve the measured initial-chunk floor; never assume small prompts
        # scale to long prompts using only KV bytes/token.
        return self.reference_workspace + max(0, tokens - self.reference_tokens) * self.growth_bytes

    def observe(self, tokens, workspace_bytes):
        if self.computed:
            excess = workspace_bytes - self.workspace(tokens)
            if excess > 0:
                self.reference_workspace += excess

    def record_oom(self, tokens):
        if self.computed:
            self.oom_ceiling = min(self.oom_ceiling, max(0, (tokens - 1) // self.quantum * self.quantum))

    def snapshot(self, *, free_bytes, reserved_bytes, allocated_bytes, cache_bytes):
        # Reserved but unallocated PyTorch blocks are reusable. Prefix/image GPU
        # entries are reclaimable because long text evicts them before inference.
        available = max(0, free_bytes + max(0, reserved_bytes - allocated_bytes) + cache_bytes)
        usable = max(0, available - self.reserve_bytes)
        computed_limit = None
        if self.computed:
            if usable < self.reference_workspace:
                # The calibrated 8K-first policy lacks a safe projection at this
                # memory level. Fail closed rather than inventing a short limit.
                computed_limit = 0
            else:
                projected = self.reference_tokens + int((usable - self.reference_workspace) // self.growth_bytes)
                computed_limit = min(self.architecture_limit, self.oom_ceiling, projected)
                computed_limit = computed_limit // self.quantum * self.quantum
        accepted = computed_limit if self.computed else min(self.configured_limit, self.architecture_limit)
        self.last = {
            'max_input_tokens': accepted,
            'max_input_tokens_with_images': accepted if self.image_computed else min(accepted, self.image_limit if self.image_limit is not None else self.configured_limit),
            'context_limit': {
                'mode': self.mode, 'estimated': self.computed,
                'computed_max_input_tokens': computed_limit,
                'configured_max_input_tokens': self.configured_limit,
                'configured_cap_applied': not self.computed,
                'model_max_position_embeddings': self.architecture_limit,
                'image_limit_basis': 'computed_memory_estimate' if self.image_computed else ('configured_chunked_ceiling' if self.image_chunked else 'configured_single_pass_ceiling'),
                'configured_max_input_tokens_with_images': self.image_limit,
                'available_request_mib': round(available / MIB, 1),
                'safety_reserve_mib': self.reserve_bytes / MIB,
                'full_attention_kv_bytes_per_token': self.kv_bytes,
                'retained_hidden_bytes_per_token': self.hidden_bytes,
                'projected_growth_bytes_per_token': self.growth_bytes,
                'calibration_reference_tokens': self.reference_tokens if self.computed else None,
                'calibration_workspace_mib': round(self.reference_workspace / MIB, 1) if self.computed else None,
                'rounding_tokens': self.quantum,
                'fallback_reason': self.fallback_reason,
                'snapshot_unix_seconds': int(time.time()),
            },
        }
        return self.last
