"""Request-local implicit causal attention for unpadded Qwen3.5 prefill.

Lower-right causality includes every cached key and only the already processed
part of the new query block. A plain is_causal=True would be wrong for Q != K.
Use Torch's existing CUDA kernels; restore configs before nested queue work.
"""
import copy
from contextlib import contextmanager
import torch

NAME = 'clef_prefill_lower_right'


def lower_right_sdpa(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
    from torch.nn.attention.bias import causal_lower_right
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    if attention_mask is not None or module.training or query.shape[-2] > key.shape[-2]:
        raise ValueError('Implicit prefill requires isolated unpadded causal inference')
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    output = torch.nn.functional.scaled_dot_product_attention(query, key, value,
        attn_mask=causal_lower_right(query.shape[-2], key.shape[-2]),
        dropout_p=dropout, scale=scaling, is_causal=False)
    return output.transpose(1, 2).contiguous(), None


class ImplicitPrefillAttention:
    def __init__(self, language, *, enabled, device, unpadded=True):
        self.originals = []
        self.active = False
        self.enabled = False
        self.reason = 'disabled'
        if not enabled:
            return
        if (torch.device(device).type != 'cuda' or torch.version.hip
                or not torch.backends.cuda.mem_efficient_sdp_enabled()):
            self.reason = 'unsupported_backend'; return
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
        config = language.config
        modules = [layer.self_attn for layer in language.layers
                   if getattr(layer, 'layer_type', None) == 'full_attention']
        if (not unpadded or language.training or config._attn_implementation != 'sdpa'
                or not getattr(config, 'is_causal', True)
                or not modules or any(not isinstance(m, Qwen3_5Attention) for m in modules)
                or any(kind not in {'full_attention','linear_attention'} for kind in config.layer_types)):
            self.reason = 'unsupported_model'; return
        try:
            from torch.nn.attention.bias import causal_lower_right
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except ImportError:
            self.reason = 'unsupported_library'; return
        # Registries only dispatch requests whose config selects our private name.
        ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, lambda **kwargs: None)
        ALL_ATTENTION_FUNCTIONS.register(NAME, lower_right_sdpa)
        self.originals = [(language, config), *[(m, m.config) for m in modules]]
        self.enabled = True; self.reason = 'implicit_lower_right'
        try:
            self.install()
        except BaseException:
            self.close()
            raise

    def install(self):
        if not self.enabled or self.active:
            return
        self.active = True
        for module, original in self.originals:
            module.config = copy.copy(original)
            module.config._attn_implementation = NAME

    def restore(self):
        if self.active:
            for module, config in self.originals:
                module.config = config
            self.active = False

    @contextmanager
    def suspended(self):
        self.restore()
        try:
            yield
        finally:
            self.install()

    def close(self):
        self.restore()
        self.enabled = False
