"""Bound math-SDPA vision workspace without dropping patches or key context.

Only query rows are tiled. Each tile attends to all keys in its original image
segment, preserving upstream bidirectional vision attention and rotary positions.
This is inference-only and deliberately does not handle causal or masked text.
"""
import copy
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def vision_math_attention(module, query, key, value, attention_mask=None, *,
                          dropout=0.0, scaling=None, is_causal=False, **kwargs):
    if module.training or dropout or is_causal or attention_mask is not None:
        raise ValueError("Tiled vision attention requires unmasked noncausal inference")
    if query.shape[1] != key.shape[1] or key.shape != value.shape:
        raise ValueError("Tiled vision attention requires matching Q/K/V heads")
    chunk = module._clef_vision_query_chunk
    length = query.shape[-2]
    # Small images keep the ordinary path to avoid extra launch overhead.
    step = length if length <= module._clef_vision_dense_threshold else chunk
    result = torch.empty_like(query)
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, length, step):
            end = min(start + step, length)
            result[:, :, start:end] = F.scaled_dot_product_attention(
                query[:, :, start:end], key, value, dropout_p=0.0,
                scale=scaling, is_causal=False)
    return result.transpose(1, 2).contiguous(), None


def install_vision_math_attention(model, *, query_chunk=256, dense_threshold=1024):
    if query_chunk < 1 or dense_threshold < 0:
        raise ValueError("Invalid vision attention tiling policy")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register("clef_math_vision", vision_math_attention)
    count = 0
    for module in model.language_model.model.visual.modules():
        if isinstance(module, Qwen3_5VisionAttention):
            # Do not alter the language backbone's attention implementation.
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "clef_math_vision"
            module._clef_vision_query_chunk = query_chunk
            module._clef_vision_dense_threshold = dense_threshold
            count += 1
    if not count:
        raise RuntimeError("No compatible Qwen3.5 vision attention layers found")
    return count
