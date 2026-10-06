"""Select the block size of Transformers' native PyTorch GDN algorithm.

This is independent of outer prefill chunking. It changes the partition of the
same recurrence, retaining the native FP32 computation and continuation state.
"""


def native_chunk_kernel(kernel, block_tokens=64):
    if block_tokens not in {16, 32, 64}:
        raise ValueError('Native GDN block size must be 16, 32 or 64')
    if block_tokens == 64:
        return kernel

    def torch_block_gated_delta_rule(*args, **kwargs):
        # Callers supply Q/K/V/g/beta; refusing a conflicting explicit block size
        # avoids silently changing the positional API of the upstream routine.
        if len(args) > 5:
            raise TypeError('Native block override requires keyword GDN options')
        if 'chunk_size' in kwargs and kwargs['chunk_size'] != block_tokens:
            raise ValueError('Conflicting native GDN block size')
        kwargs['chunk_size'] = block_tokens
        return kernel(*args, **kwargs)

    return torch_block_gated_delta_rule
