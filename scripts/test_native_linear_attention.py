"""Compare native GDN partitions and mutable continuation state on CPU or GPU."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from native_linear_attention import native_chunk_kernel

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--device', default='cpu')
a = p.parse_args()
torch.manual_seed(782)
torch.set_num_threads(2)
for size in [1, 63, 64, 177]:
    heads, width = 2, 16
    tensors = [torch.randn(2, size, heads, width, device=a.device) for _ in range(3)]
    tensors += [-torch.rand(2, size, heads, device=a.device) * .1,
                torch.rand(2, size, heads, device=a.device)]
    options = dict(initial_state=torch.randn(2, heads, width, width, device=a.device) * .01,
                   output_final_state=True, use_qk_l2norm_in_kernel=True)
    reference = torch_chunk_gated_delta_rule(*tensors, **options)
    for block in [16, 32, 64]:
        kernel = native_chunk_kernel(torch_chunk_gated_delta_rule, block)
        output = kernel(*tensors, **options)
        for left, right in zip(reference, output):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
        # Subsequent prefill depends on all of the returned recurrent state.
        tail = [v[:, :1] for v in tensors]
        left = torch_chunk_gated_delta_rule(*tail, **{**options, 'initial_state': reference[1]})
        right = kernel(*tail, **{**options, 'initial_state': output[1]})
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)
assert native_chunk_kernel(torch_chunk_gated_delta_rule, 64) is torch_chunk_gated_delta_rule
try:
    native_chunk_kernel(torch_chunk_gated_delta_rule, 8)
except ValueError:
    pass
else:
    raise AssertionError('Unvalidated partition accepted')
print('PASS: native GDN blocks, uneven lengths, batch isolation and continuation states')
