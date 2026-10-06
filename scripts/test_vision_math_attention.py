"""Compare tiled bidirectional vision attention with dense math SDPA."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from vision_math_attention import vision_math_attention

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
torch.manual_seed(103)
module = SimpleNamespace(training=False, _clef_vision_query_chunk=31,
                         _clef_vision_dense_threshold=64)
for length in [1, 63, 65, 129]:
    q, k, v = [torch.randn(2, 4, length, 32, device=args.device, dtype=torch.float16) for _ in range(3)]
    with sdpa_kernel(SDPBackend.MATH):
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=.17)
    actual, weights = vision_math_attention(module, q, k, v, scaling=.17)
    assert weights is None and actual.shape == (2, length, 4, 32)
    assert torch.allclose(reference, actual.transpose(1, 2), atol=.003, rtol=.003)
    # Tail queries must see all keys, including keys beyond a tile boundary.
    changed = v.clone(); changed[:, :, -1] += 10
    other, _ = vision_math_attention(module, q, k, changed, scaling=.17)
    assert not torch.equal(actual, other)
for kwargs in [dict(is_causal=True), dict(dropout=.1), dict(attention_mask=torch.ones(1, 1, 129, 129))]:
    try:
        vision_math_attention(module, q, k, v, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported attention policy accepted")
print("PASS: dense/tiled parity, small images, uneven tails, full-key access, unsupported-policy rejection")
