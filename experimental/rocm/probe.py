"""Small hardware correctness gate before spending VRAM on the model."""
import argparse
import copy
import json
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--base-only", action="store_true")
args = parser.parse_args()
assert torch.version.hip and torch.cuda.is_available()
assert torch.cuda.device_count() == 1
torch.manual_seed(42)
report = {"torch": torch.__version__, "hip": torch.version.hip,
          "gpu": str(torch.cuda.get_device_properties(0)), "checks": {}}
started = time.perf_counter()

def check(name, actual, expected, tolerance):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert torch.isfinite(actual).all(), name
    error = (actual-expected).abs().max().item()
    scale = max(expected.abs().max().item(), 1e-6)
    assert error / scale < tolerance, (name, error/scale)
    report["checks"][name] = {"relative_max_error": error/scale}
    print(json.dumps({"passed": name, **report["checks"][name]}), flush=True)

a, b = torch.randn(128, 256).half(), torch.randn(256, 128).half()
check("fp16_matmul", a.cuda() @ b.cuda(), a.float() @ b.float(), 0.003)
x = torch.randn(8, 128).half()
check("layer_norm", F.layer_norm(x.cuda(), (128,)), F.layer_norm(x.float(), (128,)), 0.003)
q, k, v = [torch.randn(1, 4, 64, 32) for _ in range(3)]
with sdpa_kernel(SDPBackend.MATH):
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    actual = F.scaled_dot_product_attention(q.half().cuda(), k.half().cuda(), v.half().cuda(), is_causal=True)
    check("causal_attention", actual, reference, 0.004)
    suffix = F.scaled_dot_product_attention(q[:, :, 32:].half().cuda(), k.half().cuda(), v.half().cuda(),
        attn_mask=(torch.arange(64)[None, :] <= torch.arange(32, 64)[:, None]).cuda())
    check("chunked_attention", suffix, reference[:, :, 32:], 0.004)
conv_x, conv_w = torch.randn(1, 16, 64).half(), torch.randn(16, 1, 4).half()
check("depthwise_conv", F.conv1d(conv_x.cuda(), conv_w.cuda(), groups=16),
      F.conv1d(conv_x.float(), conv_w.float(), groups=16), 0.004)
if not args.base_only:
    import bitsandbytes as bnb
    report["bitsandbytes"] = bnb.__version__
    for nested in [False, True]:
        w = torch.randn(128, 256).half()
        packed, state = bnb.functional.quantize_4bit(w.cuda(), quant_type="nf4", compress_statistics=nested)
        restored = bnb.functional.dequantize_4bit(packed, state)
        cpu_state = copy.deepcopy(state)
        cpu_state.to("cpu")
        reference = bnb.functional.dequantize_4bit(packed.cpu(), cpu_state)
        check(f"nf4_dequant_nested_{nested}", restored, reference, 0.0001)
        check(f"nf4_linear_nested_{nested}", bnb.matmul_4bit(a.cuda(), packed.t(), quant_state=state),
              a.float() @ reference.float().t(), 0.004)
        check(f"nf4_single_row_nested_{nested}", bnb.matmul_4bit(a[:1].cuda(), packed.t(), quant_state=state),
              a[:1].float() @ reference.float().t(), 0.004)
        if not nested:
            embedding = bnb.nn.EmbeddingNF4(128, 256, dtype=torch.float16, device="meta")
            embedding.dtype = torch.float16
            embedding.weight = bnb.nn.Params4bit.from_prequantized(
                packed.cpu(), {key: value.cpu() for key, value in state.as_dict(packed=True).items()},
                requires_grad=False, device="cpu", module=embedding)
            embedding = embedding.cuda()
            ids = torch.tensor([0, 3, 127, 3], device="cuda")
            check("nf4_compact_embedding", embedding(ids), reference[ids.cpu()], 0.0001)
    report["checks"]["nf4_roundtrip"] = {"relative_rms_error": ((restored.float().cpu()-w.float()).square().mean()/w.float().square().mean()).sqrt().item()}
    assert report["checks"]["nf4_roundtrip"]["relative_rms_error"] < 0.15
torch.cuda.synchronize()
report.update(passed=True, seconds=time.perf_counter()-started,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
print(json.dumps(report, indent=2), flush=True)
