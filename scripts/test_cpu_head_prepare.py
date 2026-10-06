"""Native-head equivalence across all question types and failed-call cleanup."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/cloudflare'))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import joint_schema_model as joint
from cpu_head_prepare import head_from_cpu

if not torch.cuda.is_available() or torch.version.hip:
    raise SystemExit('This test requires an NVIDIA CUDA GPU')
torch.manual_seed(7319)
questions=tuple(joint.EncodedQuestion(str(i),i,(7+i*21,19+i*21),
    ((0,3),(20,33)) if i==0 else ((0,3),(20,33),(126,129)),
    ('a','b') if i==0 else ('a','b','c')) for i in range(3))
record=joint.EncodedRecord(tuple(range(129)),questions,'synthetic')
errors=[]
for dtype in (torch.float16,torch.bfloat16):
    head=joint.JointSchemaHead(32,64,2,2,4,128,0.).to(device='cuda',dtype=dtype).eval()
    hidden=torch.randn(1,129,32,device='cuda',dtype=dtype)
    ids=torch.randint(0,64,(1,129),device='cuda');mask=torch.ones_like(ids)
    lexical=torch.randn(64,32,device='cuda',dtype=dtype)
    hidden_cpu=hidden.cpu();saved=hidden_cpu.clone()
    original_norm=head.hidden_norm.forward;original_projection=head.memory_projection.forward
    with torch.inference_mode(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        expected=head(hidden,ids,mask,[record],lexical)
        for chunk in (1,31,256):
            actual,usage=head_from_cpu(head,hidden_cpu,ids,mask,[record],lexical,chunk_tokens=chunk)
            for a,b in zip(actual[0],expected[0]):
                torch.testing.assert_close(a,b,atol=.01,rtol=.04)
                errors.append((str(dtype),chunk,(a-b).abs().max().item()))
            assert head.hidden_norm.forward==original_norm
            assert head.memory_projection.forward==original_projection
            assert torch.equal(hidden_cpu,saved)
        scorer=head.residual_scorer.forward
        def fail(*args):raise RuntimeError('intentional cleanup test')
        head.residual_scorer.forward=fail
        try:
            try:head_from_cpu(head,hidden_cpu,ids,mask,[record],lexical,chunk_tokens=31)
            except RuntimeError as exc:assert str(exc)=='intentional cleanup test'
            else:raise AssertionError('Injected failure was not raised')
        finally:head.residual_scorer.forward=scorer
        assert head.hidden_norm.forward==original_norm
        assert head.memory_projection.forward==original_projection
        restored=head(hidden,ids,mask,[record],lexical)
        for a,b in zip(restored[0],expected[0]):torch.testing.assert_close(a,b,atol=0,rtol=0)
print('PASS: native choice/noul/score head agreement; cross-chunk spans; unchanged CPU inputs; failure cleanup')
print('maximum logit difference',max(row[2] for row in errors))
