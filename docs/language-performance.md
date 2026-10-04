# Language performance experiments

Measured October 3, 2026, on full Clef 27B with the existing optimized FLA kernels,
using a spare RTX 3090 at a constant 250 W power limit. These are isolated
experiments; the saved inference adapters and running services were unchanged.
They should not be compared directly with the 400 W live service.

Two private photographs, 512² and 1024² processing budgets, and six typed questions
formed the timing workload. Each condition used two warm-ups and five measured
repeats per photograph and resolution. Reported values are means of the two
photographs' medians. Controls were measured in each experiment batch.

## Main results at 1024 fidelity

| Approach | Language time | Whole request | Request speedup vs matched control |
|---|---:|---:|---:|
| Compact schema formatting | 1.405 s | 1.757 s | 1.06× |
| Warm image/state prefix reuse | 0.729 s | 0.947 s | 1.97× |
| Compact schema + warm prefix reuse | 0.505 s | 0.701 s | 2.74× |
| 2×2 visual-feature pooling, fresh input | 0.930 s | 1.270 s | 1.50× |
| 2×2 pooling + compact schema + FlashAttention | 0.754 s | 1.075 s | 1.78× |
| Warm vision-feature reuse only | 1.564 s | 1.745 s | 1.10× |

Matched baseline requests were approximately 1.87–1.92 s. Baseline language time
was approximately 1.52–1.57 s. The benefits differ at 512: compact schema alone
gave 1.15× request speedup; compact schema plus prefix reuse gave 1.66×.

Group-128 Marlin INT4 language linears, NF4 FP16 linear computation, cuBLASLt,
NF4 projection fusion and compiled MLP functions did not establish a meaningful
language-stage speedup. INT4 and FP16 computation were slower at high fidelity.
The vision encoder remained NF4 in these language experiments.

## Quality and scope

Compact schema plus prefix reuse retained all 24 original visual answers.
Compact formatting also matched frozen NF4 category labels for all 27 selected
messages from the prior private 100-email benchmark. Both formats agreed with
25/27 frozen assistant reference labels, which are not human-verified truth.
Probabilities changed by up to 6.35 percentage points despite unchanged labels.

Eight cache checks with new or reordered questions retained the same choices as
uncached inference. Their maximum probability difference was 0.154 percentage
points. Cache hits require identical image bytes, processing settings and state;
the question suffix can change. Similar wording or a different image does not
qualify for this prototype's whole-prefix cache.

The extra-detail test included printed text, frame patterns, lifeboats, clothing
texture and background details. Its baseline scored 23/24; 2×2 pooling retained
the same answers, including correct printed text at 1024. Combining pooling and
compact schema lost one extra-detail answer at 512. Aggressive 4×4 pooling lost
both printed-text details and an original clothing-color answer at 1024.
Pooling is therefore an experimental opt-in candidate, not a validated default.

## Layering the approaches

For repeated image/state context, retain attention KV, convolution and recurrent
states plus prefix hidden states. Clone mutable cache state per question branch,
run the schema suffix, and give the native decision head both hidden-state parts.
No answer cache is used. A cache build costs roughly 0.50 s at 512 and 1.32 s at
1024 after kernel warm-up; the first new cache-state kernel compilation took
7.924 s in an earlier trial. Cache misses can be slower, and repeated requests
amortize the build. A production implementation needs bounded entries and model,
processor and input fingerprints.

For fresh images, 2×2 spatial pooling keeps the high-resolution vision pass but
reduces its language-facing tokens to approximately one-quarter. Update image
placeholders, question/option spans and multimodal position IDs together.
The original native decision head remains. This is untrained spatial pooling,
not the upstream FastV or FasterVLM implementation.

Avoid duplicate preprocessing with either path. The existing API processes images
once for its input-length check and again for inference. The measured extra pass
added approximately 92 ms at high fidelity. Preserve strict input rejection when
reusing the first encoded record.

Cached FlashAttention requires lower-right causal alignment. Initial adapter
attempts failed; PyTorch's CausalBias dispatcher also requires preserving the
public SDPA function identity. The corrected path passed eight changed-question
checks with unchanged choices and at most 0.539 percentage points of probability
difference. At 1024 it measured 0.688 s against a 1.899 s baseline, a 2.76× speedup.
That adds little beyond the simpler 2.74× memory-efficient attention cache path;
baseline language drift in the correction batch was approximately 2.7%.
No public inference-engine replacement was validated here; upstream llama.cpp
still documents Clef image input as unsupported.

## Measurement controls

- Cool the GPU to at most 70 °C before each request; exclude cooling waits.
- Reject thermal-throttled measurements and compare baseline controls.
- Measure CUDA stream time for vision, language and head separately.
- Include JPEG decode, one preprocessing pass, transfer and formatting in whole
  request time; exclude HTTP upload and base64 transport.
- Preserve original images and email data outside source control.
- Treat two photographs and a selected email sample as limited validation.

The benchmark report and raw measurements are separate user-facing artifacts.
This document contains aggregate findings only; no images, email text, personal
identifiers, features or hidden-state tensors are included in this repository.

References: [PyTorch SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html),
[lower-right causal bias](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.bias.causal_lower_right.html),
[Marlin](https://github.com/IST-DASLab/marlin),
[FastV](https://github.com/pkunlp-icler/FastV),
[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

The validated runtime now implements compact schema formatting, bounded exact-prefix
reuse, single-pass preprocessing and optional 2×2 pooling. See
[runtime optimizations](runtime-optimizations.md) for controls and current defaults.
