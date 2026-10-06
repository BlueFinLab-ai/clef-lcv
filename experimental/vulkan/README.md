# Clef LCV direct Vulkan backend assessment

This directory documents a proposed backend, not a working runtime or launch
configuration. No Ollama service is required. See the [RX 580 plan](../../docs/amd-rx580.md).

## Shared application boundary

Keep CPU tokenization, image decoding/preprocessing, schema/span construction,
the HTTP API, portal and FIFO queue. Introduce an execution boundary for model
loading, vision encoding, chunked language prefill, joint decision scoring,
cache snapshots and memory accounting. The current CUDA implementation should
continue to serve existing deployments while Vulkan is developed separately.

Caching policy and prefix keys can remain shared. Tensor ownership, snapshot
cloning, byte accounting, allocation, eviction and synchronization need a
backend-specific implementation. Vulkan buffers are not PyTorch CUDA tensors.
Evicted or branched buffers must not be reused while commands still reference
them. The cache includes recurrent/convolution state, full-attention KV and
retained hidden states for the joint head, not just ordinary transformer KV.

## Operation map from the current code

| Current component | Required Vulkan work | Candidate reuse / unresolved issue |
|---|---|---|
| bitsandbytes NF4 linear layers | Packed four-bit matrix multiplication with block scale decoding | Existing GGML quantized matrix shaders for a converted format; NF4 codebook and double-quant scales need explicit support to reuse our checkpoint directly |
| Compact NF4 input and lexical embeddings | Gather selected rows and dequantize those rows only | Implement or verify compact quantized row lookup; avoid allocating dense vocabulary matrices |
| Language transformer | Norms, projection, activations, rotary positions and full attention | Reuse GGML Vulkan primitives; validate masks, positions and model-specific layouts |
| Gated delta / causal convolution layers | Prefill and recurrent update, persistent state and causal convolution | GGML has an upstream Vulkan GATED_DELTA_NET implementation; verify shapes, state continuation and other required operations against our model |
| Vision encoder and merger | Patch projection, positional encoding, attention, projection and optional spatial pooling | Existing tensor primitives are candidates; build the complete trained graph and preserve image-grid semantics |
| Joint schema head | Span means, normalization, attention/routing, lexical anchors and trained scoring layers | Reproduce the original trained head; a generic Qwen chat model is not an equivalent |
| Chunking and caching | Continuation positions, image features, snapshots, branching and retained hidden state | Preserve logical semantics with Vulkan buffer handles and command completion; current Python tensor code cannot be reused unchanged |
| Capacity and timing | Allocation budget, staging/workspace, fences and timestamp queries | Replace CUDA-specific measurements; distinguish host round-trip, CPU preprocessing and GPU execution |

## Existing compute layers to evaluate

**GGML's Vulkan compute backend** is the leading reuse candidate because it
already has low-bit matrix operations and a
[GATED_DELTA_NET implementation](https://github.com/ggml-org/llama.cpp/pull/20334).
Embed the library beneath our application rather than introducing another
inference server. This still requires Clef graph/weight integration, vision,
decision-head parity and cache state access. Neither complete operation
coverage nor acceptable RX 580 performance has been demonstrated.

**ExecuTorch's Vulkan delegate** provides a PyTorch export/lowering route and
documented [INT4/INT8 linear support](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-quantization.html).
Its [desktop support is experimental](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-overview.html).
Assess hybrid recurrent-state export, dynamic spans/media shapes, compact
embeddings, quantization conversion and unsupported-op fallback before selecting
it. It is not a drop-in execution device for our current Python service.

Avoid committing to a new shader runtime before auditing coverage in these
existing libraries. Neither path is currently a validated Clef Vulkan build.

## Implementation and validation sequence

1. Establish exact operations, tensor layouts and state ownership for Flash.
   Probe RX 580-compatible Vulkan features. Select the execution layer after
   checking compact row lookup, hybrid attention and vision graph coverage.
2. Validate packed matrix and embedding-row operations against CPU/CUDA
   reference outputs. Preserve the trained head and quantization metadata;
   report conversion error separately from backend numerical error.
3. Implement one language block and state continuation, then the whole Flash
   backbone and joint head. Validate all three decision types, chunked versus
   unchunked outputs and cached versus uncached outputs.
4. Implement and validate vision, image-grid positions, multi-image injection
   and optional pooling. Vision is required for the final port; an earlier
   text-only milestone is a development test, not feature-complete support.
5. Integrate backend-specific cache storage, memory pressure handling and API
   capacity discovery. Retain one service per GPU and queue incoming requests.
6. Benchmark Flash on the RX 580, including the email fixture, image processing,
   repeated prefixes and memory-pressure failures. Extend Full to larger AMD
   devices after the backend is validated.

Do not claim support, swap readiness or performance from this assessment.
There are no Vulkan binaries, kernels or converted model assets in this directory.
