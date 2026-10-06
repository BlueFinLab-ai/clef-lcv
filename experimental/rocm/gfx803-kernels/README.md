# Existing gfx803 kernel experiments

These kernels are research artifacts, not enabled service backends. The RX 580
service keeps its validated native PyTorch attention and saved rocBLAS choices.
The default NVIDIA build is unchanged.

`gfx803_gemm_lib.hip` and `gfx803_gemv_m.hip` are unchanged copies from
[Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803), revision
`685b2cd0e1a8d48497d694b9dc5a1130fd059f89`, under
`vllm/vllm/gfx803_kernels/`. The upstream vLLM Apache-2.0 license is included.

| File | SHA-256 |
|---|---|
| gfx803_gemm_lib.hip | 82209e145cc5a1a1bfbab901ce33c02c3f180cee54c2dd5f9ddb7eea95c1dc5c |
| gfx803_gemv_m.hip | 438d1120f589782fec54ec28ff0ca9a617f39c2046e65541af76df6949fd3934 |

The GEMM expects FP16 K×N weights. `adapter.py` tests NF4 dequantization and
per-call transposition, with no persistent dense-weight cache. `transpose.py`
tests the installed rocBLAS FP32 GEAM transpose with a lossless FP16 round trip.
Both paths lost to the deployed NF4/rocBLAS path on complete requests, despite
the faster raw GEMM. They are preserved to reproduce that result, not as
recommended startup options.

The native-layout GEMV-M accepts N×K weights and 2–16 activation rows.
`benchmark_native_layout.py` tests the real first MLP projection from the
compact checkpoint, including NF4 dequantization. It also tests larger prefills
by tiling activation rows into groups of 16. Eight-row operation improved,
but 128–1024-row prefills were slower. This is an operator benchmark, not an
end-to-end service improvement.

Both sources compile with the installed community stack:

```sh
hipcc --offload-arch=gfx803 -O3 -shared -fPIC -o libgfx803gemm.so gfx803_gemm_lib.hip
hipcc --offload-arch=gfx803 -O3 -shared -fPIC -o libgfx803gemv_m.so gfx803_gemv_m.hip
```

Run the native-layout benchmark in the pinned experimental ROCm image, with
the selected GPU exposed, the compact checkpoint mounted at `/checkpoint`,
this directory mounted writable at `/bench`, and the existing tuning CSV at
`/app/tuning/gfx803.csv`. The pinned community image retains its compiled
GEMV-M library. To use a freshly built library, set
`CLEF_GFX803_GEMV_LIBRARY=/bench/libgfx803gemv_m.so`.

The command inside the container is
`python /bench/benchmark_native_layout.py`. It writes
`/bench/gfx803-native-layout-results.json`. Stop the inference service during
timing comparisons so only the benchmark uses the selected GPU. No private
photos or email bodies are included here.

[Raw GEMM with PyTorch layout measurements](../gfx803-torch-layout-results-2026-10-04.json),
[GEAM layout and large-input cache checks](../gfx803-geam-results-2026-10-04.json),
[native-layout GEMV-M measurements](../gfx803-native-layout-results-2026-10-04.json).
