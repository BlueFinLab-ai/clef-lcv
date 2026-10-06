"""Experimental bridge to Schaka's unchanged gfx803 FP16 prefill GEMM.

No persistent FP16 weight copies: an NF4 weight is dequantized and transposed
per invocation. This makes the measured overhead part of the result.
"""
import ctypes
import math
import types
from pathlib import Path
import torch
import bitsandbytes as bnb


class PrefillGemm:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(Path(path).resolve()))
        self.lib.gfx803_gemm_launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        self.lib.gfx803_gemm_launch.restype = None
        self.hip = ctypes.CDLL('libamdhip64.so')
        self.hip.hipGetLastError.argtypes = []
        self.hip.hipGetLastError.restype = ctypes.c_int
        self.calls = 0
        self.enabled = False
        self.originals = []

    def gemm(self, x, transposed):
        assert x.ndim == 2 and transposed.ndim == 2
        assert x.dtype == transposed.dtype == torch.float16
        assert x.is_contiguous() and transposed.is_contiguous()
        m, k = x.shape
        assert transposed.shape[0] == k
        n = transposed.shape[1]
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        self.lib.gfx803_gemm_launch(x.data_ptr(), transposed.data_ptr(), out.data_ptr(),
                                   m, n, k, torch.cuda.current_stream(x.device).cuda_stream)
        error = self.hip.hipGetLastError()
        if error:
            raise RuntimeError(f'gfx803 GEMM launch failed: HIP error {error}')
        self.calls += 1
        return out

    def nf4(self, layer, x):
        weight = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)
        assert weight.shape == (layer.out_features, layer.in_features)
        transposed = weight.to(torch.float16).t().contiguous()
        del weight
        out = self.gemm(x.reshape(-1, x.shape[-1]).contiguous(), transposed)
        if layer.bias is not None:
            out = out + layer.bias
        return out.reshape(*x.shape[:-1], layer.out_features)

    def install(self, model):
        names = []
        for name, layer in model.named_modules():
            if not isinstance(layer, bnb.nn.Linear4bit) or layer.out_features < 8192:
                continue
            if 'language_model' not in name:
                continue
            original = layer.forward
            def forward(module, x, original=original):
                if (self.enabled and not module.training and x.device.type == 'cuda'
                        and x.dtype == torch.float16 and math.prod(x.shape[:-1]) >= 128):
                    return self.nf4(module, x)
                return original(x)
            layer.forward = types.MethodType(forward, layer)
            self.originals.append((layer, original))
            names.append({'name':name,'in_features':layer.in_features,'out_features':layer.out_features})
        return names

    def restore(self):
        for layer, original in self.originals:
            layer.forward = original
        self.originals.clear()
