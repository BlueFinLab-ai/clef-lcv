"""Use the installed rocBLAS GEAM transpose through lossless FP32 conversion."""
import ctypes
import torch


class BlasTranspose:
    def __init__(self):
        self.lib=ctypes.CDLL('librocblas.so')
        self.lib.rocblas_create_handle.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
        self.lib.rocblas_create_handle.restype=ctypes.c_int
        self.lib.rocblas_set_stream.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        self.lib.rocblas_set_stream.restype=ctypes.c_int
        self.lib.rocblas_destroy_handle.argtypes=[ctypes.c_void_p]
        self.lib.rocblas_destroy_handle.restype=ctypes.c_int
        self.lib.rocblas_sgeam.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
        self.lib.rocblas_sgeam.restype=ctypes.c_int
        self.handle=ctypes.c_void_p()
        self.check(self.lib.rocblas_create_handle(ctypes.byref(self.handle)))
        self.alpha=ctypes.c_float(1.0);self.beta=ctypes.c_float(0.0)

    @staticmethod
    def check(status):
        if status:raise RuntimeError(f'rocBLAS GEAM failed: status {status}')

    def __call__(self, weight):
        assert weight.ndim==2 and weight.dtype==torch.float16 and weight.is_contiguous()
        n,k=weight.shape
        source=weight.float()
        out=torch.empty((k,n),device=weight.device,dtype=torch.float32)
        self.check(self.lib.rocblas_set_stream(self.handle,torch.cuda.current_stream(weight.device).cuda_stream))
        # Row-major N×K is column-major K×N. Transpose into column-major N×K,
        # which is exactly the row-major K×N layout consumed by the HIP GEMM.
        self.check(self.lib.rocblas_sgeam(self.handle,112,111,n,k,ctypes.byref(self.alpha),source.data_ptr(),k,
                    ctypes.byref(self.beta),out.data_ptr(),n,out.data_ptr(),n))
        return out.half()

    def close(self):
        if self.handle:
            self.check(self.lib.rocblas_destroy_handle(self.handle))
            self.handle=ctypes.c_void_p()
