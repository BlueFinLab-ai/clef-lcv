"""Elapsed device-stream inference windows, not a GPU utilization measurement."""
from contextlib import contextmanager


class GPUTimer:
    def __init__(self, cuda):
        self.cuda = cuda
        self.elapsed_ms = 0.0

    @contextmanager
    def measure(self):
        start = self.cuda.Event(enable_timing=True)
        end = self.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            self.elapsed_ms += start.elapsed_time(end)
