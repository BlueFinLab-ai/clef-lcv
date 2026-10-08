"""Elapsed non-inference estimate, not CPU utilization or summed core time."""
import math


def cpu_timing(usage, server_total_ms):
    def milliseconds(value):
        return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0

    gpu = usage.get('gpu_time_ms')
    processing = usage.get('latency_ms')
    if not all(milliseconds(value) for value in (gpu, processing, server_total_ms)):
        return {}
    # Inline preparation is already in latency_ms. Lookahead preparation
    # happened before the handler, so include it exactly once.
    lookahead = usage.get('preprocessing_ms', 0) if usage.get('cpu_preparation_overlapped') else 0
    if not milliseconds(lookahead):
        lookahead = 0
    estimate = max(0, processing - gpu) + lookahead
    # A shared batch, rounding or a pre-prepared input cannot attribute more
    # elapsed time than this request's server window outside inference.
    estimate = min(estimate, max(0, server_total_ms - gpu))
    return {'cpu_time_ms': round(estimate, 1), 'cpu_time_estimated': True}
