"""Request-local tiled preparation for Clef's unchanged native joint head.

The full hidden/normalized matrices remain in CPU RAM. All token rows are
normalized and projected on CUDA in bounded chunks. Native head forwarding gets
the projected GPU memory and GPU vectors for its original question/option spans.
No tokens, trained operations, weights, or criteria are removed.
"""
import time
from types import MethodType

import torch
import joint_schema_model as joint


class _Sequence:
    def __init__(self, normalized, memory, device, counters):
        self.normalized, self.memory = normalized, memory
        self.device, self.counters = device, counters

    def __getitem__(self, index):
        values = self.normalized[index]
        self.counters['span_transfer_bytes'] += values.numel()*values.element_size()
        # Let the original head's mean reduction run on CUDA, including spans
        # crossing preparation chunks. CPU reduction could change FP16 rounding.
        return values.to(device=self.device, copy=True)


class _PreparedBatch:
    def __init__(self, normalized, memory, device, counters):
        self.normalized, self.memory = normalized, memory
        self.device, self.counters = device, counters

    def __getitem__(self, index):
        batch, tokens = index
        return _Sequence(self.normalized[batch, tokens], self.memory[batch, tokens],
                         self.device, self.counters)


class _Input:
    def __init__(self, device):
        self.device = device


@torch.inference_mode()
def head_from_cpu(head, hidden_cpu, ids, mask, records, lexical_weight, *, chunk_tokens=4096):
    if not isinstance(head, joint.JointSchemaHead) or head.training:
        raise ValueError('CPU head preparation requires the native Clef inference head')
    if hidden_cpu.device.type != 'cpu' or hidden_cpu.shape[0] != 1 or len(records) != 1:
        raise ValueError('CPU head preparation requires one CPU-resident hidden matrix')
    if ids.device.type != 'cuda' or torch.version.hip or chunk_tokens < 1:
        raise ValueError('CPU head preparation requires CUDA and a positive chunk size')
    if not bool(torch.all(mask == 1)):
        raise ValueError('CPU head preparation currently requires unpadded input')
    device = ids.device
    if head.hidden_norm.weight.device != device or head.memory_projection.weight.device != device:
        raise ValueError('Native head weights must remain on the request GPU')
    started = time.perf_counter()
    normalized_cpu = memory = None
    counters = {'span_transfer_bytes': 0}
    for start in range(0, hidden_cpu.shape[1], chunk_tokens):
        end = min(start+chunk_tokens, hidden_cpu.shape[1])
        current = hidden_cpu[:, start:end].to(device=device, copy=True)
        normalized = head.hidden_norm(current)
        projected = head.memory_projection(normalized)
        if normalized_cpu is None:
            normalized_cpu = torch.empty(hidden_cpu.shape, dtype=normalized.dtype, device='cpu')
            memory = torch.empty((*hidden_cpu.shape[:-1], projected.shape[-1]),
                                 dtype=projected.dtype, device=device)
        memory[:, start:end].copy_(projected)
        normalized_cpu[:, start:end].copy_(normalized)
        del current, normalized, projected
    torch.cuda.current_stream(device).synchronize()
    prepare_ms = (time.perf_counter()-started)*1000
    prepared = _PreparedBatch(normalized_cpu, memory, device, counters)
    payload = _Input(device)
    original_norm = head.hidden_norm.forward
    original_projection = head.memory_projection.forward

    def norm_forward(module, values):
        return prepared if values is payload else original_norm(values)

    def projection_forward(module, values):
        return values.memory if isinstance(values, _Sequence) else original_projection(values)

    # Keep the upstream head.forward byte-for-byte intact, and supply precisely
    # its original normalized spans and projected full-token memory on demand.
    try:
        head.hidden_norm.forward = MethodType(norm_forward, head.hidden_norm)
        head.memory_projection.forward = MethodType(projection_forward, head.memory_projection)
        results = head(payload, ids, mask, records, lexical_weight)
        torch.cuda.current_stream(device).synchronize()
    finally:
        head.hidden_norm.forward = original_norm
        head.memory_projection.forward = original_projection
    usage = {'head_cpu_prepared': True, 'head_prepare_chunk_tokens': chunk_tokens,
             'head_cpu_prepare_ms': round(prepare_ms, 1),
             'head_projected_memory_mib': round(memory.numel()*memory.element_size()/2**20, 1),
             'head_normalized_cpu_mib': round(normalized_cpu.numel()*normalized_cpu.element_size()/2**20, 1),
             'head_span_upload_mib': round(counters['span_transfer_bytes']/2**20, 3)}
    return results, usage
