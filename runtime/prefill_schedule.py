"""Choose new-token blocks from live reusable VRAM, not cached-prefix length.

This is a conservative workspace estimate, not an allocation guarantee. The
caller preserves context/image/schema checkpoints and owns the mutable state.
CPU-streamed attention history uses bounded blocks; it is not counted as VRAM.
"""
import math

MIB = 2**20


class WorkspaceHeadroom:
    """Refresh driver headroom at coarse boundaries; track own live allocations.

    CUDA memory queries between tiny state/schema forwards can drain the launch
    pipeline. PyTorch allocation counters are CPU metadata and update our usable
    budget between driver samples. External allocations still require refreshes.
    """
    def __init__(self, device, *, refresh_tokens=32768):
        self.device, self.refresh_tokens = device, refresh_tokens
        self.capacity = None
        self.cursor = 0
        self.samples = 0

    def available(self, cursor):
        import torch
        if self.capacity is None or cursor-self.cursor >= self.refresh_tokens:
            self.capacity = torch.cuda.mem_get_info(self.device)[0]+torch.cuda.memory_reserved(self.device)
            self.cursor = cursor
            self.samples += 1
        return max(0,self.capacity-torch.cuda.memory_allocated(self.device))


class PrefillSchedule:
    def __init__(self, config, *, dtype_bytes, maximum=16384, target=4096, initial_target=8192, reserve_mib=512,
                 mode='none', implicit=False, quantum=256):
        if min(maximum,target,initial_target)<1 or reserve_mib < 0 or quantum < 1 or dtype_bytes < 1:
            raise ValueError('Invalid adaptive prefill budget')
        self.maximum, self.reserve, self.quantum = maximum, reserve_mib*MIB, quantum
        self.target = min(target,maximum)
        self.initial_target = min(initial_target,maximum)
        self.mode, self.implicit = mode, implicit
        self.hidden = config.hidden_size*dtype_bytes
        self.kv = config.layer_types.count('full_attention')*2*config.num_key_value_heads*config.head_dim*dtype_bytes
        self.expanded = 2*config.num_attention_heads*config.head_dim*dtype_bytes
        self.concat = 2*config.num_key_value_heads*config.head_dim*dtype_bytes
        # FFN intermediates, Q/K/V/gates, native linear-attention scratch, and
        # normalization outputs. Learned model parameters are already allocated.
        self.activations = dtype_bytes*(2*config.intermediate_size + 12*config.hidden_size
            + 8*config.num_attention_heads*config.head_dim)
        self.mask_element = dtype_bytes
        self.peak_projected = self.memory_limited = 0

    def workspace(self, query, history, *, hidden_allocation=0):
        persistent = query*self.kv if self.mode in {'none','kv_gpu'} else 0
        if self.mode in {'kv_gpu','kv_stream'}:
            # Blockwise attention's FP32 output and accumulator plus current KV.
            attention = query*(self.expanded*2+self.concat)
        else:
            attention = (history+query)*(self.expanded+self.concat)
            if not self.implicit and history:
                attention += self.mask_element*query*(history+query)
        return math.ceil(1.25*(query*self.activations+persistent+attention+hidden_allocation)) + 64*MIB

    def choose(self, cursor, stop, available_bytes, *, hidden_allocation=0, initial=False):
        desired = min(self.initial_target if initial else self.target, stop-cursor)
        if desired < 1:
            raise ValueError('Prefill must advance')
        usable = max(0, available_bytes-self.reserve)
        low, high = 0, (desired+self.quantum-1)//self.quantum
        while low < high:
            middle = (low+high+1)//2
            query = min(desired, middle*self.quantum)
            if self.workspace(query,cursor,hidden_allocation=hidden_allocation) <= usable:
                low = middle
            else:
                high = middle-1
        if not low:
            # Do not try a large allocation and corrupt a partly updated cache.
            raise MemoryError('Insufficient prefill workspace for one bounded chunk')
        chosen = min(desired,low*self.quantum)
        self.memory_limited += chosen < desired
        self.peak_projected = max(self.peak_projected,self.workspace(chosen,cursor,hidden_allocation=hidden_allocation))
        return chosen
