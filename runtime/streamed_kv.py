"""Experimental exact blockwise full attention over request-owned CPU KV.

CUDA-only, single unpadded text request. Uses PyTorch's efficient SDPA operator
with log-sum-exp output; no compiler or custom GPU binary is required. CPU KV
is preallocated in pinned RAM. Two bounded GPU buffers overlap history uploads
with attention computation. The native recurrent layers and Clef head remain.
"""
import copy
import os
import time
from types import MethodType

import torch
from transformers.cache_utils import DynamicCache


def merge_attention(output, lse, part, part_lse):
    """Combine separately normalized blocks using their global softmax weights."""
    part_lse = part_lse[..., :part.shape[-2]].float()
    if output is None:
        return part.float(), part_lse
    combined = torch.logaddexp(lse, part_lse)
    output.mul_(torch.exp(lse - combined).unsqueeze(-1))
    output.addcmul_(part, torch.exp(part_lse - combined).unsqueeze(-1))
    return output, combined


def efficient_block(query, key, value, scaling, *, causal=False):
    output, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        query, key, value, None, True, 0.0, causal, scale=scaling)
    return output, lse


class StreamedKVOffload:
    def __init__(self, language, device, capacity, *, storage='cpu', prefix=None):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
        self.block_tokens = int(os.environ.get('CLEF_KV_STREAM_BLOCK_TOKENS', '4096'))
        if self.block_tokens < 1:
            raise ValueError('KV streaming block size must be positive')
        if torch.device(device).type != 'cuda' or torch.version.hip:
            raise ValueError('Streamed KV requires NVIDIA CUDA')
        layers = [(i, layer.self_attn) for i, layer in enumerate(language.layers)
                  if language.config.layer_types[i] == 'full_attention']
        if not layers or any(not isinstance(layer, Qwen3_5Attention) for _, layer in layers):
            raise ValueError('Streamed KV requires compatible Qwen3.5 full-attention layers')
        if not getattr(language.config, 'is_causal', True):
            raise ValueError('Streamed KV requires causal text attention')
        if storage not in {'cpu','cuda'}:raise ValueError('Invalid KV storage tier')
        self.storage = storage
        self.language, self.device, self.capacity = language, device, capacity
        self.cache = DynamicCache(config=language.config)
        self.cpu = {}
        self.buffers = None
        self.host_buffers = None
        self.tail_buffers = None
        self.copy_stream = torch.cuda.Stream(device=device)
        self.upload_bytes = self.offload_bytes = self.blocks = 0
        self.offload_ms = 0.0
        self.original_config = language.config
        self.originals = [(layer, layer.forward) for _, layer in layers]
        # CPU cache snapshots are immutable. Only recurrent state is restored to
        # CUDA here; full KV is copied into owned storage by _allocate below.
        self.prefix = prefix
        if prefix is not None:
            from host_prefix_cache import copy_to_device
            full_ids={i for i,_ in layers}
            for i, source in enumerate(prefix['cache'].layers):
                if i not in full_ids:
                    self.cache.layers[i]=copy_to_device(source, device)
            for i,_ in layers:
                source=prefix['cache'].layers[i]
                self._allocate(i, source.keys, source.values)
        # We construct causality inside each SDPA block. Avoid allocating the
        # ordinary Q_length x total_history mask that would defeat bounded VRAM.
        ALL_MASK_ATTENTION_FUNCTIONS.register('clef_streamed_kv', lambda **kwargs: None)
        language.config = copy.copy(language.config)
        language.config._attn_implementation = 'clef_streamed_kv'
        for index, layer in layers:
            def forward(module, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, index=index, **kwargs):
                return self.forward_layer(module, index, hidden_states,
                                          position_embeddings, past_key_values)
            layer.forward = MethodType(forward, layer)

    def _allocate(self, index, key, value):
        if index not in self.cpu:
            # Token-major layout makes a history block contiguous in CPU RAM.
            # Head-major sliced history has gaps between heads, which can force
            # PyTorch to pack a temporary CPU buffer before the CUDA transfer.
            borrowed=key.shape[-2] if self.prefix is not None and self.storage=='cpu' else 0
            shape = (self.capacity-borrowed, *key.shape[:-2], key.shape[-1])
            options={'device':'cpu'} if self.storage=='cpu' else {'device':self.device}
            self.cpu[index] = {'keys': torch.empty(shape, dtype=key.dtype, **options),
                               'values': torch.empty(shape, dtype=value.dtype, **options),
                               'length': 0,'borrowed':borrowed}
            if self.prefix is not None:
                from shared_host_blocks import FrozenTensor
                length=key.shape[-2]
                for name,source in [('keys',key),('values',value)]:
                    if borrowed:
                        self.cpu[index]['prefix_'+name]=source if isinstance(source,FrozenTensor) else FrozenTensor(source.shape,source.dtype,2,(source,),source.element_size())
                        continue
                    offset=0
                    blocks=source.blocks if isinstance(source,FrozenTensor) else (source,)
                    for block in blocks:
                        size=block.shape[-2]
                        self.cpu[index][name][offset:offset+size].copy_(block.permute(2,0,1,3))
                        offset+=size
                    if offset!=length:raise ValueError('Invalid cached KV segments')
                self.cpu[index]['length']=length
                self._publish(index,length,key.dtype)
        if self.buffers is None and self.storage=='cpu':
            shape = (self.block_tokens, *key.shape[:-2], key.shape[-1])
            self.buffers = [(torch.empty(shape, dtype=key.dtype, device=self.device),
                             torch.empty(shape, dtype=value.dtype, device=self.device))
                            for _ in range(2)]
            self.host_buffers=[(torch.empty(shape,dtype=key.dtype,pin_memory=True),
                                torch.empty(shape,dtype=value.dtype,pin_memory=True)) for _ in range(2)]

    def attend(self, module, query, key, value, state):
        from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
        compute = torch.cuda.current_stream(self.device)
        length = state['length']
        count = (length + self.block_tokens - 1) // self.block_tokens
        ready = [torch.cuda.Event(), torch.cuda.Event()]
        consumed = [torch.cuda.Event(), torch.cuda.Event()]
        self.copy_stream.wait_stream(compute)

        def upload(block):
            slot = block % 2
            start = block * self.block_tokens
            end = min(start + self.block_tokens, length)
            if block>=2:ready[slot].synchronize()  # Previous DMA no longer reads this host slot.
            for target,name in zip(self.host_buffers[slot],('keys','values')):
                self._read_history(state,name,start,end,target)
            with torch.cuda.stream(self.copy_stream):
                if block >= 2:
                    self.copy_stream.wait_event(consumed[slot])
                for target,source in zip(self.buffers[slot],self.host_buffers[slot]):
                    target[:end-start].copy_(source[:end-start], non_blocking=True)
                ready[slot].record(self.copy_stream)
            self.upload_bytes += 2 * (end-start) * key.shape[0] * key.shape[1] * key.shape[-1] * key.element_size()

        if self.storage=='cpu':
            for block in range(min(2, count)):upload(block)
        output = lse = None
        for block in range(count):
            slot = block % 2
            size = min(self.block_tokens, length - block*self.block_tokens)
            if self.storage=='cpu':
                compute.wait_event(ready[slot])
                block_key, block_value = self.buffers[slot]
                block_key,block_value=block_key[:size],block_value[:size]
            else:
                start=block*self.block_tokens
                block_key,block_value=state['keys'][start:start+size],state['values'][start:start+size]
            part, part_lse = efficient_block(query,
                repeat_kv(block_key.permute(1, 2, 0, 3), module.num_key_value_groups),
                repeat_kv(block_value.permute(1, 2, 0, 3), module.num_key_value_groups), module.scaling)
            output, lse = merge_attention(output, lse, part, part_lse)
            consumed[slot].record(compute)
            if self.storage=='cpu' and block + 2 < count:
                upload(block + 2)
            self.blocks += 1
        # The current chunk is already in VRAM. Its square causal mask is
        # implicit; all CPU history blocks precede every current query token.
        part, part_lse = efficient_block(query, repeat_kv(key, module.num_key_value_groups),
            repeat_kv(value, module.num_key_value_groups), module.scaling, causal=True)
        output, _ = merge_attention(output, lse, part, part_lse)
        return output.to(query.dtype)

    def forward_layer(self, module, index, hidden_states, position_embeddings, cache):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        if module.training or cache is not self.cache or hidden_states.shape[0] != 1:
            raise ValueError('Streamed KV requires isolated single-request inference')
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, module.head_dim)
        query, gate = torch.chunk(module.q_proj(hidden_states).view(*shape, -1, module.head_dim*2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        query = module.q_norm(query.reshape(head_shape)).transpose(1, 2)
        key = module.k_norm(module.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        value = module.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        self._allocate(index, key, value)
        state = self.cpu[index]
        start, end = state['length'], state['length'] + key.shape[-2]
        if end > self.capacity:
            raise ValueError('Streamed KV capacity exceeded')
        output = self.attend(module, query, key, value, state)
        started = time.perf_counter()
        if self.storage=='cpu':
            length=key.shape[-2]
            if self.tail_buffers is None or self.tail_buffers[0].shape[0]<length:
                tail_shape=(length,*key.shape[:-2],key.shape[-1])
                self.tail_buffers=[torch.empty(tail_shape,dtype=key.dtype,pin_memory=True),torch.empty(tail_shape,dtype=value.dtype,pin_memory=True)]
            for target,source in zip(self.tail_buffers,(key,value)):
                target[:length].copy_(source.permute(2,0,1,3),non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
            for name,source in zip(('keys','values'),self.tail_buffers):
                offset=state.get('borrowed',0)
                state[name][start-offset:end-offset].copy_(source[:length])
        else:
            state['keys'][start:end].copy_(key.permute(2,0,1,3))
            state['values'][start:end].copy_(value.permute(2,0,1,3))
        # CPU history must be ready before the next chunk can upload it.
        if self.storage=='cpu':
            torch.cuda.current_stream(self.device).synchronize()
            self.offload_ms += (time.perf_counter() - started)*1000
            self.offload_bytes += (key.numel()+value.numel())*key.element_size()
        state['length'] = end
        self._publish(index,end,key.dtype)
        output = output.transpose(1, 2).reshape(*shape, -1).contiguous()
        return module.o_proj(output * torch.sigmoid(gate)), None

    def _publish(self,index,end,dtype):
        state=self.cpu[index]
        layer = self.cache.layers[index]
        borrowed=state.get('borrowed',0)
        for name in ('keys','values'):
            tail=state[name][:end-borrowed].permute(1,2,0,3)
            if borrowed:
                from shared_host_blocks import FrozenTensor
                prefix=state['prefix_'+name]
                setattr(layer,name,FrozenTensor((*prefix.shape[:-2],end,prefix.shape[-1]),dtype,2,
                    (*prefix.blocks,tail) if end>borrowed else prefix.blocks,prefix.element_bytes))
            else:setattr(layer,name,tail)
        layer.dtype, layer.device, layer.is_initialized = dtype, torch.device('cpu' if self.storage=='cpu' else self.device), True

    def _read_history(self,state,name,start,end,target):
        borrowed=state.get('borrowed',0)
        if start<borrowed:
            offset=0
            for block in state['prefix_'+name].blocks:
                length=block.shape[-2];left=max(start,offset);right=min(end,offset+length,borrowed)
                if left<right:
                    target[left-start:right-start].copy_(block[:,:,left-offset:right-offset].permute(2,0,1,3))
                offset+=length
        if end>borrowed:
            left=max(start,borrowed)
            target[left-start:end-start].copy_(state[name][left-borrowed:end-borrowed])

    def stats(self):
        used = sum(2 * s['length'] * s['keys'].shape[1] * s['keys'].shape[2] * s['keys'].shape[3] * s['keys'].element_size()
                   for s in self.cpu.values())
        allocated = sum(s[n].numel()*s[n].element_size() for s in self.cpu.values() for n in ('keys', 'values'))
        staging = sum(t.numel()*t.element_size() for pair in self.buffers or [] for t in pair)
        return {'kv_stream_layout': 'token_major', 'kv_stream_block_tokens': self.block_tokens, 'kv_stream_history_blocks': self.blocks,
                'kv_uploaded_mib': round(self.upload_bytes/2**20, 1),
                'kv_offloaded_mib': round(self.offload_bytes/2**20, 1),
                'kv_offload_ms': round(self.offload_ms, 1),
                'kv_storage':self.storage,
                'active_cpu_kv_mib': round(used/2**20, 1) if self.storage=='cpu' else 0.,
                'active_gpu_kv_mib':round(used/2**20,1) if self.storage=='cuda' else 0.,
                'kv_cpu_allocated_mib':round(allocated/2**20,1) if self.storage=='cpu' else 0.,
                'kv_pinned_allocated_mib':round(sum(t.numel()*t.element_size() for pair in self.host_buffers or [] for t in pair)/2**20+
                    sum(t.numel()*t.element_size() for t in self.tail_buffers or [])/2**20,1),
                'kv_gpu_staging_mib': round(staging/2**20, 1)}

    def close(self):
        if self.originals:
            self.copy_stream.synchronize()
            torch.cuda.current_stream(self.device).synchronize()
            for layer, forward in self.originals:
                layer.forward = forward
            self.originals.clear()
            self.language.config = self.original_config
        self.cpu.clear()
        self.buffers = None
        self.host_buffers=self.tail_buffers=None
        self.cache = None
