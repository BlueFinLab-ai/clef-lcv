"""Experimental request-owned CPU storage for active hidden vectors and full KV.

Copies preserve activation precision. Linear/recurrent state stays on the GPU.
No prefix retention or multiple active requests are supported in this mode.
"""
import time
import os
import torch
from transformers.cache_utils import DynamicCache
from host_memory import available_memory


def check_host_capacity(tokens, config, mode, *, head_cpu=False,cached_tokens=0):
    hidden=tokens*config.hidden_size*2*(2 if head_cpu or os.environ.get('CLEF_HEAD_CPU_PREPARE','0')=='1' else 1)
    full=config.layer_types.count('full_attention')
    kv=max(0,tokens-cached_tokens)*full*2*config.num_key_value_heads*config.head_dim*2 if mode in {'kv_hidden','kv_stream'} else 0
    estimate=hidden+kv+(kv//max(1,full))*2
    sample=available_memory()
    if sample is None or estimate>sample['available_bytes']//2:
        raise ValueError('Insufficient measured CPU RAM for active context offloading')
    return estimate


class ActiveKVOffload:
    def __init__(self, language, device):
        self.cache=DynamicCache(config=language.config)
        self.device=device
        self.cpu={};self.hooks=[]
        self.upload_ms=self.offload_ms=0.
        self.upload_bytes=self.offload_bytes=0
        for index,kind in enumerate(language.config.layer_types):
            if kind!='full_attention':continue
            def before(module,args,index=index):self.load(index)
            def after(module,args,output,index=index):self.save(index)
            self.hooks.extend([language.layers[index].register_forward_pre_hook(before),
                               language.layers[index].register_forward_hook(after)])

    def load(self,index):
        if index not in self.cpu:return
        started=time.perf_counter()
        layer=self.cache.layers[index]
        for name,value in self.cpu[index].items():
            setattr(layer,name,value.to(device=self.device,copy=True))
            self.upload_bytes+=value.numel()*value.element_size()
        torch.cuda.current_stream(self.device).synchronize()
        self.upload_ms+=(time.perf_counter()-started)*1000

    def save(self,index):
        started=time.perf_counter()
        layer=self.cache.layers[index]
        previous=self.cpu.get(index,{})
        current={}
        for name in ('keys','values'):
            value=getattr(layer,name)
            if value.ndim!=4:raise RuntimeError('Unsupported active KV layout')
            old=previous.get(name)
            length=0 if old is None else old.shape[-2]
            if value.shape[-2]<length:raise RuntimeError('Active KV history unexpectedly shrank')
            tail=value[:,:,length:,:].to(device='cpu',copy=True,memory_format=torch.contiguous_format)
            current[name]=torch.cat((old,tail),dim=2) if old is not None else tail
            self.offload_bytes+=tail.numel()*tail.element_size()
            setattr(layer,name,current[name])
        self.cpu[index]=current
        self.offload_ms+=(time.perf_counter()-started)*1000

    def stats(self):
        return {'kv_upload_ms':round(self.upload_ms,1),'kv_offload_ms':round(self.offload_ms,1),
                'kv_uploaded_mib':round(self.upload_bytes/2**20,1),'kv_offloaded_mib':round(self.offload_bytes/2**20,1),
                'active_cpu_kv_mib':round(sum(t.numel()*t.element_size() for d in self.cpu.values() for t in d.values())/2**20,1)}

    def close(self):
        for hook in self.hooks:hook.remove()
        self.hooks.clear();self.cpu.clear();self.cache=None
