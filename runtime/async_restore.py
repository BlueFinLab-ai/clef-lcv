"""Bounded CUDA H2D staging with per-layer readiness. No model work on I/O thread."""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
from shared_host_blocks import FrozenTensor


class AsyncRestoreRuntime:
    def __init__(self, staging_mib=128):
        if staging_mib < 2: raise ValueError('Restore staging must be at least 2 MiB')
        self.slot_bytes = int(staging_mib)*2**20//2
        self.buffers = self.stream = self.executor = None
        self.plan = None
        self.hooks = []
        self.restores = 0

    def install(self, language_model):
        if self.hooks: return
        for index, layer in enumerate(language_model.layers):
            def before(module, args, index=index):
                if self.plan is not None:
                    self.plan.wait(index)
            self.hooks.append(layer.register_forward_pre_hook(before))

    def start(self, cache, chunks, device):
        self.finish()
        if not hasattr(cache, 'layers'):
            raise NotImplementedError('Async restore requires a layer-addressable hybrid cache')
        if self.buffers is None:
            self.buffers = [torch.empty(self.slot_bytes,dtype=torch.uint8,pin_memory=True) for _ in range(2)]
            self.stream = torch.cuda.Stream(device=device)
            self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='clef-cache-io')
        plan = RestorePlan(cache,chunks,device,self.stream,self.buffers)
        self.plan = plan
        try:
            plan.future = self.executor.submit(plan.transfer)
        except BaseException:
            self.plan = None
            raise
        self.restores += 1
        plan.wait('hidden');plan.wait('global')
        return plan.cache,plan.chunks

    def finish(self, suppress=False):
        if self.plan is not None:
            plan,self.plan = self.plan,None
            try:plan.future.result()
            except BaseException:
                if not suppress:raise

    def close(self):
        self.finish()
        if self.executor is not None:self.executor.shutdown(wait=True)
        self.executor = None;self.buffers = None

    def stats(self):
        return {'enabled':True,'staging_mib':0 if self.buffers is None else 2*self.slot_bytes/2**20,
                'active':self.plan is not None,'restores':self.restores,
                'strategy':'per_layer_ready_pinned_double_buffer'}


class RestorePlan:
    def __init__(self,cache,chunks,device,stream,buffers):
        self.device,self.stream,self.buffers = device,stream,buffers
        self.groups = {'hidden':[],'global':[]}
        self.memo = {};seen=set()
        def collect(item,group):
            if id(item) in seen:return
            seen.add(id(item))
            if isinstance(item,FrozenTensor):
                dest=torch.empty(item.shape,dtype=item.dtype,device=device)
                dest.record_stream(stream)
                self.memo[id(item)]=dest;self.groups[group].append((item,dest))
            elif isinstance(item,torch.Tensor):
                if item.device.type!='cpu':raise NotImplementedError('Only CPU snapshots can be staged')
                dest=torch.empty(item.shape,dtype=item.dtype,device=device)
                dest.record_stream(stream)
                self.memo[id(item)]=dest;self.groups[group].append((item,dest))
            elif isinstance(item,dict):
                for value in item.values():collect(value,group)
            elif isinstance(item,(list,tuple)):
                for value in item:collect(value,group)
            elif hasattr(item,'__dict__'):
                collect(vars(item),group)
        collect(chunks,'hidden')
        for index,layer in enumerate(cache.layers):
            self.groups[index]=[];collect(layer,index)
        collect(cache,'global')
        del collect  # Break recursive closure -> plan -> GPU destination cycles.
        self.cache,self.chunks=copy.deepcopy((cache,chunks),self.memo)
        self.memo.clear()
        self.ready={key:threading.Event() for key in self.groups}
        self.events={key:torch.cuda.Event() for key in self.groups}
        self.error=None;self.future=None

    @torch.inference_mode()
    def transfer(self):
        slots=[torch.cuda.Event(),torch.cuda.Event()];used=[False,False];ordinal=0
        try:
            with torch.cuda.device(self.device),torch.cuda.stream(self.stream):
                for group,pairs in self.groups.items():
                    for source,dest in pairs:
                        segments=source.segments(dest) if isinstance(source,FrozenTensor) else [(source,dest)]
                        for piece,target in segments:
                            src=piece.contiguous().reshape(-1).view(torch.uint8)
                            dst=target.reshape(-1).view(torch.uint8)
                            for start in range(0,src.numel(),self.buffers[0].numel()):
                                slot=ordinal%2;ordinal+=1
                                if used[slot]:slots[slot].synchronize()
                                count=min(src.numel()-start,self.buffers[slot].numel())
                                self.buffers[slot][:count].copy_(src[start:start+count])
                                dst[start:start+count].copy_(self.buffers[slot][:count],non_blocking=True)
                                slots[slot].record(self.stream);used[slot]=True
                    self.events[group].record(self.stream)
                    self.ready[group].set()
                    # record_stream protects DMA lifetime; avoid pinning every
                    # old KV allocation while later layers append new state.
                    pairs.clear()
                self.stream.synchronize()
        except BaseException as exc:
            self.error=exc
            for ready in self.ready.values():ready.set()
            raise
        finally:
            # The destination memo is embedded in the working cache; CPU source
            # references need not survive completion of the last DMA operation.
            self.groups.clear()

    def wait(self,group):
        self.ready[group].wait()
        if self.error is not None:raise self.error
        torch.cuda.current_stream(self.device).wait_event(self.events[group])
