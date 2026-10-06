"""Immutable CPU KV/hidden segments shared only from an explicitly reused parent.

GPU attention still consumes dense tensors. This does not implement paged GPU
attention or splice unrelated ancestry. Mutable GDN state is always copied.
"""
import copy
import math
import torch


class FrozenTensor:
    def __init__(self,shape,dtype,axis,blocks,element_bytes):
        self.shape,self.dtype,self.axis,self.blocks=tuple(shape),dtype,axis,tuple(blocks)
        self.element_bytes=element_bytes

    @property
    def ndim(self):return len(self.shape)
    @property
    def nbytes(self):return math.prod(self.shape)*self.element_bytes

    def numel(self):return math.prod(self.shape)

    def element_size(self):return self.element_bytes

    def segments(self,dest):
        offset=0
        for block in self.blocks:
            length=block.shape[self.axis]
            target=dest.narrow(self.axis,offset,length)
            # KV heads are separate contiguous sequence planes. Avoid flattening
            # a strided destination into an accidental temporary allocation.
            if self.axis==2:
                for batch in range(block.shape[0]):
                    for head in range(block.shape[1]):yield block[batch,head],target[batch,head]
            else:
                for batch in range(block.shape[0]):yield block[batch],target[batch]
            offset+=length

    def materialize(self,device):
        dest=torch.empty(self.shape,dtype=self.dtype,device=device)
        for source,target in self.segments(dest):target.copy_(source)
        return dest

    def slice(self,start,end):
        offset=0;blocks=[]
        for block in self.blocks:
            length=block.shape[self.axis];left=max(start,offset);right=min(end,offset+length)
            if left<right:blocks.append(block.narrow(self.axis,left-offset,right-left))
            offset+=length
        shape=list(self.shape);shape[self.axis]=end-start
        return FrozenTensor(shape,self.dtype,self.axis,blocks,self.element_bytes)


def storages(value,seen=None,result=None):
    seen=set() if seen is None else seen;result={} if result is None else result
    if id(value) in seen:return result
    seen.add(id(value))
    if isinstance(value,torch.Tensor):
        storage=value.untyped_storage();result[storage._cdata]=storage.nbytes()
    elif isinstance(value,dict):
        for item in value.values():storages(item,seen,result)
    elif isinstance(value,(list,tuple)):
        for item in value:storages(item,seen,result)
    elif hasattr(value,'__dict__'):storages(vars(value),seen,result)
    return result


def _hidden_parent(chunks):
    if not chunks or not all(isinstance(c,FrozenTensor) and c.axis==1 for c in chunks):return None
    first=chunks[0];shape=list(first.shape);shape[1]=sum(c.shape[1] for c in chunks)
    return FrozenTensor(shape,first.dtype,1,[b for c in chunks for b in c.blocks],first.element_bytes)


class SnapshotPlan:
    def __init__(self,entry,parent=None,block_tokens=8192):
        self.source=entry;self.parent=parent;self.block_tokens=block_tokens
        self.specs={};self.reused={};self.allocation_bytes=0
        def tensor(item,axis=None,reuse=None):
            if id(item) in self.specs:return
            if axis is not None and reuse is not None:
                if reuse.axis!=axis or reuse.dtype!=item.dtype or any(a!=b for i,(a,b) in enumerate(zip(reuse.shape,item.shape)) if i!=axis):reuse=None
                elif reuse.shape[axis]>item.shape[axis]:reuse=None
            length=0 if reuse is None else reuse.shape[axis]
            self.specs[id(item)]=(item,axis,reuse,length)
            if reuse is not None:self.reused.update(storages(reuse))
            elements=item.numel() if axis is None else item.numel()//max(1,item.shape[axis])*(item.shape[axis]-length)
            self.allocation_bytes+=elements*item.element_size()
        # Only full-attention keys/values are segment-addressable. GDN's fixed
        # convolution/recurrent tensors remain independent snapshots per child.
        layers=getattr(entry['cache'],'layers',())
        parents=getattr(parent['cache'],'layers',()) if parent is not None else ()
        for i,layer in enumerate(layers):
            for name in ('keys','values'):
                item=getattr(layer,name,None)
                old=getattr(parents[i],name,None) if i<len(parents) else None
                if isinstance(item,torch.Tensor) and item.ndim==4:
                    tensor(item,2,old if isinstance(old,FrozenTensor) else None)
        hidden=_hidden_parent(parent['chunks']) if parent is not None else None
        offset=0
        for chunk in entry['chunks']:
            if isinstance(chunk,torch.Tensor) and chunk.ndim==3:
                overlap=min(chunk.shape[1],max(0,(hidden.shape[1] if hidden is not None else 0)-offset))
                reuse=hidden.slice(offset,offset+overlap) if overlap else None
                tensor(chunk,1,reuse);offset+=chunk.shape[1]
        seen=set()
        def visit(item):
            if id(item) in seen:return
            seen.add(id(item))
            if isinstance(item,torch.Tensor):tensor(item)
            elif isinstance(item,dict):
                for child in item.values():visit(child)
            elif isinstance(item,(list,tuple)):
                for child in item:visit(child)
            elif hasattr(item,'__dict__'):visit(vars(item))
        visit((entry['cache'],entry['chunks']))
        del visit

    def retained_increment(self,live_storages):
        return self.allocation_bytes+sum(size for key,size in self.reused.items() if key not in live_storages)

    def copy(self):
        memo={}
        for ident,(source,axis,reuse,length) in self.specs.items():
            if axis is None:
                memo[ident]=source.detach().to(device='cpu',copy=True,memory_format=torch.contiguous_format)
                continue
            blocks=list(reuse.blocks) if reuse is not None else []
            if length<source.shape[axis]:
                tail=source.narrow(axis,length,source.shape[axis]-length).detach().to(device='cpu',copy=True,memory_format=torch.contiguous_format)
                blocks.extend(tail.narrow(axis,start,min(self.block_tokens,tail.shape[axis]-start)) for start in range(0,tail.shape[axis],self.block_tokens))
            memo[ident]=FrozenTensor(source.shape,source.dtype,axis,blocks,source.element_size())
        return copy.deepcopy((self.source['cache'],self.source['chunks']),memo)
