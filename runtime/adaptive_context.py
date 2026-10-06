"""Memory-based per-request CUDA routing. No fixed token streaming cutoff.

Calibration is keyed by GPU/model/kernel fingerprint, uses cold request peaks,
and reserves headroom. Cached prefixes do not reduce total-context accounting.
"""
import hashlib
import json
import math
import os
from pathlib import Path

MIB=2**20


class AdaptiveContext:
    def __init__(self, config, head_width, *, native_budget, fingerprint,
                 reserve_mib=256, calibration_path=None, stream_enabled=True):
        self.config=config;self.native=native_budget;self.width=head_width
        self.reserve=reserve_mib*MIB;self.quantum=4096
        self.architecture=config.max_position_embeddings
        self.kv=config.layer_types.count('full_attention')*2*config.num_key_value_heads*config.head_dim*2
        self.hidden=config.hidden_size*2
        self.projected=head_width*2
        self.floor=max(512*MIB, native_budget.reference_workspace-native_budget.reference_tokens*self.kv+64*MIB)
        self.samples={};self.oom={};self.stream_enabled=stream_enabled
        self.fingerprint=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
        self.path=Path(calibration_path) if calibration_path else None
        if self.path and self.path.exists():
            try:
                saved=json.loads(self.path.read_text())
                if saved.get('fingerprint')==self.fingerprint:
                    self.samples=saved['samples'];self.oom=saved['oom']
            except (ValueError,KeyError,OSError):pass
        for sample in self.samples.values():sample['floor']=math.ceil(sample['observed_floor'])+16*MIB

    def persist(self):
        if not self.path:return
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'fingerprint':self.fingerprint,'samples':self.samples,'oom':self.oom}))
            os.replace(temporary,self.path)
        except OSError:pass

    def workspace(self, mode, tokens, schema_bytes=0):
        # Native head cross-attention retains several projected-memory arrays.
        head=256*MIB+6*tokens*self.projected+schema_bytes
        if mode=='none':
            return max(self.native.workspace(tokens),head+2*tokens*self.hidden)
        learned=self.samples.get(mode)
        floor=int(learned['floor']) if learned else self.floor
        if mode=='kv_gpu':return max(floor+tokens*self.kv,head)
        if mode=='kv_stream':return max(floor,head)
        raise ValueError('Invalid routing mode')

    def oom_allows(self,mode,tokens,available):
        old=self.oom.get(mode)
        if old is None:return True
        if isinstance(old,int):return tokens<old
        # An OOM during contention must not permanently cap a GPU after memory
        # becomes available again. Near the failed budget, keep the backoff.
        return tokens<old['tokens'] or available>old['available']*1.10

    def limits(self, available, host_available, *, cached_tokens=0):
        usable=max(0,available-self.reserve)
        result={}
        for mode in ('none','kv_gpu','kv_stream'):
            low,high=0,self.architecture//self.quantum
            while low<high:
                middle=(low+high+1)//2
                tokens=middle*self.quantum
                fits=self.workspace(mode,tokens)<=usable and self.oom_allows(mode,tokens,available) and (mode=='none' or self.stream_enabled)
                if mode=='kv_stream':
                    fresh=max(0,tokens-cached_tokens)
                    host=fresh*self.kv+2*tokens*self.hidden+2*fresh*self.kv//max(1,self.config.layer_types.count('full_attention'))
                    fits=fits and self.stream_enabled and host_available is not None and host<=host_available//2
                if fits:low=middle
                else:high=middle-1
            result[mode]=low*self.quantum
        return result

    def choose(self,tokens,available,host_available,*,schema_bytes=0,minimum_mode='none',cached_tokens=0):
        modes=['none','kv_gpu','kv_stream'];modes=modes[modes.index(minimum_mode):]
        for mode in modes:
            if mode!='none' and not self.stream_enabled:continue
            if not self.oom_allows(mode,tokens,available):continue
            if self.workspace(mode,tokens,schema_bytes)>max(0,available-self.reserve):continue
            if mode=='kv_stream':
                if not self.stream_enabled or host_available is None:continue
                fresh=max(0,tokens-cached_tokens)
                host=fresh*self.kv+2*tokens*self.hidden+2*fresh*self.kv//max(1,self.config.layer_types.count('full_attention'))
                if host>host_available//2:continue
            return {'mode':mode,'workspace_bytes':self.workspace(mode,tokens,schema_bytes),
                    'head_cpu':mode!='none','reason':'resident_fits' if mode=='none' else 'bounded_resident_fits' if mode=='kv_gpu' else 'resident_memory_exceeded'}
        return None

    def observe(self,mode,tokens,workspace,*,reused_tokens=0):
        if reused_tokens or tokens<8192:return  # Warm suffix work must not lower the cold floor.
        if mode=='none':
            self.native.observe(tokens,workspace);return
        variable=tokens*self.kv if mode=='kv_gpu' else 0
        measured=max(256*MIB,workspace-variable)
        # Head-dominated streaming samples do not measure backbone's fixed floor.
        if mode=='kv_stream' and workspace>=256*MIB+6*tokens*self.projected: return
        old=self.samples.get(mode)
        observed=max(measured,old['observed_floor'] if old else 0)
        self.samples[mode]={'observed_floor':observed,'floor':math.ceil(observed)+16*MIB,
                            'max_cold_tokens':max(tokens,old['max_cold_tokens'] if old else 0),
                            'samples':(old['samples'] if old else 0)+1}
        self.persist()

    def record_oom(self,mode,tokens,available):
        self.oom[mode]={'tokens':tokens,'available':available};self.persist()

    def stats(self):
        return {'kind':'memory_adaptive','reserve_mib':self.reserve/MIB,
                'fingerprint':self.fingerprint,'cold_calibration':self.samples,
                'mode_oom_tokens':self.oom,'stream_kernel_available':self.stream_enabled}
