"""Opt-in cold-input phase events. No hooks/events in ordinary requests."""
from contextlib import contextmanager
import math


def rate(tokens, milliseconds):
    if type(tokens) is not int or tokens < 1 or type(milliseconds) not in (int,float):return None
    if not math.isfinite(milliseconds) or milliseconds <= 0:return None
    return round(tokens*1000/milliseconds,2)


class ColdPassTiming:
    def __init__(self, model, cuda):
        base=model.language_model.model
        self.modules={'language':base.language_model,'head':model.head}
        if hasattr(base,'visual'):self.modules['vision']=base.visual
        self.cuda=cuda
        self.events=[]
        self.failed=False

    @contextmanager
    def capture(self):
        handles=[];pending={role:[] for role in self.modules}
        def before(role):
            def hook(module,args,kwargs):
                start=self.cuda.Event(enable_timing=True);start.record()
                count=0
                if role=='language':
                    values=kwargs.get('input_ids')
                    if values is None:values=kwargs.get('inputs_embeds')
                    if values is None and args:values=args[0]
                    if values is not None and len(values.shape)>=2:count=int(values.shape[0]*values.shape[1])
                pending[role].append((start,count))
            return hook
        def after(role):
            def hook(module,args,kwargs,output):
                if not pending[role]:self.failed=True;return
                start,count=pending[role].pop()
                end=self.cuda.Event(enable_timing=True);end.record()
                if output is None:self.failed=True
                self.events.append((role,start,end,count))
            return hook
        try:
            for role,module in self.modules.items():
                handles.append(module.register_forward_pre_hook(before(role),with_kwargs=True))
                handles.append(module.register_forward_hook(after(role),with_kwargs=True,always_call=True))
            yield
        except BaseException:
            self.failed=True
            raise
        finally:
            for handle in handles:handle.remove()
            if any(pending.values()):self.failed=True

    def summary(self, usage):
        # The enclosing GPUTimer has already synchronized its ending event.
        # Read event durations without adding phase-boundary synchronization.
        times={role:0. for role in self.modules};calls={role:0 for role in self.modules};tokens=0
        for role,start,end,count in self.events:
            times[role]+=start.elapsed_time(end);calls[role]+=1
            if role=='language':tokens+=count
        input_tokens=usage.get('input_tokens',0)
        hits=sum(usage.get(k,0) or 0 for k in ('reused_prefix_tokens','image_feature_cache_hits','image_preprocess_cache_hits','token_cache_hits'))
        valid=(not self.failed and hits==0 and type(input_tokens) is int and input_tokens>0 and tokens==input_tokens and calls.get('language',0)>0)
        reason=None if valid else 'inference_retry_or_failure' if self.failed else 'cache_reuse_detected' if hits else 'language_token_count_mismatch'
        return {'mode':'cold_input','valid':valid,'invalid_reason':reason,'cache_reuse_disabled':True,
            'model_startup_excluded':True,'kernel_first_use_may_be_included':True,'gpu_batch_size':1,'language_forward_calls':calls.get('language',0),
            'processed_language_tokens':tokens,'input_tokens':input_tokens,
            'language_window_ms':round(times['language'],3),
            'vision_window_ms':round(times['vision'],3) if calls.get('vision') else None,
            'head_window_ms':round(times['head'],3) if calls.get('head') else None,
            'head_timing_complete':calls.get('head',0)>0,
            'language_tokens_per_second':rate(tokens,times['language']) if valid else None,
            'gpu_window_tokens_per_second':rate(tokens,usage.get('gpu_time_ms')) if valid else None,
            'timing_definition':'Device-stream forward windows include launches/transfers; not GPU-busy time. Language excludes vision and decision head.'}
