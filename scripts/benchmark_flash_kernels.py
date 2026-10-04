"""Compare native FP16 Flash fallback and optimized kernels on one loaded model.

Optional email fixtures remain outside the source project; only answers and
aggregate timings are written. Model weights, encoding, caches and SDPA match.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu', required=True)
p.add_argument('--data-dir', required=True)
p.add_argument('--model-dir', default='model-nf4-compact')
p.add_argument('--email-inputs', type=Path)
p.add_argument('--images-dir', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--max-tokens', type=int, default=24576)
p.add_argument('--cases', help='Comma-separated case names; default is all available cases.')
p.add_argument('--include-native-prefill', action='store_true',
    help='Also compare native chunk prefill with optimized recurrence, convolution and normalization.')
p.add_argument('--include-adaptive-prefill', action='store_true')
p.add_argument('--fla-max-chunk-tokens', type=int, default=512)
p.add_argument('--interleaved-rounds', type=int, default=0,
    help='Alternate modes over this many cold-prefix rounds; no email pass in this mode.')
p.add_argument('--allow-capacity-misses', action='store_true',
    help='Allow >8K checkpoints to be declined by the unchanged memory budget on small GPUs.')
a = p.parse_args()
if a.fla_max_chunk_tokens <= 0:
    p.error('FLA chunk threshold must be positive.')
if a.interleaved_rounds < 0 or (a.interleaved_rounds and a.email_inputs):
    p.error('Interleaved rounds must be nonnegative and cannot include email fixtures.')
a.output.parent.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, CLEF_DATA_DIR=a.data_dir,
    CLEF_MODEL_DIR=a.model_dir, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CLEF_REQUIRE_FAST_LINEAR_ATTENTION='1', CLEF_LINEAR_PREFILL_BACKEND='fla', PYTORCH_ALLOC_CONF='expandable_segments:True',
    OMP_NUM_THREADS='4')
sys.path.insert(0, str(root/'builds/flash'))
import clef_app as app
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from optimized_inference import InferenceEngine, encode_compact, pool_record, remap_checkpoints, answers
from context_budget import ContextBudget
from reusable_inputs import InputCache

model, processor, count = app.load_model()
blocks = [m for m in model.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]
attributes = ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
              'recurrent_gated_delta_rule', 'norm')
fast = [{k: getattr(b, k) for k in attributes} for b in blocks]
fallback = []
for b in blocks:
    norm = qwen.Qwen3_5RMSNormGated(b.head_v_dim, eps=b.layer_norm_epsilon).to(device='cuda', dtype=torch.float16)
    norm.load_state_dict(b.norm.state_dict())
    fallback.append(dict(causal_conv1d_fn=None, causal_conv1d_update=qwen.torch_causal_conv1d_update,
        chunk_gated_delta_rule=qwen.torch_chunk_gated_delta_rule,
        recurrent_gated_delta_rule=qwen.torch_recurrent_gated_delta_rule, norm=norm))
def select(mode):
    for b, attrs, native in zip(blocks, fast if mode != 'fallback' else fallback, fallback):
        for k, value in attrs.items(): setattr(b, k, value)
        if mode == 'native-prefill': b.chunk_gated_delta_rule = native['chunk_gated_delta_rule']
        elif mode == 'adaptive-prefill': b.chunk_gated_delta_rule = app.make_adaptive_chunk_kernel(attrs['chunk_gated_delta_rule'], a.fla_max_chunk_tokens)
modes = ('fallback', 'optimized') + (('native-prefill',) if a.include_native_prefill else ()) + (('adaptive-prefill',) if a.include_adaptive_prefill else ())

result = {'completed':False, 'gpu':torch.cuda.get_device_name(), 'gpu_uuid':a.gpu,
    'compute_dtype':'float16', 'nf4_layers':count, 'linear_layers':len(blocks),
    'scope':'Same loaded compact NF4 Flash model; native fallback versus optimized linear attention, convolution and gated normalization, optionally native or adaptive chunk prefill with optimized recurrence/conv/norm. Encoding and efficient SDPA unchanged. Per-request timings include preparation/forward/head/finish and exclude encoding and HTTP. Email wall totals include encoding and loop overhead; there is no HTTP queue.',
    'trials':[], 'email_trials':[]}
def save(): a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
save()
questions = {'action':{'type':'choice','instructions':'What action is appropriate?',
    'criteria':{'act':'Act immediately','wait':'Wait until later'}},
    'urgent':{'type':'noul','instructions':'Does the final incident require immediate attention?'},
    'impact':{'type':'score','instructions':'Rate the outage impact described in the state.',
    'criteria':['No impact','Partial disruption','Complete outage']}}
def sized(tokens):
    r={'state':'Background archive: ', 'questions':questions}
    suffix='\nCurrent incident: checkout is offline and customers cannot place orders. Act immediately.'
    for _ in range(4):
        r['state']=r['state'].removesuffix(suffix)+suffix
        n=len(encode_compact(processor,r)[0].input_ids)
        if n==tokens:return r
        assert n<tokens
        r['state']=r['state'].removesuffix(suffix)+'word '*(tokens-n)+suffix
    raise AssertionError('Could not size prompt')
circle=Image.new('RGB',(1024,1024),'white');ImageDraw.Draw(circle).ellipse((180,180,844,844),fill='red')
visionq={'color':{'type':'choice','instructions':'What color is the circle?',
    'criteria':{'red':'Red circle','blue':'Blue circle'}}, **questions}
cases=[('short', {'state':'Checkout is offline. Act immediately.','questions':questions},False)]
cases.extend(('text-'+str(n),sized(n),False) for n in (8192,12288,a.max_tokens))
for side in (256,512,1024):
    im=circle.resize((side,side))
    cases.append(('circle-'+str(side),{'state':'Checkout is offline. Inspect the image.','images':[im],
        'questions':visionq,'media_kwargs':{'images_kwargs':{'do_resize':False}}},False))
cases.append(('circle-1024-pooled',cases[-1][1],True))
cases.append(('sixteen-images',{'state':'Inspect all images.','images':[circle.resize((256,256))]*16,
    'questions':{'count':{'type':'choice','instructions':'How many images are supplied?',
        'criteria':{'one':'One','eight':'Eight','sixteen':'Sixteen'}}},
    'media_kwargs':{'images_kwargs':{'do_resize':False}}},False))
if a.images_dir:
    photos=[]
    for name in ('IMG_3350.jpeg','IMG_2666.jpeg'):
        with Image.open(a.images_dir/name) as im:photos.append(ImageOps.exif_transpose(im).convert('RGB'))
    cases.append(('two-original-photos',{'state':'Inspect all supplied photographs.','images':photos,
        'questions':{'count':{'type':'choice','instructions':'How many images are supplied?',
            'criteria':{'one':'One image','two':'Two images','three':'Three images'}}},
        'media_kwargs':{'images_kwargs':{'do_resize':True,'min_pixels':1024,'max_pixels':20000000}}},False))
if a.cases:
    names = a.cases.split(',')
    if len(names) != len(set(names)) or not set(names) <= {c[0] for c in cases}:
        p.error('Unknown or duplicate case names.')
    cases = [c for c in cases if c[0] in names]
budget = ContextBudget(model.language_model.config.text_config, dtype_bytes=2,
    calibrated=True, configured_limit=a.max_tokens, image_limit=a.max_tokens,image_chunked=True)
@torch.inference_mode()
def invoke(engine, record, pooling=False, caching=True, inputs=None):
    e,b,points=encode_compact(processor,record,with_checkpoints=True,input_cache=inputs)
    if pooling:
        original=e;e,b=pool_record(e,b,model.language_model.config.image_token_id)
        points=remap_checkpoints(points,original,e)
    assert len(e.input_ids)<=a.max_tokens
    key=hashlib.sha256(repr([(im.size,hashlib.sha256(im.tobytes()).hexdigest()) for im in record.get('images',[])]).encode()).hexdigest()
    tick=time.perf_counter()
    prepared=engine.prepare_request(budget.workspace(len(e.input_ids)) if len(e.input_ids)>8192 else None)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        logits,usage=engine.forward(model,processor,e,b,key,pooling=pooling,cache_enabled=caching,
            checkpoint_boundaries=points,feature_cache_enabled=caching,
            prefill_chunk_tokens=4096,prefill_initial_chunk_tokens=8192,prefill_images=True)
    answer=answers(record,e,logits);engine.finish_request();torch.cuda.synchronize()
    return {'seconds':time.perf_counter()-tick,'input_tokens':len(e.input_ids),
        'answers':answer,'usage':{**prepared,**usage},'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
def numbers(x,path=''):
    if isinstance(x,dict):
        return {k2:v2 for k,v in x.items() for k2,v2 in numbers(v,path+'/'+k).items()}
    return {path:float(x)} if isinstance(x,(float,int)) else {}
def compare(left,right):
    for key,value in left['answers'].items():
        assert value.get('choice')==right['answers'][key].get('choice'),key
    l,r=numbers(left['answers']),numbers(right['answers'])
    assert l.keys()==r.keys()
    delta=max((abs(l[k]-r[k]) for k in l),default=0)
    assert delta<=.02,delta
    return delta
if a.interleaved_rounds:
    result['interleaved_trials'] = []
    engine = InferenceEngine('auto',reserve_mib=256,prefill_reserve_mib=512,utilization=1.,elastic=True)
    engine.initialize_memory_budget(); inputs = InputCache()
    for name,record,pooling in cases:
        for mode in modes:
            select(mode); engine.clear(); invoke(engine,record,pooling,False,inputs)
        reference = None
        for roundno in range(a.interleaved_rounds):
            for mode in (modes if roundno % 2 == 0 else modes[::-1]):
                select(mode); engine.clear(); row = invoke(engine,record,pooling,False,inputs)
                if reference is None: reference = row
                row.update(case=name,mode=mode,round=roundno,max_numeric_delta=compare(reference,row))
                result['interleaved_trials'].append(row);save()
                print(name,mode,roundno,round(row['seconds'],3),flush=True)
    engine.clear();result['completed']=True;save()
    print('PASS: interleaved cold-prefix comparisons; identical choices',flush=True)
    sys.exit(0)
for mode in modes:
    select(mode)
    engine=InferenceEngine('auto',reserve_mib=256,prefill_reserve_mib=512,utilization=1.,elastic=True)
    engine.initialize_memory_budget();inputs=InputCache()
    for name,record,pooling in cases:
        engine.clear()
        warmup=invoke(engine,record,pooling,False,inputs)
        cold=invoke(engine,record,pooling,False,inputs)
        build=invoke(engine,record,pooling,True,inputs)
        warm=invoke(engine,record,pooling,True,inputs)
        assert warm['usage']['prefix_cache']=='hit' or (a.allow_capacity_misses and warm['input_tokens']>8192
            and warm['usage']['prefix_cache'] in ('miss','not_retained')), (mode,name,warm['usage'])
        row={'mode':mode,'case':name,'cold':cold,'build':build,'warm':warm,
             'cold_warm_max_numeric_delta':compare(cold,warm)}
        if mode!='fallback':
            previous=next(r for r in result['trials'] if r['mode']=='fallback' and r['case']==name)
            row['cross_kernel_max_numeric_delta']=max(compare(previous['cold'],cold),compare(previous['warm'],warm))
        result['trials'].append(row);save()
        print(mode,name,round(cold['seconds'],3),round(warm['seconds'],3),flush=True)
    engine.clear();gc.collect();torch.cuda.empty_cache()
    if a.email_inputs:
        data=json.loads(a.email_inputs.read_text());assert len(data)==100
        engine=InferenceEngine('auto',reserve_mib=256,prefill_reserve_mib=512,utilization=1.,elastic=True)
        engine.initialize_memory_budget();inputs=InputCache()
        invoke(engine,data[0]['record'],caching=True,inputs=inputs)
        engine.clear()
        for phase in ('first','repeat'):
            rows=[];start=time.perf_counter()
            for item in data:
                r=invoke(engine,item['record'],caching=True,inputs=inputs)
                r.update(n=item['n'],reference=item['reference']);rows.append(r)
                if item['n']%20==0:print(mode,'emails',phase,item['n'],flush=True)
            trial={'mode':mode,'phase':phase,'wall_seconds':time.perf_counter()-start,'rows':rows}
            if mode!='fallback':
                previous=next(t for t in result['email_trials'] if t['mode']=='fallback' and t['phase']==phase)
                trial['max_numeric_delta']=max(compare(l,r) for l,r in zip(previous['rows'],rows))
            result['email_trials'].append(trial);save()
        engine.clear();gc.collect();torch.cuda.empty_cache()
result['completed']=True;save();print('PASS: FP16 text, images, pooling, long/chunked prefixes, cached answers and optional frozen emails',flush=True)
