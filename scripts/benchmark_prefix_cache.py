"""Measure alternating reusable contexts with one entry versus automatic VRAM sizing."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('profile', choices=['full','flash'])
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'builds'/args.profile))
app=importlib.import_module('clef_app')
from optimized_inference import InferenceEngine, encode_compact, answers
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
model,processor,_=app.load_model()
questions={'color':{'type':'choice','instructions':'What is the explicitly stated alert color?',
    'criteria':{'red':'Red','blue':'Blue','other':'Another color'}}}
records=[{'state':f'Workflow {i}. '+('Read the supplied evidence and evaluate it carefully. '*160)
    +'The alert color is red.','questions':questions} for i in range(4)]
encoded=[encode_compact(processor,r) for r in records]
results={'profile':args.profile,'gpu':torch.cuda.get_device_name(),'contexts':4,'trials':[]}
baseline=None
for label,budget,capacity in [('one_entry',256,1),('automatic','auto',32)]:
    engine=InferenceEngine(budget,capacity)
    def infer(i):
        e,b=encoded[i]
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            logits,usage=engine.forward(model,processor,e,b,'text')
        result=answers(records[i],e,logits)['color']
        torch.cuda.synchronize()
        return result,usage
    # Fill each context once. The one-entry policy necessarily retains only the last.
    for i in range(4):infer(i)
    rows=[];tick=time.perf_counter()
    for i in list(range(4))*2:
        start=time.perf_counter();answer,usage=infer(i)
        rows.append({'context':i,'ms':(time.perf_counter()-start)*1000,'answer':answer,'usage':usage})
    trial={'mode':label,'wall_s':time.perf_counter()-tick,'rows':rows,'cache':engine.stats(),
        'hits':sum(r['usage']['prefix_cache']=='hit' for r in rows)}
    if baseline is None:baseline=[r['answer']['choice'] for r in rows]
    assert [r['answer']['choice'] for r in rows]==baseline
    results['trials'].append(trial)
    print(label,trial['wall_s'],'s',trial['hits'],'hits',engine.stats(),flush=True)
    engine.clear();del engine;gc.collect();torch.cuda.empty_cache()
args.output.write_text(json.dumps(results,indent=2,allow_nan=False))
print('PASS: all eight choices unchanged',flush=True)
