#!/usr/bin/env python3
"""Cold-input GPU comparison over the standard API. No cache flush or restart.

Uses only the standard library. Prompts, images, answers and credentials are not
saved; output contains workload hashes, server metadata and measured usage.
"""
import argparse,copy,hashlib,http.client,json,math,os,ssl,statistics,time
from pathlib import Path
from urllib.parse import urlsplit


def synthetic_request(words):
    sentence='The synthetic warehouse record says the shipment is complete and all items were delivered safely.'
    tokens=sentence.split();text=' '.join(tokens[i%len(tokens)] for i in range(words))
    return {'state':text,'questions':{'complete':{'type':'noul','instructions':'Does the record say the shipment is complete?'}},'images':[]}


def load_requests(path):
    value=json.loads(Path(path).read_text())
    if isinstance(value,dict) and 'rows' in value:value=[r.get('request',r) for r in value['rows']]
    elif isinstance(value,list):value=[r.get('request',r) for r in value]
    else:value=[value]
    if not value or any(not isinstance(r,dict) or not r.get('questions') or 'state' not in r for r in value):
        raise ValueError('Expected a decision request, request array, or fixture with rows[].request')
    return value


class API:
    def __init__(self,url,key='',timeout=600):
        parsed=urlsplit(url.rstrip('/'))
        if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use an HTTP(S) base URL without credentials/query/fragment')
        self.url=url.rstrip('/');self.prefix=parsed.path.rstrip('/')
        if self.prefix.endswith('/v1'):self.prefix=self.prefix[:-3]
        cls=http.client.HTTPSConnection if parsed.scheme=='https' else http.client.HTTPConnection
        extra={'context':ssl.create_default_context()} if parsed.scheme=='https' else {}
        self.connection=cls(parsed.hostname,parsed.port,timeout=timeout,**extra)
        self.key=key
    def request(self,path,body=None):
        headers={'Content-Type':'application/json'}
        if self.key:headers['Authorization']='Bearer '+self.key
        if body is not None:headers['X-Clef-Benchmark']='cold'
        data=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode() if body is not None else None
        self.connection.request('POST' if body is not None else 'GET',self.prefix+path,body=data,headers=headers)
        response=self.connection.getresponse();raw=response.read()
        value=json.loads(raw)
        if response.status>=400:
            # Only report status/limits; a server's error detail may echo input.
            raise RuntimeError(f'HTTP {response.status}; input_tokens={value.get("input_tokens")}, max_input_tokens={value.get("max_input_tokens")}')
        return value
    def close(self):self.connection.close()


def validate(usage):
    b=usage.get('cold_benchmark') or {}
    if b.get('mode')!='cold_input' or b.get('valid') is not True:
        raise ValueError('Missing/invalid cold-pass timing; update service or inspect invalid_reason')
    if usage.get('batch_size',1)!=1 or usage.get('reused_prefix_tokens',0)!=0 or b.get('processed_language_tokens')!=usage.get('input_tokens'):
        raise ValueError('Cold benchmark reused tokens or has incorrect token/batch accounting')
    for k in ('language_window_ms','language_tokens_per_second'):
        if type(b.get(k)) not in (int,float) or not math.isfinite(b[k]) or b[k]<=0:raise ValueError('Invalid phase measurement')
    return b


def aggregate(samples):
    total_tokens=sum(r['input_tokens'] for r in samples)
    language_ms=sum(r['language_window_ms'] for r in samples)
    client_ms=sum(r['client_ms'] for r in samples)
    gpu_ms=sum(r['gpu_time_ms'] for r in samples)
    return {'passes':len(samples),'total_input_tokens':total_tokens,'total_language_window_ms':round(language_ms,3),
        'total_client_ms':round(client_ms,3),'language_tokens_per_second':round(total_tokens*1000/language_ms,2),
        'gpu_window_tokens_per_second':round(total_tokens*1000/gpu_ms,2),
        'end_to_end_tokens_per_second':round(total_tokens*1000/client_ms,2),
        'language_tps_min':min(r['language_tokens_per_second'] for r in samples),
        'language_tps_max':max(r['language_tokens_per_second'] for r in samples),
        'client_p50_ms':statistics.median(r['client_ms'] for r in samples),'peak_allocated_mib':max(r['peak_allocated_mib'] for r in samples),
        'total_queue_wait_ms':round(sum(r['queue_wait_ms'] for r in samples),3)}


def run(args):
    key=Path(args.api_key_file).read_text().strip() if args.api_key_file else os.environ.get('CLEF_API_KEY','').strip()
    cases=[{'label':'request-file','requests':load_requests(args.request_file)}] if args.request_file else [
        {'label':f'{n}-words','requests':[synthetic_request(n)]} for n in args.words]
    report={'schema_version':1,'mode':'cold_input','model_startup_excluded':True,'cache_flags':{'prefix_cache':False,'input_cache':False},
        'untimed_warmup_repeats':args.warmup,'timed_repeats':args.repeats,'concurrency':1,
        'timing_definition':'Language-model device-stream forward windows include launches/transfers; not GPU-busy time. End-to-end includes queue/network/preprocessing/head/vision.',
        'servers':[]}
    for url in args.url:
        api=API(url,key,args.timeout);server={'url':url,'cases':[]};report['servers'].append(server)
        try:
            health=api.request('/health');models=api.request('/v1/models');model=args.model or models['data'][0]['id']
            server.update(model=model,gpu=health.get('gpu'),dtype=health.get('compute_dtype'),quantization=health.get('quantization'),
                revision=health.get('source_revision'),runtime_backend=health.get('startup_strategy',{}).get('runtime_backend'))
            for case in cases:
                bodies=[]
                for original in case['requests']:
                    body=copy.deepcopy(original);body.update(model=model,prefix_cache=False,input_cache=False);bodies.append(body)
                digest=hashlib.sha256(json.dumps(bodies,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                # Warm kernels/shape setup while keeping every input-cache layer off.
                warmup_retries=0
                for _ in range(args.warmup):
                    for body in bodies:
                        u=api.request('/v1/systemone',body)['usage'];b=u.get('cold_benchmark') or {}
                        if b.get('mode')=='cold_input' and b.get('invalid_reason')=='inference_retry_or_failure' and not u.get('reused_prefix_tokens',0):
                            warmup_retries+=1  # Untimed routing/shape calibration; never accept a timed retry.
                        else:validate(u)
                samples=[]
                for repeat in range(args.repeats):
                    for index,body in enumerate(bodies):
                        start=time.perf_counter();value=api.request('/v1/systemone',body);client_ms=(time.perf_counter()-start)*1000
                        u=value['usage'];b=validate(u)
                        samples.append({'repeat':repeat+1,'request_index':index+1,'input_tokens':u['input_tokens'],'client_ms':round(client_ms,3),
                            'queue_wait_ms':u.get('queue_wait_ms',0),'gpu_time_ms':u['gpu_time_ms'],
                            'language_window_ms':b['language_window_ms'],'language_tokens_per_second':b['language_tokens_per_second'],
                            'vision_window_ms':b.get('vision_window_ms'),'head_window_ms':b.get('head_window_ms'),
                            'language_forward_calls':b['language_forward_calls'],'peak_allocated_mib':u.get('peak_allocated_mib',0),
                            'execution_strategy':u.get('execution_strategy',u.get('prefill_mode'))})
                result={'label':case['label'],'payload_sha256':digest,'request_count':len(bodies),
                    'images_present':any(body.get('images') for body in bodies),'untimed_warmup_retry_requests':warmup_retries,'summary':aggregate(samples),'samples':samples}
                server['cases'].append(result)
                print(json.dumps({'gpu':server['gpu'],'model':model,'case':case['label'],**result['summary']}),flush=True)
        except Exception as exc:
            server['error']=str(exc)
            print(json.dumps({'url':url,'error':str(exc)}),flush=True)
        finally:
            api.close();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    return 1 if any('error' in s for s in report['servers']) else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',action='append',required=True,help='Service base URL; repeat to compare GPUs')
    p.add_argument('--model',help='Override each service model discovery')
    p.add_argument('--request-file',type=Path,help='One request, array, or saved rows[].request fixture')
    p.add_argument('--words',type=int,nargs='+',default=[1024,4096],help='Synthetic text sizes in words; report uses actual server token counts')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--api-key-file',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.repeats<=20 or not 0<=args.warmup<=5 or any(n<1 or n>131072 for n in args.words) or args.timeout<=0:
        p.error('Use 1–20 repeats, 0–5 warmups, 1–131072 words and a positive timeout')
    return run(args)
if __name__=='__main__':raise SystemExit(main())
