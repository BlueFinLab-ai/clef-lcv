"""Validate concurrent mixed requests, queued 413 and TCP disconnect on a live GPU.

Fixtures are synthetic text and two generated JPEGs. No request payloads are saved.
"""
import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from PIL import Image
from transformers import AutoProcessor
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', required=True)
p.add_argument('--project', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(a.project/'runtime'), str(a.project/'vendor/cloudflare')]
from optimized_inference import encode_compact
from reusable_inputs import InputCache
processor = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
fixtures = InputCache()


def call(path, body=None):
    started = time.perf_counter()
    req = urllib.request.Request(a.url+path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=360) as response:
            return {'status':response.status, 'body':json.load(response),
                    'headers':{k.lower():v for k,v in response.headers.items()}, 'wall_seconds':time.perf_counter()-started}
    except urllib.error.HTTPError as exc:
        return {'status':exc.code,'body':json.load(exc),'headers':{k.lower():v for k,v in exc.headers.items()},
                'wall_seconds':time.perf_counter()-started}


async def http(path, body=None):
    return await asyncio.to_thread(call, path, body)


async def until(predicate, timeout=10):
    deadline = time.monotonic()+timeout
    while True:
        h = (await http('/health'))['body']
        assert h.get('queue',{}).get('gpu_workers') == 1
        if predicate(h): return h
        assert time.monotonic()<deadline, 'Expected queue state not observed'
        await asyncio.sleep(.025)


def sized(record, tokens):
    record = dict(record)
    for _ in range(4):
        n = len(encode_compact(processor,record,input_cache=fixtures)[0].input_ids)
        if n==tokens:return record
        assert n<tokens
        record['state'] += 'word '*(tokens-n)
    raise AssertionError('Token fixture size mismatch')


def text(state):
    return {'state':state,'questions':{'action':{'type':'choice','instructions':'What action is appropriate?',
             'criteria':{'act':'Act immediately','wait':'Wait until later'}}}}


def comparable(reference, row):
    assert row['status'] == reference['status'] == 200
    for key, answer in reference['body']['answers'].items():
        other = row['body']['answers'][key]
        assert answer.get('choice') == other.get('choice')
        if 'probabilities' in answer:
            assert max(abs(v-other['probabilities'][k]) for k,v in answer['probabilities'].items()) <= .002
    assert row['body']['usage']['prefix_cache'] != 'memory_fallback'


async def main():
    deadline=time.monotonic()+240
    while True:
        try:
            before=(await http('/health'))['body']
            if before.get('queue',{}).get('enabled'):break
        except OSError:pass
        assert time.monotonic()<deadline, 'Queue deployment not ready'
        await asyncio.sleep(2)
    metadata=(await http('/v1/models'))['body']['data'][0]
    model=metadata['id'];limit=metadata['max_input_tokens']
    def wire(body):return dict(body,model=model)
    red,blue=Image.new('RGB',(512,512),'red'),Image.new('RGB',(512,512),'blue')
    def image_url(im):
        b=io.BytesIO();im.save(b,format='JPEG');return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
    photo={'state':'Inspect the supplied images.','images':[image_url(red),image_url(blue)],
           'media_kwargs':{'images_kwargs':{'do_resize':True,'min_pixels':1024,'max_pixels':20000000}},
           'questions':{'count':{'type':'choice','instructions':'How many images are supplied?',
                                'criteria':{'one':'One image','two':'Two images','three':'Three images'}}}}
    urgent=text('Checkout is down. This is urgent; act immediately.')
    normal=text('No incident is occurring. A routine update is planned for next week; wait until later.')
    bodies=[urgent,normal,photo,{**photo,'image_pooling':True}]
    refs=[await http('/v1/systemone',wire(b)) for b in bodies]
    assert all(r['status']==200 for r in refs)
    assert refs[0]['body']['answers']['action']['choice']=='act'
    assert refs[1]['body']['answers']['action']['choice']=='wait'
    assert refs[2]['body']['answers']['count']['choice']=='two'
    result={'completed':False,'url':a.url,'before':before,'initial_models':metadata,'serial_references':refs}
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    blocker=wire(sized(text(f'Queue pressure {time.time_ns()}. Checkout is down. Act immediately. '),8192))
    large=wire(sized(text('Over-limit queue case. Checkout is down. Act immediately. '),limit+1))
    active=asyncio.create_task(http('/v1/systemone',blocker))
    await until(lambda h:h['queue']['active_requests']==1)
    jobs=[]
    for i,b in enumerate([*bodies,large,urgent]):
        jobs.append(asyncio.create_task(http('/v1/systemone',wire(b))))
        await until(lambda h:h['queue']['waiting_requests']>=i+1)
    busy=await until(lambda h:h['queue']['waiting_requests']>=6)
    assert busy['context_snapshot_status']=='last_idle_busy'
    assert busy['max_input_tokens']==limit
    rows=await asyncio.gather(*jobs)
    active_row=await active
    assert active_row['status']==200
    for ref,row in zip(refs,rows[:4]):comparable(ref,row)
    comparable(refs[0],rows[-1])
    assert rows[4]['status']==413
    assert rows[4]['body']['input_tokens']==limit+1 and rows[4]['body']['max_input_tokens']==limit
    assert float(rows[4]['headers']['x-clef-queue-wait-ms'])>0
    waits=[r['body']['usage']['queue_wait_ms'] for r in rows if r['status']==200]
    assert all(ms>0 for ms in waits)
    image_usage=rows[2]['body']['usage']
    assert image_usage['prefix_cache']=='hit' or image_usage.get('image_feature_cache_hits',0)>=2
    result.update(busy=busy,blocker=active_row,queued=rows)
    save()
    # Eight simultaneous inbound calls are buffered, with one active GPU call.
    burst=await asyncio.gather(*(http('/v1/systemone',wire(bodies[i%4])) for i in range(8)))
    for i,row in enumerate(burst):comparable(refs[i%4],row)
    assert sum(r['body']['usage']['queue_wait_ms']>10 for r in burst)>=6
    result['concurrent_eight']=burst;save()
    # A long call may evict a small prefix on an 8 GB card. Once pressure is
    # over, an immediate repeated photo pair must recover prefix reuse.
    image_recovery=[await http('/v1/systemone',wire(photo)) for _ in range(2)]
    for row in image_recovery:comparable(refs[2],row)
    assert image_recovery[-1]['body']['usage']['prefix_cache']=='hit'
    result['image_recovery']=image_recovery;save()
    # Actual TCP disconnect before execution removes its queued request.
    cancelled_before=(await http('/health'))['body']['queue']['cancelled']
    blocker=wire(sized(text(f'Queue cancellation {time.time_ns()}. Checkout is down. Act immediately. '),8192))
    active=asyncio.create_task(http('/v1/systemone',blocker))
    await until(lambda h:h['queue']['active_requests']==1)
    parsed=urllib.parse.urlparse(a.url)
    reader,writer=await asyncio.open_connection(parsed.hostname,parsed.port)
    raw=json.dumps(wire(normal)).encode()
    writer.write(f'POST /v1/systemone HTTP/1.1\r\nHost: {parsed.netloc}\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n'.encode()+raw)
    await writer.drain()
    await until(lambda h:h['queue']['waiting_requests']==1)
    writer.close();await writer.wait_closed()
    await until(lambda h:h['queue']['cancelled']>cancelled_before and h['queue']['waiting_requests']==0)
    assert (await active)['status']==200
    final=(await http('/health'))['body']
    assert final['queue']['active_requests']==final['queue']['waiting_requests']==0
    assert final['queue']['admitted_payload_mib']==0 and final['queue']['admitted_requests']==0
    current=(await http('/v1/models'))['body']['data'][0]
    assert current['max_input_tokens']==metadata['max_input_tokens']
    assert current['max_input_tokens_with_images']==metadata['max_input_tokens_with_images']
    result.update(completed=True,final_health=final,final_models=current,real_disconnect_cancelled=True)
    save()
    print(json.dumps({'completed':True,'url':a.url,'waiting_peak':busy['queue']['waiting_requests'],
                      'queue_wait_ms':waits,'all_valid_status_200':True,'queued_413':True,'real_disconnect_cancelled':True}),flush=True)


if __name__=='__main__':asyncio.run(main())
