"""Best-effort request observations, never called by the GPU inference worker.

Producers take a nonblocking lock and only enqueue bounded host metadata.
Aggregation, exposition formatting, JSON serialization and storage run on
separate threads. Scrapes return a previously published immutable snapshot.
"""
from collections import Counter, deque
from dataclasses import dataclass
import asyncio
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time

NUMBERS = ('input_tokens', 'output_tokens', 'reused_prefix_tokens', 'batch_size',
           'gpu_time_ms', 'latency_ms', 'queue_wait_ms', 'server_total_ms',
           'cpu_time_ms', 'prefix_restore_ms', 'prefix_offload_ms', 'peak_allocated_mib',
           'scheduler_matching_ms', 'scheduler_candidates', 'scheduler_bypasses',
           'worker_handoff_ms','cache_assembly_ms','independent_working_state_mib',
           'independent_history_padding_tokens','independent_suffix_padding_tokens')
CACHE_STATES = {'hit', 'miss', 'disabled', 'not_retained', 'memory_fallback'}
TIERS = {'gpu', 'cpu', 'miss', 'recompute'}
BUCKETS = (.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 300)
DURATIONS = {'request_duration': 'http_time_ms', 'queue_wait': 'queue_wait_ms',
             'inference_window': 'gpu_time_ms', 'processing': 'latency_ms',
             'cpu_estimated': 'cpu_time_ms', 'prefix_restore': 'prefix_restore_ms',
             'prefix_offload': 'prefix_offload_ms'}


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except (OverflowError, TypeError):
        return False


@dataclass(frozen=True)
class ObservationConfig:
    metrics: bool = False
    log_mode: str = 'off'
    log_images: bool = False
    log_dir: Path | None = None
    queue_items: int = 4096
    log_queue_bytes: int = 64 * 2**20
    rotate_bytes: int = 8 * 2**20
    backups: int = 4
    retention_seconds: float = 7 * 86400
    archive_bytes: int = 1024 * 2**20

    def __post_init__(self):
        if self.log_mode not in {'off', 'metadata', 'payload'}:
            raise ValueError('Request log mode must be off, metadata or payload')
        if self.log_mode != 'off' and self.log_dir is None:
            raise ValueError('Request logging requires a log directory')
        if self.log_images and self.log_mode != 'payload':
            raise ValueError('Image capture requires payload logging')
        for value in (self.queue_items, self.log_queue_bytes, self.rotate_bytes,
                      self.backups, self.retention_seconds, self.archive_bytes):
            if not number(value) or value < 1:
                raise ValueError('Observation limits must be finite and positive')

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        for name in ('CLEF_METRICS', 'CLEF_REQUEST_LOG_IMAGES'):
            if env.get(name, '0') not in {'0', '1'}:
                raise ValueError(name + ' must be 0 or 1')
        path = env.get('CLEF_REQUEST_LOG_DIR')
        return cls(metrics=env.get('CLEF_METRICS', '0') == '1',
                   log_mode=env.get('CLEF_REQUEST_LOG_MODE', 'off'),
                   log_images=env.get('CLEF_REQUEST_LOG_IMAGES', '0') == '1',
                   log_dir=Path(path) if path else None,
                   queue_items=int(env.get('CLEF_OBSERVATION_QUEUE_ITEMS', '4096')),
                   log_queue_bytes=int(env.get('CLEF_REQUEST_LOG_QUEUE_MIB', '64')) * 2**20,
                   rotate_bytes=int(env.get('CLEF_REQUEST_LOG_ROTATE_MIB', '8')) * 2**20,
                   backups=int(env.get('CLEF_REQUEST_LOG_BACKUPS', '4')),
                   retention_seconds=float(env.get('CLEF_REQUEST_LOG_RETENTION_DAYS', '7')) * 86400,
                   archive_bytes=int(env.get('CLEF_REQUEST_LOG_ARCHIVE_MIB', '1024')) * 2**20)


class DropQueue:
    """Count and byte bounds; no producer waits for a lock or a consumer."""
    def __init__(self, count, size):
        self.rows = deque()
        self.count_limit, self.byte_limit = count, size
        self.bytes = self.dropped = 0
        self.lock = threading.Lock()

    def offer(self, value, size):
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return False
        try:
            if len(self.rows) >= self.count_limit or self.bytes + size > self.byte_limit:
                self.dropped += 1
                return False
            self.rows.append((value, size))
            self.bytes += size
            return True
        finally:
            self.lock.release()

    def take(self, limit=128):
        with self.lock:
            result = []
            while self.rows and len(result) < limit:
                value, size = self.rows.popleft()
                self.bytes -= size
                result.append(value)
            return result


class PayloadArchive:
    """Optional single-writer SQLite archive with lossless, deduplicated images.

    SQLite already ships in the runtime. Images are BLOBs; foreign keys preserve
    ordering and make retention safe for images referenced by several requests.
    No headers, GPU tensors, image decoding to pixels or re-encoding are used.
    """
    def __init__(self, config):
        self.config = config
        self.db = sqlite3.connect(config.log_dir / 'payloads.sqlite3')
        os.chmod(config.log_dir / 'payloads.sqlite3', 0o600)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA max_page_count=' + str(max(128, config.archive_bytes // 4096)))
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, created REAL,
                record TEXT NOT NULL, bytes INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS request_age ON requests(created);
            CREATE TABLE IF NOT EXISTS images(hash TEXT PRIMARY KEY, mime TEXT, data BLOB);
            CREATE TABLE IF NOT EXISTS request_images(request_id TEXT REFERENCES requests(id)
                ON DELETE CASCADE, ordinal INTEGER, hash TEXT REFERENCES images(hash),
                PRIMARY KEY(request_id, ordinal));
            CREATE INDEX IF NOT EXISTS image_references ON request_images(hash);
        ''')
        self.db.commit()

    def used(self):
        return (self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM requests').fetchone()[0]
                + self.db.execute('SELECT COALESCE(SUM(length(data)),0) FROM images').fetchone()[0])

    def orphans(self):
        self.db.execute('DELETE FROM images WHERE NOT EXISTS '
                        '(SELECT 1 FROM request_images WHERE request_images.hash=images.hash)')

    def put(self, metadata, request, result):
        # model_dump and JSON encoding happen only in this background writer.
        if hasattr(request, 'model_dump'):
            body = request.model_dump(exclude={'images'})
            images = request.images
        else:
            body = {k:v for k,v in request.items() if k != 'images'}
            images = request.get('images', [])
        blobs, refs = {}, []
        if self.config.log_images:
            for ordinal, image in enumerate(images, 1):
                header, encoded = image.split(',', 1)
                mime = {'data:image/jpeg;base64': 'image/jpeg',
                        'data:image/png;base64': 'image/png',
                        'data:image/webp;base64': 'image/webp'}.get(header)
                if not mime or len(encoded) > ((10 * 2**20 + 2)//3)*4:
                    raise ValueError('Invalid or oversized capture image')
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > 10 * 2**20:
                    raise ValueError('Oversized capture image')
                digest = hashlib.sha256(raw).hexdigest()
                blobs[digest] = (mime, raw)
                refs.append({'ordinal': ordinal, 'sha256': digest, 'mime': mime})
        body['images'] = refs
        if isinstance(result, tuple):
            status, raw = result
            result = {'http_status':status, 'body':json.loads(raw)}
        record = {**metadata, 'request': body, 'result': result,
                  'image_count': len(images), 'images_captured': self.config.log_images}
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        size = len(serialized.encode('utf-8'))
        # A too-large record cannot evict the entire existing archive.
        if size + sum(len(raw) for _, raw in blobs.values()) > self.config.archive_bytes * .85:
            return False
        with self.db:
            self.db.execute('DELETE FROM requests WHERE created < ?',
                            (time.time() - self.config.retention_seconds,))
            self.orphans()
            extra = sum(len(raw) for digest, (_, raw) in blobs.items()
                        if not self.db.execute('SELECT 1 FROM images WHERE hash=?', (digest,)).fetchone())
            while self.used() + size + extra > self.config.archive_bytes * .85:
                oldest = self.db.execute('SELECT id FROM requests ORDER BY created LIMIT 1').fetchone()
                if oldest is None:
                    return False
                self.db.execute('DELETE FROM requests WHERE id=?', oldest)
                self.orphans()
                # Deleted references may make an incoming shared blob new again.
                extra = sum(len(raw) for digest, (_, raw) in blobs.items()
                            if not self.db.execute('SELECT 1 FROM images WHERE hash=?', (digest,)).fetchone())
            for digest, (mime, raw) in blobs.items():
                self.db.execute('INSERT OR IGNORE INTO images VALUES(?,?,?)', (digest, mime, raw))
            self.db.execute('INSERT INTO requests VALUES(?,?,?,?)',
                            (metadata['request_id'], metadata['unix_ms']/1000, serialized, size))
            self.db.executemany('INSERT INTO request_images VALUES(?,?,?)',
                               [(metadata['request_id'], r['ordinal'], r['sha256']) for r in refs])
        return True

    def close(self):
        self.db.close()


class MetadataWriter:
    def __init__(self, config):
        self.config = config
        config.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = config.log_dir / 'requests.jsonl'
        self.archive = PayloadArchive(config) if config.log_mode == 'payload' else None

    def write(self, event, capture):
        row = {**event, 'schema_version': 1, 'unix_ms': event['completed_unix_ms'],
               'request_id': secrets.token_hex(16)}
        if capture is not None:
            request, result = capture
            try:
                row['payload_saved'] = self.archive.put(row, request, result)
            except Exception as exc:
                row['payload_saved'] = False
                row['payload_error_type'] = type(exc).__name__
        encoded = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        if self.path.exists() and self.path.stat().st_size + len(encoded) > self.config.rotate_bytes:
            for i in range(self.config.backups, 0, -1):
                previous = self.path if i == 1 else self.path.with_name('requests.'+str(i-1)+'.jsonl')
                target = self.path.with_name('requests.'+str(i)+'.jsonl')
                if previous.exists():
                    previous.replace(target)
        # Rotation bounds file count, including across service restarts.
        cutoff = time.time() - self.config.retention_seconds
        for i in range(1, self.config.backups+1):
            path = self.path.with_name('requests.'+str(i)+'.jsonl')
            if path.exists() and path.stat().st_mtime < cutoff:
                path.unlink()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, 'ab') as stream:
            stream.write(encoded)
        return row.get('payload_saved', True)

    def close(self):
        if self.archive:
            self.archive.close()


class Observability:
    def __init__(self, config, *, profile='flash', gauges=None):
        self.config = config
        self.profile = profile if profile in {'flash', 'full'} else 'unknown'
        self.gauges = gauges
        self.enabled = config.metrics or config.log_mode != 'off'
        self.events = DropQueue(config.queue_items, config.queue_items * 2048)
        self.logs = DropQueue(config.queue_items, config.log_queue_bytes)
        self.stop_event = threading.Event()
        self.threads = []
        self.statuses, self.cache = Counter(), Counter()
        self.tokens = Counter()
        self.histograms = {name: [0, 0.0, [0]*len(BUCKETS)] for name in DURATIONS}
        self.live = {}
        self.log_written = self.log_errors = self.payload_dropped = self.metric_errors = 0
        self.last_log_error = None
        self.snapshot = ''
        self.publish()

    def start(self):
        if self.config.metrics:
            self.threads.append(threading.Thread(target=self._collect, name='clef-metrics', daemon=True))
        if self.config.log_mode != 'off':
            self.threads.append(threading.Thread(target=self._write, name='clef-request-logs', daemon=True))
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2)

    def record(self, status, elapsed_ms, usage, *, wire_bytes=0, capture=None):
        if not self.enabled:
            return
        status = status if type(status) is int and 100 <= status <= 599 else 500
        event = {'status': status, 'http_time_ms': elapsed_ms, 'profile': self.profile,
                 'completed_unix_ms':round(time.time()*1000, 1)}
        event['error_category'] = ('cancelled' if status == 499 else 'server_error' if status >= 500
                                   else 'client_error' if status >= 400 else 'none')
        # Fixed allowlist: no prompt strings, credentials, IDs in metric labels,
        # entry inventories, serialization, CUDA calls or memory traversal.
        for name in NUMBERS:
            value = usage.get(name)
            if number(value):
                event[name] = value
        state, tier = usage.get('prefix_cache'), usage.get('prefix_cache_tier')
        event['prefix_cache'] = state if isinstance(state,str) and state in CACHE_STATES else 'unknown'
        event['prefix_cache_tier'] = tier if isinstance(tier,str) and tier in TIERS else 'unknown'
        trace = usage.get('cache_trace') or {}
        request_id = trace.get('request_id') if isinstance(trace, dict) else None
        if isinstance(request_id, str):
            event['cache_request_id'] = request_id[:64]
        if self.config.metrics:
            self.events.offer(event, 2048)
        if self.config.log_mode != 'off':
            # Raw wire length gives a conservative bound without walking or
            # serializing a possibly large request in the producer. Payloads
            # carry an additional 8x wire reservation for parsed object overhead.
            size = 2048 + max(0, wire_bytes)*8 if capture is not None else 2048
            if not self.logs.offer((event, capture), size) and capture is not None:
                self.payload_dropped += 1
                self.logs.offer(({**event, 'payload_dropped': True}, None), 2048)

    def consume(self, event):
        self.statuses[event['status']] += 1
        self.cache[(event['prefix_cache'], event['prefix_cache_tier'])] += 1
        for name in ('input_tokens', 'output_tokens', 'reused_prefix_tokens'):
            self.tokens[name] += event.get(name, 0)
        for name, field in DURATIONS.items():
            if field in event:
                value = event[field] / 1000
                h = self.histograms[name]
                h[0] += 1; h[1] += value
                for i, bound in enumerate(BUCKETS):
                    h[2][i] += value <= bound
        for name in ('batch_size', 'peak_allocated_mib'):
            if name in event:
                self.live[name] = event[name]

    def publish(self):
        if self.config.metrics and self.gauges is not None:
            try:
                self.live.update({k:v for k,v in self.gauges().items() if number(v)})
            except Exception:
                self.metric_errors += 1
        lines = ['# HELP clef_observability_enabled Optional best-effort metrics collection.',
                 '# TYPE clef_observability_enabled gauge',
                 'clef_observability_enabled '+str(int(self.config.metrics)),
                 '# TYPE clef_requests_total counter']
        label = 'profile="'+self.profile+'"'
        for status in sorted(set(self.statuses) | {200,400,401,413,422,429,499,500,503,504}):
            lines.append(f'clef_requests_total{{{label},status="{status}"}} {self.statuses[status]}')
        lines.append('# TYPE clef_cache_requests_total counter')
        for (state, tier), value in sorted(self.cache.items()):
            lines.append(f'clef_cache_requests_total{{{label},result="{state}",tier="{tier}"}} {value}')
        for name in ('input_tokens','output_tokens','reused_prefix_tokens'):
            lines += ['# TYPE clef_'+name+'_total counter',f'clef_{name}_total{{{label}}} {self.tokens[name]}']
        for name, (count, total, buckets) in self.histograms.items():
            metric='clef_'+name+'_seconds'
            lines.append('# TYPE '+metric+' histogram')
            for bound, value in zip(BUCKETS, buckets):
                lines.append(f'{metric}_bucket{{{label},le="{bound}"}} {value}')
            lines += [f'{metric}_bucket{{{label},le="+Inf"}} {count}',
                      f'{metric}_count{{{label}}} {count}',f'{metric}_sum{{{label}}} {total}']
        for name, value in self.live.items():
            metric='clef_'+name
            if name=='peak_allocated_mib':metric='clef_last_peak_allocated_bytes';value*=2**20
            elif name=='batch_size':metric='clef_last_batch_size'
            lines += ['# TYPE '+metric+(' counter' if name.endswith('_total') else ' gauge'),f'{metric}{{{label}}} {value}']
        for name, value in self.stats().items():
            if type(value) is int:
                metric='clef_observation_'+name
                lines += ['# TYPE '+metric+(' counter' if name.endswith('_total') else ' gauge'),f'{metric} {value}']
        lines += ['# TYPE clef_observation_snapshot_timestamp_seconds gauge',
                  'clef_observation_snapshot_timestamp_seconds '+str(time.time())]
        self.snapshot = '\n'.join(lines)+'\n'

    def stats(self):
        return {'metrics_enabled': self.config.metrics, 'log_mode': self.config.log_mode,
                'images_captured': self.config.log_images, 'metric_queue_items':len(self.events.rows),
                'log_queue_items':len(self.logs.rows), 'log_queue_bytes':self.logs.bytes,
                'metrics_dropped_total':self.events.dropped,'logs_dropped_total':self.logs.dropped,
                'payloads_dropped_total':self.payload_dropped,'logs_written_total':self.log_written,
                'log_errors_total':self.log_errors,'metric_errors_total':self.metric_errors,
                'last_log_error_type':self.last_log_error}

    def _collect(self):
        next_publish=0
        while not self.stop_event.is_set() or self.events.rows:
            for event in self.events.take():
                try:self.consume(event)
                except Exception:self.metric_errors+=1
            if time.monotonic()>=next_publish:
                self.publish();next_publish=time.monotonic()+.5
            if not self.stop_event.is_set():self.stop_event.wait(.025)
        self.publish()

    def _write(self):
        writer=None
        try:
            try:writer=MetadataWriter(self.config)
            except Exception as exc:self.log_errors+=1;self.last_log_error=type(exc).__name__
            while not self.stop_event.is_set() or self.logs.rows:
                for event,capture in self.logs.take(16):
                    if writer is None:
                        self.log_errors+=1
                        continue
                    try:
                        if not writer.write(event,capture):self.payload_dropped+=1
                        self.log_written+=1
                    except Exception as exc:
                        self.log_errors+=1;self.last_log_error=type(exc).__name__
                if not self.stop_event.is_set():self.stop_event.wait(.025)
        finally:
            if writer is not None:
                try:writer.close()
                except Exception:self.log_errors+=1


class ObservationMiddleware:
    def __init__(self, app, observer):
        self.app,self.observer=app,observer

    async def __call__(self, scope, receive, send):
        if not self.observer.enabled or scope['type']!='http' or scope.get('method')!='POST' or scope.get('path')!='/v1/systemone':
            return await self.app(scope,receive,send)
        started=time.perf_counter();status=500
        async def observed_send(message):
            nonlocal status
            if message['type']=='http.response.start':status=message['status']
            await send(message)
        try:
            return await self.app(scope,receive,observed_send)
        except asyncio.CancelledError:
            status=499
            raise
        finally:
            ticket=scope.get('clef_payload_ticket')
            capture=None
            if self.observer.config.log_mode=='payload' and 'clef_capture_request' in scope:
                capture=(scope['clef_capture_request'],scope.get('clef_capture_result',{'status':status}))
            # The HTTP response has already been sent; failures here must never
            # propagate into the response or queue an inference retry.
            try:self.observer.record(status,(time.perf_counter()-started)*1000,
                    scope.get('clef_observation_usage',{}),wire_bytes=ticket.bytes if ticket else 0,capture=capture)
            except Exception:self.observer.metric_errors+=1
