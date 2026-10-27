"""Bounded cache diagnostics. Never retain prompts, token IDs or image URLs."""
import hashlib
import json
import secrets
import time
from collections import Counter, deque
from contextvars import ContextVar
from threading import RLock


class CacheTelemetry:
    def __init__(self, event_limit=2048, frequency=None):
        self.frequency = frequency
        self.limit = max(0, int(event_limit))
        self.events = deque(maxlen=self.limit)
        self.counters = Counter()
        self.sequence = 0
        self.salt = secrets.token_bytes(32)
        self.lock = RLock()
        self.request = ContextVar('cache_request', default=None)

    def fingerprint(self, key):
        if key is None:
            return None
        # Salting makes low-entropy prefixes harder to identify from diagnostics.
        # Identity is stable across CPU/GPU tiers only within this process.
        raw = json.dumps(key, default=str, separators=(',', ':')).encode()
        return hashlib.sha256(self.salt + raw).hexdigest()[:24]

    def begin(self):
        request_id = secrets.token_hex(8)
        token = self.request.set(request_id)
        self.record('request_begin', 'service')
        return token, request_id, self.sequence

    def end(self, scope):
        token, request_id, start = scope
        self.record('request_end', 'service')
        trace = {'request_id': request_id, 'first_event': start,
                 'last_event': self.sequence}
        self.request.reset(token)
        return trace

    def record(self, action, tier, key=None, *, entry=None, reason=None, **fields):
        with self.lock:
            self.sequence += 1
            self.counters[f'{tier}.{action}' + ('.' + reason if reason else '')] += 1
            if not self.limit:
                return
            event = {'event': self.sequence, 'unix_ms': round(time.time() * 1000, 1),
                     'request_id': self.request.get(), 'action': action, 'tier': tier}
            if key is not None:
                event['key'] = self.fingerprint(key)
            if reason:
                event['reason'] = reason
            if entry is not None:
                now = time.monotonic_ns()
                event.update(tokens=len(entry.get('ids', ())), uses=entry.get('uses', 0),
                             logical_bytes=entry.get('bytes'),
                             age_ms=round((now-entry.get('created_ns', now))/1e6, 1),
                             idle_ms=round((now-entry.get('last_use', now))/1e6, 1))
                if self.frequency is not None:
                    event['eviction_frequency'] = self.frequency.effective(entry)
            event.update(fields)
            self.events.append(event)

    def stats(self):
        with self.lock:
            return {'event_limit': self.limit, 'last_event': self.sequence,
                    'first_retained_event': self.events[0]['event'] if self.events else None,
                    'counters': dict(self.counters)}

    def read(self, after=0, limit=512):
        with self.lock:
            rows = [dict(row) for row in self.events if row['event'] > after][:limit]
            first = self.events[0]['event'] if self.events else self.sequence + 1
            return {**self.stats(), 'events': rows,
                    'cursor_gap': after < first-1,
                    'next_cursor': rows[-1]['event'] if rows else after}

    def inventory(self, bank, tier, limit=256):
        now = time.monotonic_ns()
        rows = []
        for key, entry in list(bank.items())[:limit]:
            item = {'key': self.fingerprint(key), 'tier': tier,
                         'tokens': len(entry['ids']), 'logical_bytes': entry['bytes'],
                         'uses': entry.get('uses', 0),
                         'age_ms': round((now-entry.get('created_ns', now))/1e6, 1),
                         'idle_ms': round((now-entry.get('last_use', now))/1e6, 1)}
            if self.frequency is not None:
                item['eviction_frequency'] = self.frequency.effective(entry)
            rows.append(item)
        return {'entries': len(bank), 'truncated': len(bank) > limit, 'items': rows}
