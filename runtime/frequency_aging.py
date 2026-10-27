"""Lazy integer frequency decay, independent of lifetime reuse telemetry."""
import math
import time


class FrequencyAging:
    def __init__(self, decay_seconds=900, clock=None):
        seconds = float(decay_seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Cache frequency decay seconds must be finite and nonnegative')
        self.interval_ns = int(seconds * 1e9)
        if seconds and not self.interval_ns:
            raise ValueError('Cache frequency decay interval must be at least one nanosecond')
        self.clock = clock or time.monotonic_ns

    @property
    def policy(self):
        return 'aged_lfu_lru_tiebreak' if self.interval_ns else 'lfu_lru_tiebreak'

    def _state(self, entry, now):
        uses = entry.get('uses', 0)
        at = entry.get('frequency_updated_ns', entry.get('last_use', now))
        if not self.interval_ns:
            return uses, now
        frequency = entry.get('frequency', uses)
        periods = max(0, now-at) // self.interval_ns
        # Avoid huge shifts for very old entries. Integer halving reaches zero.
        value = frequency >> periods if periods < frequency.bit_length() else 0
        return value, at + periods*self.interval_ns

    def effective(self, entry, now=None):
        return self._state(entry, self.clock() if now is None else now)[0]

    def snapshot(self, entry, now=None):
        """Copy bookkeeping without treating admission/transfer as a hit."""
        now = self.clock() if now is None else now
        value, at = self._state(entry, now)
        return {'uses': entry.get('uses', 0), 'last_use': entry.get('last_use', now),
                'frequency': value, 'frequency_updated_ns': at}

    def touch(self, entry, uses=1):
        if uses < 0:
            raise ValueError('Reuse count must be nonnegative')
        if not uses:
            return  # Rejected restores and transfers cannot refresh recency.
        now = self.clock()
        value, at = self._state(entry, now)
        entry.update(uses=entry.get('uses', 0)+uses, last_use=now,
                     frequency=value+uses, frequency_updated_ns=at)
        # Preserve fractional periods, so frequent hits do not postpone decay.

    def merge(self, target, source):
        """Reconcile overlapping CPU/GPU histories without counting them twice."""
        if 'last_use' not in source:
            # A fresh snapshot write has no access history. Its default
            # timestamp must not make an already retained entry look recent.
            target['uses'] = max(target.get('uses', 0), source.get('uses', 0))
            return
        now = self.clock()
        left, right = self.snapshot(target, now), self.snapshot(source, now)
        winner = max((left, right), key=lambda v: (v['frequency'], v['last_use']))
        target.update(uses=max(left['uses'], right['uses']),
                      last_use=max(left['last_use'], right['last_use']),
                      frequency=winner['frequency'],
                      frequency_updated_ns=winner['frequency_updated_ns'])

    def stats(self):
        return {'enabled': bool(self.interval_ns), 'decay_seconds': self.interval_ns/1e9,
                'decay': 'integer_halving', 'rounding': 'down', 'lazy': True,
                'lifetime_hits_preserved': True}
