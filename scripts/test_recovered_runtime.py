"""CPU regressions for cold timing, memory scheduling and bounded telemetry."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from cold_benchmark import ColdPassTiming, rate
from observability import DropQueue, ObservationConfig, Observability
from prefill_schedule import PrefillSchedule, MIB
from revision_bank import RevisionBank


class Module:
    def __init__(self):
        self.before, self.after = {}, {}

    def register_forward_pre_hook(self, hook, **kwargs):
        self.before[hook] = hook
        return NS(remove=lambda: self.before.pop(hook))

    def register_forward_hook(self, hook, **kwargs):
        self.after[hook] = hook
        return NS(remove=lambda: self.after.pop(hook))

    def __call__(self, **kwargs):
        for hook in list(self.before):
            hook(self, (), kwargs)
        for hook in list(self.after):
            hook(self, (), kwargs, object())


class DeviceEvents:
    def __init__(self):
        self.clock = 0

    def Event(self, **kwargs):
        def record():
            event.time = self.clock
            self.clock += 5
        event = NS(record=record)
        event.elapsed_time = lambda other: other.time - event.time
        return event


class ColdTiming(unittest.TestCase):
    def setUp(self):
        self.language, self.head, self.vision = Module(), Module(), Module()
        self.timing = ColdPassTiming(NS(language_model=NS(model=NS(
            language_model=self.language, visual=self.vision)), head=self.head), DeviceEvents())

    def run_pass(self):
        with self.timing.capture():
            self.vision()
            self.language(input_ids=NS(shape=(1, 7)))
            self.language(inputs_embeds=NS(shape=(1, 3, 64)))
            self.head()

    def test_only_language_tokens_contribute_to_backbone_rate(self):
        self.run_pass()
        result = self.timing.summary({'input_tokens': 10, 'gpu_time_ms': 40})
        self.assertTrue(result['valid'])
        self.assertEqual(result['processed_language_tokens'], 10)
        self.assertEqual(result['language_forward_calls'], 2)
        self.assertEqual(result['language_tokens_per_second'], 1000)
        self.assertEqual(result['gpu_window_tokens_per_second'], 250)
        self.assertEqual(result['vision_window_ms'], 5)
        self.assertTrue(result['head_timing_complete'])
        for module in (self.language, self.head, self.vision):
            self.assertFalse(module.before or module.after)

    def test_reuse_and_mismatched_accounting_never_report_a_rate(self):
        self.run_pass()
        for usage, reason in [({'input_tokens': 11}, 'language_token_count_mismatch'),
                              ({'input_tokens': 10, 'token_cache_hits': 1}, 'cache_reuse_detected')]:
            result = self.timing.summary(usage)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['language_tokens_per_second'])
            self.assertEqual(result['invalid_reason'], reason)

    def test_failure_removes_hooks_and_invalidates_measurement(self):
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            with self.timing.capture():
                self.language(input_ids=NS(shape=(1, 10)))
                raise RuntimeError('inference failed')
        self.assertFalse(self.language.before or self.language.after)
        result = self.timing.summary({'input_tokens': 10})
        self.assertEqual(result['invalid_reason'], 'inference_retry_or_failure')
        self.assertIsNone(result['language_tokens_per_second'])

    def test_unavailable_rates_are_absent(self):
        for tokens, ms in [(0, 1), (True, 1), (10, 0), (10, float('nan')), (10, float('inf'))]:
            self.assertIsNone(rate(tokens, ms))


class MemoryScheduling(unittest.TestCase):
    def schedule(self, **kwargs):
        config = NS(hidden_size=1024, intermediate_size=4096, num_attention_heads=8,
                    num_key_value_heads=2, head_dim=128, layer_types=['full_attention'] * 4)
        return PrefillSchedule(config, dtype_bytes=2, **kwargs)

    def test_first_new_block_and_checkpoint_boundaries(self):
        schedule = self.schedule()
        self.assertEqual(schedule.choose(10000, 30000, 10000*MIB, initial=True), 8192)
        self.assertEqual(schedule.choose(10000, 30000, 10000*MIB), 4096)
        self.assertEqual(schedule.choose(10000, 10333, 10000*MIB), 333)

    def test_memory_pressure_reduces_block_before_allocation(self):
        schedule = self.schedule()
        available = schedule.reserve + schedule.workspace(1024, 12000)
        self.assertEqual(schedule.choose(12000, 30000, available), 1024)
        self.assertEqual(schedule.memory_limited, 1)
        with self.assertRaises(MemoryError):
            schedule.choose(12000, 30000, schedule.reserve)

    def test_streamed_history_does_not_grow_gpu_workspace(self):
        streamed = self.schedule(mode='kv_stream')
        self.assertEqual(streamed.workspace(1024, 1000), streamed.workspace(1024, 100000))
        explicit, implicit = self.schedule(), self.schedule(implicit=True)
        self.assertLess(implicit.workspace(1024, 100000), explicit.workspace(1024, 100000))


class BoundedTelemetry(unittest.TestCase):
    def test_queue_count_byte_and_busy_lock_limits(self):
        queue = DropQueue(2, 8)
        self.assertTrue(queue.offer('a', 5))
        self.assertFalse(queue.offer('b', 4))
        with queue.lock:
            self.assertFalse(queue.offer('busy', 1))
        self.assertEqual(queue.take(), ['a'])
        self.assertEqual(queue.bytes, 0)
        self.assertEqual(queue.dropped, 2)

    def test_metrics_exclude_user_content_and_invalid_numbers(self):
        observer = Observability(ObservationConfig(metrics=True))
        observer.record(200, 10, {'input_tokens': 3, 'state': 'private-prompt',
            'authorization': 'private-key', 'peak_allocated_mib': float('nan')})
        events = observer.events.take()
        self.assertEqual(len(events), 1)
        self.assertNotIn('private', str(events))
        self.assertNotIn('peak_allocated_mib', events[0])
        observer.consume(events[0])
        observer.publish()
        self.assertIn('clef_input_tokens_total{profile="flash"} 3', observer.snapshot)
        self.assertNotIn('private', observer.snapshot)

    def test_disabled_observer_does_no_work(self):
        observer = Observability(ObservationConfig())
        observer.record(200, 10, {'input_tokens': 3})
        self.assertFalse(observer.events.take() or observer.logs.take())


class RevisionAccounting(unittest.TestCase):
    def test_storage_changes_invalidate_totals_but_recency_does_not(self):
        bank = RevisionBank(a=1, b=2)
        revision = bank.revision
        bank.move_to_end('a')
        bank.setdefault('a', 3)
        self.assertEqual(bank.revision, revision)
        bank['a'] = 4
        bank.pop('b')
        self.assertEqual(bank.revision, revision + 2)
        bank.clear()
        revision = bank.revision
        bank.clear()
        self.assertEqual(bank.revision, revision)


if __name__ == '__main__':
    unittest.main()
