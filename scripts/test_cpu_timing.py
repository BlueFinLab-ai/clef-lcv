"""Check timing attribution for inline, lookahead, shared and legacy requests."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from cpu_timing import cpu_timing


class CPUTimingTests(unittest.TestCase):
    def test_inline_preparation_counted_once(self):
        usage = {'gpu_time_ms': 200, 'latency_ms': 240, 'preprocessing_ms': 25}
        self.assertEqual(cpu_timing(usage, 350), {'cpu_time_ms': 40, 'cpu_time_estimated': True})

    def test_lookahead_preparation_included(self):
        usage = {'gpu_time_ms': 200, 'latency_ms': 215, 'preprocessing_ms': 25,
                 'cpu_preparation_overlapped': True}
        self.assertEqual(cpu_timing(usage, 350)['cpu_time_ms'], 40)

    def test_shared_or_prepared_work_bounded_by_request_window(self):
        usage = {'gpu_time_ms': 200, 'latency_ms': 240, 'preprocessing_ms': 90,
                 'cpu_preparation_overlapped': True}
        self.assertEqual(cpu_timing(usage, 250)['cpu_time_ms'], 50)

    def test_rounding_and_zero_gpu(self):
        self.assertEqual(cpu_timing({'gpu_time_ms': 10.1, 'latency_ms': 10}, 10)['cpu_time_ms'], 0)
        self.assertEqual(cpu_timing({'gpu_time_ms': 0, 'latency_ms': 10}, 20)['cpu_time_ms'], 10)

    def test_missing_or_invalid_device_timing_not_called_cpu(self):
        for gpu in (None, -1, float('nan')):
            self.assertEqual(cpu_timing({'gpu_time_ms': gpu, 'latency_ms': 20}, 30), {})


if __name__ == '__main__':
    unittest.main()
