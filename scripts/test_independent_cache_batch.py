"""CPU checks of independent history isolation, padding and logical positions."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import torch
from independent_cache_batch import continuation_inputs, pack_states, IncompatibleHistory


class DynamicLayer:
    is_initialized = True


def entry(ids, marker):
    layer = DynamicLayer()
    layer.keys = torch.full((1, 1, len(ids), 2), marker)
    layer.values = layer.keys.clone()
    return {'ids': tuple(ids), 'cache': NS(layers=[layer])}


class IndependentHistory(unittest.TestCase):
    def test_histories_left_pad_without_mutating_sources(self):
        rows = [entry([1, 2], 10.), entry([3, 4, 5], 20.)]
        packed, lengths, size = pack_states(rows, 'cpu')
        self.assertEqual(lengths, [2, 3])
        self.assertGreater(size, 0)
        self.assertEqual(packed.layers[0].keys[0, 0, :, 0].tolist(), [0, 10, 10])
        self.assertEqual(packed.layers[0].keys[1, 0, :, 0].tolist(), [20, 20, 20])
        packed.layers[0].keys.zero_()
        self.assertTrue(torch.all(rows[0]['cache'].layers[0].keys == 10))

    def test_suffix_mask_hides_padding_and_future_tokens(self):
        histories = [entry([1, 2], 10.), entry([3, 4, 5], 20.)]
        records = [NS(input_ids=(1, 2, 6, 7), media=None), NS(input_ids=(3, 4, 5, 8), media=None)]
        ids, positions, bias, prefixes, suffixes = continuation_inputs(
            records, histories, 0, 'cpu', torch.float32)
        self.assertEqual(ids.tolist(), [[6, 7], [8, 0]])
        self.assertEqual(positions[0].tolist(), [[2, 3], [3, 4]])
        self.assertEqual(prefixes, [2, 3])
        self.assertEqual(suffixes, [2, 1])
        self.assertEqual(torch.isfinite(bias[0, 0]).tolist(),
                         [[False, True, True, True, False], [False, True, True, True, True]])
        self.assertEqual(torch.isfinite(bias[1, 0]).tolist(),
                         [[True, True, True, True, False], [True, True, True, True, False]])

    def test_wrong_ancestry_and_offloading_are_rejected(self):
        histories = [entry([1, 2], 10.)]
        with self.assertRaises(IncompatibleHistory):
            continuation_inputs([NS(input_ids=(1, 9, 6), media=None)], histories, 0, 'cpu', torch.float32)
        histories[0]['cache'].offloading = True
        with self.assertRaises(IncompatibleHistory):
            pack_states(histories, 'cpu')


if __name__ == '__main__':
    unittest.main()
