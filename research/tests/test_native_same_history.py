import unittest

import torch
from torch import nn

from research.lfm.audit_native_same_history import (
    adapter_mode, distribution_pair, first_divergence, safe_prefix, unsafe_text,
)
from research.lfm.native_lora import LoRALinear


class ToyTokenizer:
    all_special_ids = [0]
    pieces = ['<eos>', 'First, ', 'compute ', '18', 'the ', 'twen', 'ty', ' tokens', '²', 'one']

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.pieces[i] for i in ids)


class SameHistoryTests(unittest.TestCase):
    def test_prefix_excludes_digits_and_control_tokens(self):
        tokenizer = ToyTokenizer()
        row = safe_prefix(tokenizer, [1, 2, 3, 7, 0])
        self.assertEqual(row['ids'], [1, 2])
        self.assertEqual(row['stop_reason'], 'numeric_character')
        self.assertEqual(safe_prefix(tokenizer, [1, 0, 2])['ids'], [1])
        self.assertEqual(safe_prefix(tokenizer, [8])['ids'], [])

    def test_split_number_word_does_not_leak_partial_cardinal_at_cap(self):
        row = safe_prefix(ToyTokenizer(), [4, 5, 6, 7], cap=2)
        self.assertEqual(row['ids'], [4])
        self.assertEqual(row['stop_reason'], 'cardinal_number_word')
        self.assertIsNone(unsafe_text('First, calculate the total'))
        self.assertEqual(unsafe_text('The result is one'), 'cardinal_number_word')

    def test_same_distribution_has_zero_js_and_extreme_logits_stay_finite(self):
        logits = torch.tensor([2., 1., -10000.])
        row = distribution_pair(logits, logits)
        self.assertAlmostEqual(row['js_nats'], 0., places=12)
        self.assertTrue(row['top1_equal'])
        row = distribution_pair(logits, -logits)
        self.assertTrue(0. <= row['js_nats'] <= .693148)

    def test_scale_context_restores_adapters_after_exception(self):
        model = nn.Sequential(LoRALinear(nn.Linear(2, 2), 1, 2))
        with self.assertRaises(RuntimeError):
            with adapter_mode(model, False):
                self.assertEqual(model[0].scale, 0.)
                raise RuntimeError('test')
        self.assertEqual(model[0].scale, 2.)

    def test_divergence_includes_length_difference(self):
        self.assertIsNone(first_divergence([1, 2], [1, 2]))
        self.assertEqual(first_divergence([1], [1, 2]), 1)
        self.assertEqual(first_divergence([1, 2], [1, 3]), 1)


if __name__ == '__main__':
    unittest.main()
