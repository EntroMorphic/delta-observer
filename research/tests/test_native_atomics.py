import unittest
import torch

from research.lfm.audit_native_atomics import first_difference, token_metrics


class NativeAtomicTests(unittest.TestCase):
    def test_first_difference_covers_prefix_length(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1, 2], [1]), 1)

    def test_token_ranks_and_margins(self):
        class Tokenizer:
            def decode(self, ids):
                return str(ids)
        logits = torch.zeros(2, 64)
        logits[0, 3] = 4
        logits[1, 5] = 6
        metrics = token_metrics(logits, torch.tensor([3, 2]), Tokenizer())
        self.assertEqual(metrics[0]["target_rank"], 1)
        self.assertEqual(metrics[1]["target_rank"], 2)
        self.assertEqual(metrics[0]["target_vs_best_other_logit_margin"], 4)
        self.assertEqual(metrics[1]["target_vs_best_other_logit_margin"], -6)
        self.assertAlmostEqual(metrics[0]["top64_mass"], 1, places=5)
