import unittest

import torch

from research.lfm.audit_native_arithmetic_access import (
    numeric_choice, numeric_support, paired, score_response,
)


class ArithmeticAccessTests(unittest.TestCase):
    def test_numeric_support_has_no_answer_dependency_and_rejects_control_or_partial_tokens(self):
        class Tokenizer:
            all_special_ids = [0]
            pieces = ['123', '7', '-4', ' 8', '1.5', '00', '-0', '1e2', '42']
            def __len__(self):
                return len(self.pieces)
            def decode(self, ids, skip_special_tokens=False):
                return self.pieces[ids[0]]
            def encode(self, text, add_special_tokens=False):
                return [4, 2] if text == '42' else [self.pieces.index(text)]
        support = numeric_support(Tokenizer())
        self.assertEqual(support, [1, 2])
        logits = torch.tensor([100., 1., 2., 999., 10., 3., 4., 7., 8.])
        self.assertEqual(numeric_choice(logits, support), 2)

    def test_answer_prefix_is_not_silently_stripped_for_native_exact(self):
        task = {'validator': 'text_exact', 'content_validator': 'boxed_integer', 'expected': '112'}
        self.assertTrue(score_response(task, '112')['native_exact'])
        full = score_response(task, 'Final answer: 112')
        self.assertFalse(full['native_exact'])
        self.assertFalse(full['confirmed_content'])
        self.assertTrue(score_response(task, r'Final answer: \boxed{112}')['confirmed_content'])

    def test_paired_losses_are_reported_even_with_equal_counts(self):
        def row(identifier, success):
            return {'id': identifier, 'scores': {'confirmed_content': success}}
        result = paired([row('a', False), row('b', True)], [row('a', True), row('b', False)])
        self.assertEqual(result, {'content_gains': ['b'], 'content_losses': ['a']})
        with self.assertRaises(ValueError):
            paired([row('a', True), row('a', True)], [row('a', True), row('b', False)])


if __name__ == '__main__':
    unittest.main()
