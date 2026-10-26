import unittest

import torch

from research.lfm.build_semantic_transfer import pools
from research.lfm.native_retention_transfer import build_tasks, direction_controls
from research.lfm.audit_retention_transfer_outputs import classify_integer
from research.lfm.score_native_holdout100 import score


class RetentionTransferTests(unittest.TestCase):
    def test_four_constraints_are_individually_orthogonal_and_norm_preserved(self):
        delta = torch.tensor([1., 2., 3., 4., 5., 6.])
        gradients = [torch.eye(6)[i] for i in range(4)]
        projected, control, metrics = direction_controls(delta, gradients, torch.Generator().manual_seed(4))
        self.assertEqual(metrics['gradient_rank'], 4)
        self.assertTrue(torch.allclose(projected[:4], torch.zeros(4, dtype=torch.float64), atol=1e-12))
        self.assertAlmostEqual(float(projected.norm()), float(delta.double().norm()), places=10)
        self.assertAlmostEqual(float(control.norm()), float(projected.norm()), places=10)
        self.assertAlmostEqual(float(control @ delta.double()), float(projected @ delta.double()), places=10)
        self.assertFalse(torch.allclose(control, projected))

    def test_duplicate_gradients_and_zero_updates_do_not_produce_nan(self):
        d = torch.tensor([1., 2., 3.])
        g = torch.tensor([1., 0., 0.])
        projected, _, metrics = direction_controls(d, [g, g*2, torch.zeros_like(g)], torch.Generator())
        self.assertEqual(metrics['gradient_rank'], 1)
        self.assertEqual(float(projected[0]), 0.)
        projected, control, _ = direction_controls(torch.zeros(3), [g], torch.Generator())
        self.assertTrue(torch.equal(projected, control))
        self.assertTrue(torch.isfinite(control).all())

    def test_full_span_reports_zero_rather_than_claiming_same_norm(self):
        projected, control, metrics = direction_controls(torch.tensor([1., 2.]),
            list(torch.eye(2)), torch.Generator())
        self.assertTrue(metrics['zero_projection'])
        self.assertEqual(float(projected.norm()), 0.)
        self.assertEqual(float(control.norm()), 0.)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            direction_controls(torch.ones(3), [torch.ones(2)], torch.Generator())
        with self.assertRaises(ValueError):
            direction_controls(torch.tensor([float('nan')]), [], torch.Generator())

    def test_fresh_suite_facts_disjoint_and_existing_scoring_compatible(self):
        suite = build_tasks()
        self.assertEqual(suite, build_tasks())
        tasks = suite['tasks']
        self.assertEqual(len(tasks), 100)
        train, old_test = pools()
        arithmetic = [t for t in tasks if t['category'] == 'arithmetic']
        self.assertEqual(len(arithmetic), 40)
        self.assertFalse({tuple(t['operands']) for t in arithmetic} & set(train+old_test))
        rows = [{'id': t['id'], 'prompt': t['prompt'],
                 'response': t['expected'] if isinstance(t['expected'], str) else
                             __import__('json').dumps(t['expected']),
                 'hit_token_limit_without_eos': False} for t in tasks]
        self.assertEqual(score(tasks, rows)['strict_successes'], 100)

    def test_post_unsealing_taxonomy_is_not_a_replacement_success_parser(self):
        task = {'operands': [18, 6, 4], 'operation': 'add'}
        self.assertEqual(classify_integer(task, '112'), 'correct_bare_integer')
        self.assertEqual(classify_integer(task, '108'), 'unchanged_product')
        self.assertEqual(classify_integer(task, '104'), 'opposite_operation')
        self.assertEqual(classify_integer(task, '101'), 'other_wrong_integer')
        self.assertEqual(classify_integer(task, r'\boxed{112}'), 'not_bare_integer')
        self.assertEqual(classify_integer(task, '112 and 108'), 'not_bare_integer')


if __name__ == '__main__':
    unittest.main()
