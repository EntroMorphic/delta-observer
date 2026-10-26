import unittest
import torch

from research.lfm.audit_native_update_conflict import assign,remove_harmful_component


class UpdateConflictTests(unittest.TestCase):
    def test_projection_removes_only_positive_retention_component(self):
        delta=torch.tensor([3.,4.]);g=torch.tensor([1.,0.])
        projected,matched,metrics=remove_harmful_component(delta,g)
        self.assertTrue(torch.equal(projected,torch.tensor([0.,4.],dtype=torch.float64)))
        self.assertAlmostEqual(float(matched.norm()),5.)
        self.assertAlmostEqual(float(matched@g.double()),0.)
        self.assertTrue(metrics['projection_active'])

    def test_nonharmful_direction_is_unchanged(self):
        delta=torch.tensor([-3.,4.]);g=torch.tensor([1.,0.])
        projected,matched,metrics=remove_harmful_component(delta,g)
        self.assertTrue(torch.equal(projected,delta.double()))
        self.assertTrue(torch.equal(matched,delta.double()))
        self.assertFalse(metrics['projection_active'])

    def test_invalid_projection_inputs_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            remove_harmful_component(torch.tensor([1.]),torch.tensor([float('nan')]))
        with self.assertRaises(ValueError):
            remove_harmful_component(torch.tensor([1.,2.]),torch.tensor([1.]))

    def test_bad_assignment_does_not_partially_mutate_parameters(self):
        parameters=[torch.nn.Parameter(torch.tensor([1.,2.])),torch.nn.Parameter(torch.tensor([3.]))]
        with self.assertRaises(ValueError):assign(parameters,torch.tensor([7.,8.]))
        self.assertTrue(torch.equal(parameters[0],torch.tensor([1.,2.])))
        self.assertTrue(torch.equal(parameters[1],torch.tensor([3.])))
