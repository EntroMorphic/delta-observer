import unittest
from types import SimpleNamespace
import torch
from torch import nn

from research.lfm.audit_semantic_atomics import distribution_metrics,patched_forward
from research.lfm.audit_semantic_controls import balanced_prompt


class SemanticAtomicTests(unittest.TestCase):
    def test_target_and_operation_margins_are_not_top1_accuracy(self):
        logits=torch.zeros(64);logits[2]=5;logits[3]=3;logits[4]=9
        result=distribution_metrics(logits,2,3)
        self.assertEqual(result['target_minus_alternative_logit'],2)
        self.assertEqual(result['target_rank'],2)
        self.assertEqual(result['top1_id'],4)
        self.assertEqual(result['target_minus_top1_logit'],-4)

    def test_balanced_control_changes_operation_not_numbers(self):
        subtract=balanced_prompt((7,4,3),'subtract')
        add=balanced_prompt((7,4,3),'add')
        self.assertEqual(subtract.replace('loses','gains'),add)

    def test_patch_changes_last_position_only_and_removes_hook(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model=nn.Module()
                self.model.layers=nn.ModuleList([nn.Identity()])
                self.last_states=None

            def forward(self,input_ids,use_cache=False):
                states=input_ids.float().unsqueeze(-1).repeat(1,1,3)
                states=self.model.layers[0](states)
                self.last_states=states.detach().clone()
                return SimpleNamespace(logits=states)

        model=Toy()
        inputs={'input_ids':torch.tensor([[1,2,3]])}
        donor=torch.tensor([7.,8.,9.])
        actual=patched_forward(model,inputs,0,donor)
        self.assertTrue(torch.equal(actual,donor))
        self.assertTrue(torch.equal(model.last_states[0,:2],torch.tensor([[1.,1.,1.],[2.,2.,2.]])))
        original=model(**inputs).logits[0,-1]
        self.assertTrue(torch.equal(original,torch.tensor([3.,3.,3.])))
