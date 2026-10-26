import unittest

import torch

from research.lfm.build_retention_format import fact_pools, new_suite, pack
from research.lfm.train_retention_format import verify_groups


class RetentionFormatTests(unittest.TestCase):
    def test_exact_budget_uses_distinct_rows(self):
        pool=[{'prompt':str(i),'supervised_tokens':n} for i,n in enumerate((2,3,4,5))]
        rows=pack(pool,7,0)
        self.assertEqual(sum(r['supervised_tokens'] for r in rows),7)
        self.assertEqual(len({r['prompt'] for r in rows}),len(rows))
        with self.assertRaises(ValueError):pack(pool,20,0)

    def test_fact_splits_have_no_unintended_overlap(self):
        primary,extra,test=fact_pools()
        self.assertFalse(set(primary)&set(extra))
        self.assertFalse((set(primary)|set(extra))&set(test))
        suite=new_suite(primary,extra,test)
        self.assertEqual(len(suite['tasks']),100)
        seen=[t for t in suite['tasks'] if t['split']=='seen_fact_new_template']
        self.assertEqual(len(seen),20)
        self.assertTrue(all(tuple(t['operands']) in primary for t in seen))

    def test_token_weighted_gradient_equals_concatenated_loss(self):
        x=torch.tensor(.4,requires_grad=True)
        first=(x-torch.tensor([1.,2.])).square()
        second=(x-torch.tensor([3.,4.,5.])).square()
        mean=(first.mean()*2+second.mean()*3)/5
        actual=torch.autograd.grad(mean,x,retain_graph=True)[0]
        expected=torch.autograd.grad(torch.cat((first,second)).mean(),x)[0]
        self.assertTrue(torch.allclose(actual,expected))

    def test_bad_group_budget_rejected_before_tokenization(self):
        for group in ({'supervised_tokens':48,'rows':[{'supervised_tokens':2}]},
                      {'supervised_tokens':2,'rows':[{'supervised_tokens':48}]}):
            with self.assertRaises(ValueError):verify_groups([group],None,48)
