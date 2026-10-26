import unittest

from research.lfm.build_semantic_transfer import design,pools,render,result,suite


class SemanticTransferTests(unittest.TestCase):
    def test_same_facts_different_meaning(self):
        fact=(7,4,3)
        self.assertEqual(result(fact,'subtract'),'25')
        self.assertEqual(result(fact,'add'),'31')
        for template in range(5):
            self.assertNotEqual(render(fact,template,'subtract'),render(fact,template,'add'))
        self.assertEqual(len(set(design('repeat'))),1)
        self.assertEqual(len(set(design('paraphrase'))),4)
        self.assertEqual({op for _,op in design('contrast')},{'subtract','add'})

    def test_frozen_split_and_pairing_contract(self):
        train,test=pools()
        self.assertFalse(set(train)&set(test))
        tasks=suite(train,test)['tasks']
        self.assertEqual(len(tasks),100)
        for index in range(20):
            pair=[t for t in tasks if t['id'] in (
                f'new_facts_new_template-{index:02d}',f'counterfactual_add-{index:02d}')]
            self.assertEqual(pair[0]['operands'],pair[1]['operands'])
            self.assertNotEqual(pair[0]['expected'],pair[1]['expected'])
