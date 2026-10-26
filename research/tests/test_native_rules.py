import unittest

from research.lfm.audit_native_rules import extract_answer, probes
from research.lfm.build_native_rule_remediation import examples, test_suite
from research.lfm.evaluate_pretrained_development import validate_suite


class RuleProbeTests(unittest.TestCase):
    def test_every_order_case_has_both_directions(self):
        rows=[r for r in probes() if r['family']=='order']
        self.assertEqual(len(rows),160)
        for mode in {r['mode'] for r in rows}:
            for case in range(20):
                pair=[r for r in rows if r['mode']==mode and r['case']==case]
                self.assertEqual({r['requested'] for r in pair},{'earliest','latest'})
                self.assertNotEqual(pair[0]['expected'],pair[1]['expected'])

    def test_narrow_parser_does_not_mine_favorable_ids(self):
        row={'family':'order'}
        self.assertEqual(extract_answer(row,'Event R201'),'R201')
        self.assertIsNone(extract_answer(row,'R201 or T401'))

    def test_remediation_pairs_hold_premises_fixed(self):
        rows=examples()
        self.assertEqual(len(rows),1216)
        pairs={}
        for row in rows:
            if row['category']=='transitive_ordering':
                pairs.setdefault(row['pair_id'],[]).append(row)
        self.assertEqual(len(pairs),160)
        for pair in pairs.values():
            self.assertEqual(len(pair),2)
            self.assertEqual(pair[0]['prompt'].split(' Select ')[0],pair[1]['prompt'].split(' Select ')[0])
            self.assertNotEqual(pair[0]['response'],pair[1]['response'])
        suite=test_suite()
        validate_suite(suite)
        self.assertEqual(len(suite['tasks']),100)
