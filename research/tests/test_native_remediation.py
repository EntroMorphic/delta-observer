import json
import unittest
from collections import Counter

from research.lfm.build_native_remediation import holdout, training_examples
from research.lfm.evaluate_pretrained_development import validate_suite


class RemediationTests(unittest.TestCase):
    def test_balanced_nonconstant_training_coverage(self):
        rows=training_examples()
        self.assertEqual(len(rows),640)
        self.assertEqual(set(Counter(r['category'] for r in rows).values()),{128})
        records=[r for r in rows if r['category']=='json_record']
        self.assertEqual(len(set(r['value'] for r in records)),16)
        self.assertEqual(set(Counter(r['value'] for r in records).values()),{8})
        self.assertEqual(set(r['style'] for r in records),{'compact','spaced','pretty'})
        for row in records:
            self.assertEqual(json.loads(row['response'])['label'],row['value'])

    def test_new_suite_and_train_are_prompt_disjoint(self):
        suite=holdout()
        validate_suite(suite)
        self.assertEqual(len(suite['tasks']),100)
        train={r['prompt'] for r in training_examples()}
        self.assertFalse(train & {r['prompt'] for r in suite['tasks']})
