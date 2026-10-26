import unittest

from research.lfm.audit_native_fact_wording import crossing, render


class CrossingTests(unittest.TestCase):
    def test_labels_unchanged_across_wording_and_overlap_reported(self):
        train=[{'mode':'compound_final','prompt':render(i+3,4,2,'training'),
                'response':str((i+3)*4-2)} for i in range(128)]
        test={'tasks':[{'category':'arithmetic','prompt':render(i+3,4,2,'diagnostic'),
                'expected':str((i+3)*4-2)} for i in range(20)]}
        rows,overlap=crossing(train,test)
        self.assertEqual(len(rows),80)
        self.assertEqual(len(overlap),20)
        for facts in ('training','diagnostic'):
            for i in range(20):
                pair=[r for r in rows if r['fact_source']==facts and r['case']==i]
                self.assertEqual(pair[0]['expected'],pair[1]['expected'])
                self.assertNotEqual(pair[0]['prompt'],pair[1]['prompt'])
        test['tasks'][0]['expected']='wrong'
        with self.assertRaises(ValueError):crossing(train,test)
