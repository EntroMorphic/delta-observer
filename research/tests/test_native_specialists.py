import unittest

from research.lfm.build_native_specialists import subset


class SpecialistTests(unittest.TestCase):
    def test_within_task_exposure_sequence_is_preserved(self):
        rows=[{'category':'a','id':0},{'category':'b','id':1},{'category':'a','id':2}]
        selected,order=subset(rows,[{'row_index':2},{'row_index':1},{'row_index':0}],'a')
        self.assertEqual([selected[i]['id'] for i in order],[2,0])

    def test_duplicate_history_cannot_silently_drop_source_exposures(self):
        with self.assertRaises(ValueError):
            subset([{'category':'a'},{'category':'b'}],[{'row_index':0},{'row_index':0}],'a')
