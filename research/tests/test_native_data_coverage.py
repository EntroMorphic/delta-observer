import json
import unittest

from research.lfm.build_native_pilot_data import select_balanced_pairs


class NativeDataCoverageTests(unittest.TestCase):
    def test_shortest_bucket_cannot_swallow_other_items(self):
        buckets = {}
        for item, length in (("clock", 10), ("lantern", 13)):
            buckets[item, length, 4] = [
                {"prompt": f"{item} {i}", "response": json.dumps({'item': item}),
                 "input_ids": [0] * length, "supervised_tokens": 4}
                for i in range(32)]
        unique, repeated = select_balanced_pairs(buckets, items=("clock", "lantern"))
        self.assertEqual(len(unique), 32)
        for rows in (unique, repeated):
            for item in ("clock", "lantern"):
                self.assertEqual(sum(json.loads(row['response'])['item'] == item for row in rows), 16)
        self.assertEqual(len({row['prompt'] for row in repeated}), 4)
        self.assertTrue(all(len(a['input_ids']) == len(b['input_ids']) for a,b in zip(unique,repeated)))

    def test_missing_item_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            select_balanced_pairs({}, items=("clock",))
