import unittest

from research.lfm.compare_control_prompt import continuation_metrics


class ControlPromptTests(unittest.TestCase):
    def test_repetition_has_low_distinct_ngram_ratios(self):
        repeated = continuation_metrics([1, 2, 1, 2, 1, 2, 1, 2])
        varied = continuation_metrics(list(range(8)))
        self.assertLess(repeated["distinct_bigram_ratio"], varied["distinct_bigram_ratio"])
        self.assertLess(repeated["distinct_trigram_ratio"], varied["distinct_trigram_ratio"])


if __name__ == "__main__":
    unittest.main()
