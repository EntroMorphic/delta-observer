import unittest

from research.lfm.build_native_holdout100 import build_tasks
from research.lfm.evaluate_pretrained_development import validate_suite
from research.lfm.score_native_holdout100 import format_pass, score


class HoldoutTests(unittest.TestCase):
    def test_exactly_one_hundred_valid_tasks_twenty_per_category(self):
        suite = build_tasks()
        validate_suite(suite)
        tasks = suite["tasks"]
        self.assertEqual(len(tasks), 100)
        for category in {task["category"] for task in tasks}:
            self.assertEqual(sum(task["category"] == category for task in tasks), 20)

    def test_format_is_not_content(self):
        task = build_tasks()["tasks"][0]
        self.assertTrue(format_pass(task, '{"wrong":true}'))
        self.assertFalse(format_pass(task, '```json\n{"label":"cedar","enabled":true}\n```'))

    def test_wrong_content_can_remain_semantically_unresolved(self):
        tasks = build_tasks()["tasks"]
        rows = [{"id": task["id"], "prompt": task["prompt"],
                 "response": "explanation", "hit_token_limit_without_eos": False}
                for task in tasks]
        result = score(tasks, rows)
        self.assertEqual(result["strict_successes"], 0)
        self.assertGreater(result["content_unresolved"], 0)
        self.assertEqual(result["content_confirmed"] + result["content_parser_failures"]
                         + result["content_unresolved"], 100)
        with self.assertRaises(ValueError):
            score(tasks, rows[:-1])
        with self.assertRaises(ValueError):
            score(tasks, rows + [rows[0]])
