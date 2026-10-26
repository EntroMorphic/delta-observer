import unittest
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

from research.lfm.evaluate_pretrained_development import (
    chat_inputs, content_validation, run, strict_json, validate, validate_suite,
)


class DevelopmentValidatorTests(unittest.TestCase):
    def test_exact_json_rejects_wrong_types_and_extra_text(self):
        task = {"validator": "json_exact", "expected": {"count": 7}}
        self.assertTrue(validate(task, '{"count":7}')["pass"])
        for text in ('{"count":7.0}', '{"count":true}', '```json\n{"count":7}\n```',
                     '{"count":7,"extra":0}'):
            self.assertFalse(validate(task, text)["pass"])

    def test_text_exact_is_strict_except_outer_whitespace(self):
        task = {"validator": "text_exact", "expected": "33"}
        self.assertTrue(validate(task, "33\n")["pass"])
        self.assertFalse(validate(task, "The answer is 33.")["pass"])

    def test_manual_is_not_counted_as_failure_or_success(self):
        result = validate({"validator": "manual", "rubric": "Review content"}, "Answer")
        self.assertIsNone(result["pass"])
        self.assertEqual(result["status"], "pending_blinded_review")

    def test_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        for text in ('{"x":0,"x":7}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                strict_json(text)
        task = {"validator": "json_exact", "expected": {"x": 7}}
        self.assertFalse(validate(task, '{"x":0,"x":7}')["pass"])

    def test_suite_rejects_duplicates_and_empty_suites(self):
        task = {"id": "a", "category": "c", "prompt": " Hi ",
                "validator": "text_exact", "expected": "yes"}
        base = {"role": "development_only_not_confirmatory"}
        for tasks in ([], [task, task], [task, {**task, "id": "b", "prompt": "hi"}]):
            with self.assertRaises(ValueError):
                validate_suite({**base, "tasks": tasks})

    def test_content_parser_does_not_turn_wrong_json_into_success(self):
        task = {"validator": "json_exact", "content_validator": "json_unfenced",
                "expected": [1, 2]}
        self.assertTrue(content_validation(task, '```json\n[1,2]\n```')["pass"])
        self.assertFalse(validate(task, '```json\n[1,2]\n```')["pass"])
        self.assertFalse(content_validation(task, '```json\n[2,1]\n```')["pass"])
        self.assertFalse(content_validation(task, 'Explanation [1,2]')["pass"])

    def test_boxed_parser_rejects_conflicting_answers_and_incidental_numbers(self):
        task = {"validator": "text_exact", "content_validator": "boxed_integer",
                "expected": "33"}
        self.assertTrue(content_validation(task, r"42-9 = \boxed{33}")["pass"])
        for text in (r"\boxed{33} or \boxed{34}", "There were 33 pencils initially."):
            self.assertIsNone(content_validation(task, text)["pass"])

    def test_existing_result_is_not_overwritten_or_model_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / "baseline.json"
            result.write_text("original")
            args = argparse.Namespace(output=folder, max_new_tokens=10,
                revision="a" * 40, tasks="research/lfm/pretrained_development_tasks.json")
            with patch("research.lfm.evaluate_pretrained_development.AutoTokenizer.from_pretrained") as load:
                with self.assertRaises(FileExistsError):
                    run(args)
                load.assert_not_called()
            self.assertEqual(result.read_text(), "original")

    def test_chat_template_is_not_given_extra_special_tokens(self):
        from unittest.mock import Mock
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "<BOS>prompt"
        chat_inputs(tokenizer, "prompt")
        tokenizer.assert_called_once_with(
            "<BOS>prompt", add_special_tokens=False, return_tensors="pt"
        )

    def test_expanded_suite_is_valid_and_unique(self):
        from research.lfm.build_pretrained_development_suite import build
        suite = build()
        validate_suite(suite)
        self.assertEqual(len(suite["tasks"]), 44)
