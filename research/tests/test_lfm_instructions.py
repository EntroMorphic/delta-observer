import unittest

from research.lfm.extract_lfm_instructions import (
    extract_user_text,
    normalize_text,
    prompt_stream,
)


class LFMInstructionExtractionTests(unittest.TestCase):
    def test_extracts_first_user_turn_from_messages(self):
        record = {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "  Explain recursion.  "},
                {"role": "assistant", "content": "Later."},
            ]
        }
        self.assertEqual(extract_user_text(record, "messages"), "Explain recursion.")

    def test_supports_alternate_and_flat_schemas(self):
        self.assertEqual(
            extract_user_text({"conversations": [{"from": "human", "value": "Hi"}]}, "conversations"),
            "Hi",
        )
        self.assertEqual(extract_user_text({"instruction": "Do this"}, "messages"), "Do this")

    def test_normalization_is_case_and_whitespace_stable(self):
        self.assertEqual(normalize_text(" A  Mixed\nPrompt "), "a mixed prompt")

    def test_prompt_stream_excludes_frozen_and_duplicate_prompts(self):
        records = [
            {"instruction": "Keep this"},
            {"instruction": " keep   THIS "},
            {"instruction": "Frozen prompt"},
            {"instruction": "Another prompt"},
        ]
        stats = {}
        prompts = list(prompt_stream(records, "messages", {"frozen prompt"}, stats))
        self.assertEqual(prompts, ["Keep this", "Another prompt"])
        self.assertEqual(stats["rejected_duplicate_prompt"], 1)
        self.assertEqual(stats["rejected_frozen_prompt"], 1)


if __name__ == "__main__":
    unittest.main()
