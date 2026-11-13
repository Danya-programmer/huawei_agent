"""Tests for answer.txt sanitization."""

from __future__ import annotations

import unittest

from harness.answer import looks_like_garbage, sanitize_answer


class TestAnswerSanitize(unittest.TestCase):
    def test_accepts_short_values(self) -> None:
        self.assertEqual(sanitize_answer("42"), "42")
        self.assertEqual(sanitize_answer("Osaka\n"), "Osaka")
        self.assertEqual(sanitize_answer('"Move Your Body"'), '"Move Your Body"')

    def test_rejects_shell_command(self) -> None:
        raw = (
            'python3 -c "import pandas as pd; df = pd.read_csv(\'sales.csv\'); '
            "print(len(df))\" > answer.txt"
        )
        self.assertTrue(looks_like_garbage(raw))
        self.assertIsNone(sanitize_answer(raw))

    def test_rejects_markdown_fence_only(self) -> None:
        self.assertIsNone(sanitize_answer("```"))
        self.assertIsNone(sanitize_answer("```\n```"))

    def test_rejects_tool_xml(self) -> None:
        self.assertIsNone(sanitize_answer("<arg_value>ls kb/</arg_value>"))

    def test_rejects_chatty_reasoning(self) -> None:
        raw = "Let me start by exploring the kb/ directory first."
        self.assertIsNone(sanitize_answer(raw))

    def test_extracts_last_line_answer(self) -> None:
        raw = "Some scratch notes\n42"
        # two short lines — last line kept
        self.assertEqual(sanitize_answer(raw), "42")

    def test_coding_done(self) -> None:
        self.assertEqual(sanitize_answer("DONE", coding=True), "DONE")
        self.assertIsNone(sanitize_answer("python3 test_mylib.py", coding=True))


if __name__ == "__main__":
    unittest.main()
