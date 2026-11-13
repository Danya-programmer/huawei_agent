"""Unit tests for LLM_MODEL resolution."""

from __future__ import annotations

import os
import unittest

from harness.domain.types import DEFAULT_MODEL, resolve_model


class TestResolveModel(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("LLM_MODEL", None)

    def test_default(self) -> None:
        os.environ.pop("LLM_MODEL", None)
        self.assertEqual(resolve_model(), DEFAULT_MODEL.value)
        self.assertEqual(resolve_model(), "qwen3.8-27b-noreason")

    def test_env_override(self) -> None:
        os.environ["LLM_MODEL"] = "qwen3.6-35b-a3b-noreason"
        self.assertEqual(resolve_model(), "qwen3.6-35b-a3b-noreason")

    def test_explicit_wins(self) -> None:
        os.environ["LLM_MODEL"] = "qwen3.6-35b-a3b-noreason"
        self.assertEqual(resolve_model("qwen3.8-27b"), "qwen3.8-27b")

    def test_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_model("gpt-4o")


if __name__ == "__main__":
    unittest.main()
