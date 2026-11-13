"""Smoke-test openJiuwen config wiring without calling the LLM."""

from __future__ import annotations

import unittest


class TestReactSmoke(unittest.TestCase):
    def test_build_config_or_skip(self) -> None:
        try:
            import openjiuwen  # noqa: F401
        except ImportError:
            self.skipTest("openjiuwen not installed in this environment")
        from harness.agents.react_agent import smoke_build_config

        info = smoke_build_config()
        self.assertEqual(info["model"], "qwen3.8-27b-noreason")
        self.assertTrue(info["has_model_client"])


if __name__ == "__main__":
    unittest.main()
