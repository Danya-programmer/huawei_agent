"""Unit tests for usage extraction / cost aggregation."""

from __future__ import annotations

import unittest

from harness.cost import cost_from_usage
from harness.usage import extract_usage, usage_from_mapping


class TestUsage(unittest.TestCase):
    def test_openai_shape(self) -> None:
        usage = usage_from_mapping(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
            }
        )
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.reasoning_tokens, 5)
        self.assertEqual(cost_from_usage(usage), 20 + 5 + 10)

    def test_nested_walk(self) -> None:
        payload = {
            "content": "ok",
            "meta": {
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 7,
                    "reasoning_tokens": 3,
                }
            },
        }
        usage = extract_usage(payload)
        self.assertEqual(usage.prompt_tokens, 50)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertEqual(usage.reasoning_tokens, 3)


if __name__ == "__main__":
    unittest.main()
