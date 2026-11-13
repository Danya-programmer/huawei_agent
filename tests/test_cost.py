"""Unit tests for the competition token-cost formula."""

from __future__ import annotations

import math
import unittest

from harness.infra.cost import compute_call_cost, cost_from_usage
from harness.domain.types import TokenUsage


class TestComputeCallCost(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(compute_call_cost(0, 0, 0), 0)

    def test_prompt_only_ceil(self) -> None:
        # 1..10 prompt -> 1; 11 -> 2
        self.assertEqual(compute_call_cost(1, 0, 0), 1)
        self.assertEqual(compute_call_cost(10, 0, 0), 1)
        self.assertEqual(compute_call_cost(11, 0, 0), 2)
        self.assertEqual(compute_call_cost(100, 0, 0), 10)

    def test_full_formula(self) -> None:
        prompt, completion, reasoning = 25, 40, 7
        expected = completion + reasoning + math.ceil(prompt / 10)
        self.assertEqual(
            compute_call_cost(prompt, completion, reasoning),
            expected,
        )
        self.assertEqual(expected, 40 + 7 + 3)

    def test_reasoning_counts_as_output(self) -> None:
        self.assertEqual(compute_call_cost(0, 0, 15), 15)

    def test_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            compute_call_cost(-1, 0, 0)

    def test_from_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=19, completion_tokens=5, reasoning_tokens=2)
        self.assertEqual(cost_from_usage(usage), 5 + 2 + 2)


if __name__ == "__main__":
    unittest.main()
