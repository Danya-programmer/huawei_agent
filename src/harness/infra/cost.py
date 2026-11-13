"""Token cost accounting helpers."""

from __future__ import annotations

import math

from harness.domain.types import TokenUsage


def compute_call_cost(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> int:
    """Budget cost of one LLM call in output-token units.

    Formula from the competition rules:
        cost = completion_tokens + reasoning_tokens + ceil(prompt_tokens / 10)

    Reasoning tokens count as output. Provider cache is ignored (full prompt).
    """
    if prompt_tokens < 0 or completion_tokens < 0 or reasoning_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return (
        int(completion_tokens)
        + int(reasoning_tokens)
        + math.ceil(int(prompt_tokens) / 10)
    )


def cost_from_usage(usage: TokenUsage) -> int:
    """Apply the competition formula to a TokenUsage aggregate."""
    return compute_call_cost(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )
