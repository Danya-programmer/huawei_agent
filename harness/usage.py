"""Extract token usage from heterogeneous LLM / agent response shapes."""

from __future__ import annotations

from typing import Any, Optional

from harness.types import TokenUsage


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_from_mapping(data: Any) -> TokenUsage:
    """Parse OpenAI-like usage dict (also nested completion_tokens_details)."""
    if not isinstance(data, dict):
        return TokenUsage()

    # Some SDKs nest under "usage".
    if "prompt_tokens" not in data and isinstance(data.get("usage"), dict):
        data = data["usage"]

    details = data.get("completion_tokens_details") or data.get("output_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}

    prompt = _as_int(
        data.get("prompt_tokens")
        or data.get("input_tokens")
        or data.get("prompt")
    )
    completion = _as_int(
        data.get("completion_tokens")
        or data.get("output_tokens")
        or data.get("completion")
    )
    reasoning = _as_int(
        details.get("reasoning_tokens")
        or data.get("reasoning_tokens")
        or data.get("reasoning")
    )
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
    )


def extract_usage(payload: Any) -> TokenUsage:
    """Best-effort walk of agent/LLM result objects for usage fields."""
    total = TokenUsage()
    seen: set[int] = set()

    def walk(obj: Any) -> None:
        nonlocal total
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, dict):
            token_keys = (
                "prompt_tokens",
                "completion_tokens",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
            )
            if any(k in obj for k in token_keys):
                total = total + usage_from_mapping(obj)
                return
            if isinstance(obj.get("usage"), dict):
                total = total + usage_from_mapping(obj["usage"])
                for key, value in obj.items():
                    if key == "usage":
                        continue
                    walk(value)
                return
            for value in obj.values():
                walk(value)
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
            return

        # pydantic / SDK objects
        for attr in ("usage", "token_usage", "model_dump", "dict"):
            if not hasattr(obj, attr):
                continue
            try:
                value = getattr(obj, attr)
                if callable(value):
                    value = value()
            except Exception:  # noqa: BLE001
                continue
            walk(value)

    walk(payload)
    return total
