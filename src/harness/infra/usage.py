"""Extract token usage from heterogeneous LLM / agent response shapes."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Optional

from harness.domain.types import TokenUsage

_active_bucket: ContextVar[Optional[list[TokenUsage]]] = ContextVar(
    "hack_llm_usage_bucket", default=None
)
_hook_lock = threading.Lock()
_hook_registered = False


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_from_mapping(data: Any) -> TokenUsage:
    """Parse OpenAI-like usage dict (also nested completion_tokens_details)."""
    if data is None:
        return TokenUsage()
    if not isinstance(data, dict):
        # openjiuwen UsageMetadata / pydantic
        if hasattr(data, "model_dump") and callable(data.model_dump):
            try:
                data = data.model_dump()
            except Exception:  # noqa: BLE001
                data = {
                    "input_tokens": getattr(data, "input_tokens", 0),
                    "output_tokens": getattr(data, "output_tokens", 0),
                    "reasoning_tokens": getattr(data, "reasoning_tokens", 0),
                    "prompt_tokens": getattr(data, "prompt_tokens", None),
                    "completion_tokens": getattr(data, "completion_tokens", None),
                }
        else:
            data = {
                "input_tokens": getattr(data, "input_tokens", 0),
                "output_tokens": getattr(data, "output_tokens", 0),
                "reasoning_tokens": getattr(data, "reasoning_tokens", 0),
                "prompt_tokens": getattr(data, "prompt_tokens", None),
                "completion_tokens": getattr(data, "completion_tokens", None),
            }
    if not isinstance(data, dict):
        return TokenUsage()

    # Some SDKs nest under "usage".
    if "prompt_tokens" not in data and "input_tokens" not in data and isinstance(
        data.get("usage"), dict
    ):
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
            if obj.get("usage_metadata") is not None:
                total = total + usage_from_mapping(obj.get("usage_metadata"))
            for value in obj.values():
                walk(value)
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)
            return

        # pydantic / SDK objects — prefer usage_metadata (openjiuwen AssistantMessage)
        for attr in ("usage_metadata", "usage", "token_usage", "model_dump", "dict"):
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


def _meta_from_llm_result(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get("usage_metadata") or result.get("usage")
    return getattr(result, "usage_metadata", None) or getattr(result, "usage", None)


async def _on_llm_invoke_output(*, result: Any = None, **_kwargs: Any) -> None:
    bucket = _active_bucket.get()
    if bucket is None:
        return
    usage = usage_from_mapping(_meta_from_llm_result(result))
    if usage.prompt_tokens or usage.completion_tokens or usage.reasoning_tokens:
        bucket.append(usage)


def _ensure_llm_usage_hook() -> bool:
    """Register a process-wide openjiuwen LLM callback (idempotent)."""
    global _hook_registered
    with _hook_lock:
        if _hook_registered:
            return True
        try:
            from openjiuwen.core.runner import Runner
            from openjiuwen.core.runner.callback.events import LLMCallEvents
        except ImportError:
            return False
        # Only LLM_INVOKE_OUTPUT — LLM_OUTPUT also fires for the same call
        # and would double-count if both were registered.
        Runner.callback_framework.register_sync(
            LLMCallEvents.LLM_INVOKE_OUTPUT,
            _on_llm_invoke_output,
            namespace="hack_usage",
        )
        _hook_registered = True
        return True


@asynccontextmanager
async def capture_llm_usage() -> AsyncIterator[list[TokenUsage]]:
    """Accumulate per-call usage while openjiuwen Model.invoke runs in this task."""
    bucket: list[TokenUsage] = []
    hooked = _ensure_llm_usage_hook()
    token = _active_bucket.set(bucket)
    try:
        yield bucket
    finally:
        _active_bucket.reset(token)
        if not hooked:
            bucket.clear()


def sum_usage(parts: list[TokenUsage]) -> TokenUsage:
    total = TokenUsage()
    for part in parts:
        total = total + part
    return total