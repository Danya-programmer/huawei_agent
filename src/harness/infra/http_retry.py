"""Tiny retry helper for LLM proxy rate limits (429) and transient errors."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 4,
    base_delay_s: float = 1.5,
    is_retryable: Callable[[BaseException], bool] | None = None,
) -> T:
    """Call ``fn``; on retryable errors sleep with exponential backoff."""
    check = is_retryable or _default_retryable
    last: BaseException | None = None
    attempts = max(1, int(retries) + 1)
    for i in range(attempts):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001
            last = exc
            if i >= attempts - 1 or not check(exc):
                raise
            delay = base_delay_s * (2**i)
            time.sleep(delay)
    assert last is not None
    raise last


def _default_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return any(
        needle in msg or needle in name
        for needle in ("429", "too many requests", "timeout", "temporarily", "503", "502")
    )


def error_is_rate_limit(exc: BaseException | str) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg
