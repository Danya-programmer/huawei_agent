"""Validate / sanitize answer.txt contents before claiming SOLVED."""

from __future__ import annotations

import re
from typing import Optional

_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)
_TOOL_XML_RE = re.compile(r"</?(?:arg_value|tool_call|function_call|invoke)\b", re.I)
_CMD_LINE_RE = re.compile(
    r"^(?:python3?|bash|sh|echo|find|ls|cat|cd|pip|sudo|export|curl|wget)\b",
    re.I | re.M,
)
_REDIRECT_RE = re.compile(r">\s*answer\.txt\b", re.I)
_CODEY_RE = re.compile(
    r"\b(?:import\s+\w+|from\s+\w+\s+import|def\s+\w+\s*\(|print\s*\()",
    re.I,
)
_CHATTY_RE = re.compile(
    r"^(?:let me|i'll|i will|here (?:a|is|are)|looking at|starting by|"
    r"first[, ]|okay[, ]|sure[, ]|i need to)\b",
    re.I | re.M,
)


def looks_like_garbage(text: str) -> bool:
    """True when ``text`` is a tool command, markdown blob, or reasoning dump."""
    raw = (text or "").strip()
    if not raw:
        return True
    if raw in {"```", "`", "..."}:
        return True
    if _TOOL_XML_RE.search(raw):
        return True
    if _REDIRECT_RE.search(raw):
        return True
    if _CMD_LINE_RE.search(raw):
        return True
    if _CODEY_RE.search(raw) and "DONE" not in raw.upper():
        return True
    if raw.startswith("```") or raw.endswith("```"):
        # Pure fence or fenced block without a short payload.
        inner = _FENCE_RE.sub("", raw).strip()
        if not inner or looks_like_garbage(inner):
            return True
    if len(raw) > 400:
        return True
    if _CHATTY_RE.search(raw):
        return True
    # Multi-line dumps that aren't a short final value.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 4:
        return True
    return False


def sanitize_answer(text: Optional[str], *, coding: bool = False) -> Optional[str]:
    """Return a contest-ready answer string, or None if unusable."""
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None

    if coding:
        if "DONE" in raw.upper() and not looks_like_garbage(raw.replace("DONE", "x")):
            # Allow DONE even inside short noise; prefer exact token.
            return "DONE"
        if raw.strip().upper() == "DONE":
            return "DONE"
        if looks_like_garbage(raw):
            return None
        return raw if raw.upper() == "DONE" else None

    # Strip markdown fences.
    cleaned = _FENCE_RE.sub("", raw).strip()
    if not cleaned:
        return None

    if looks_like_garbage(cleaned):
        # Last non-empty line sometimes holds the value after reasoning.
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if not lines:
            return None
        candidate = lines[-1]
        # Strip common "Answer:" prefixes.
        candidate = re.sub(r"^(?:final\s+)?answer\s*[:=]\s*", "", candidate, flags=re.I)
        candidate = candidate.strip().strip("`").strip('"').strip("'")
        if candidate and not looks_like_garbage(candidate) and len(candidate) <= 200:
            return candidate
        return None

    # Prefer last line if short multi-line (e.g. "…\n42").
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if len(lines) > 1:
        last = lines[-1]
        last = re.sub(r"^(?:final\s+)?answer\s*[:=]\s*", "", last, flags=re.I).strip()
        if last and not looks_like_garbage(last) and len(last) <= 200:
            return last
    return cleaned if len(cleaned) <= 400 else None
