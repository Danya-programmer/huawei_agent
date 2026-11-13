"""Shared types for tasks, results, and model names."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TaskType(str, Enum):
    MATH = "math"
    DATA_ANALYSIS = "data_analysis"
    SEARCH = "search"
    CODING = "coding"


class Status(str, Enum):
    """Outcome of attempting a task."""

    SOLVED = "solved"
    SKIPPED = "skipped"
    UNSOLVABLE = "unsolvable"
    FAILED = "failed"
    BUDGET = "budget"
    ERROR = "error"


class ModelName(str, Enum):
    """Allowed proxy models (switchable per call)."""

    QWEN38_27B = "qwen3.8-27b"
    QWEN38_27B_NOREASON = "qwen3.8-27b-noreason"
    QWEN36_35B = "qwen3.6-35b-a3b"
    QWEN36_35B_NOREASON = "qwen3.6-35b-a3b-noreason"


# Contest default: cheaper noreason 3.8; override with LLM_MODEL.
DEFAULT_MODEL = ModelName.QWEN38_27B_NOREASON

_ALLOWED_MODELS = {m.value for m in ModelName}


def resolve_model(explicit: Optional[str] = None) -> str:
    """Pick proxy model: explicit arg → ``LLM_MODEL`` env → ``DEFAULT_MODEL``."""
    raw = (explicit or os.environ.get("LLM_MODEL") or "").strip()
    if not raw:
        return DEFAULT_MODEL.value
    if raw not in _ALLOWED_MODELS:
        allowed = ", ".join(sorted(_ALLOWED_MODELS))
        raise ValueError(f"unknown LLM_MODEL={raw!r}; allowed: {allowed}")
    return raw


@dataclass(frozen=True)
class Task:
    """One competition task loaded from task.json.

    Only task_id, task_type, instruction, and reward are required.
    Fields like expected/check appear in examples only and must not be required.
    """

    task_id: str
    task_type: TaskType
    instruction: str
    reward: int
    task_dir: Path
    # Optional extras from task.json (ignored by scoring contract).
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def work_dir(self) -> Path:
        """Directory the agent should treat as cwd (workspace/ if present)."""
        workspace = self.task_dir / "workspace"
        return workspace if workspace.is_dir() else self.task_dir

    @property
    def answer_path(self) -> Path:
        return self.task_dir / "answer.txt"


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class Result:
    """Return value of Solver.solve(task)."""

    status: Status
    answer: Optional[str] = None
    cost: int = 0
    model: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: Optional[str] = None
    # Free-form notes for the journal / debugging (e.g. reason for skip).
    meta: dict[str, Any] = field(default_factory=dict)
