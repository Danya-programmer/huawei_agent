"""Triage: structural skips and expected cost for budget-aware scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from harness.runtime.estimate import COST_SKIP
from harness.runtime.rank import assess
from harness.domain.types import Task, TaskType

Lane = Literal["react", "coding", "skip"]


@dataclass(frozen=True)
class TaskProfile:
    """Triage result for one task (no LLM, no template solvers)."""

    task_type: TaskType
    difficulty_proxy: str
    feasible: bool
    fail_reason: Optional[str]
    cost_lo: int
    cost_hi: int
    expected_cost: int
    lane: Lane
    priority: float = 0.0


def triage(task: Task) -> TaskProfile:
    """Score one task. Math always runs. Other types skip on a structural fail or p < tau."""
    if task.task_type not in {
        TaskType.MATH,
        TaskType.DATA_ANALYSIS,
        TaskType.SEARCH,
        TaskType.CODING,
    }:
        return TaskProfile(
            task_type=task.task_type,
            difficulty_proxy="unknown",
            feasible=False,
            fail_reason="unknown_task_type",
            cost_lo=COST_SKIP,
            cost_hi=COST_SKIP,
            expected_cost=COST_SKIP,
            lane="skip",
            priority=0.0,
        )

    estimate = assess(task)
    difficulty = _difficulty_proxy(task)
    if estimate.skip:
        lane: Lane = "skip"
    elif task.task_type == TaskType.CODING:
        lane = "coding"
    else:
        lane = "react"

    return TaskProfile(
        task_type=task.task_type,
        difficulty_proxy=difficulty,
        feasible=not estimate.skip,
        fail_reason=estimate.fail_reason,
        cost_lo=estimate.cost_hat,
        cost_hi=estimate.cost_hat,
        expected_cost=estimate.cost_hat,
        lane=lane,
        priority=estimate.priority,
    )


def _difficulty_proxy(task: Task) -> str:
    n = len(task.instruction)
    if n <= 200:
        return "easy"
    if n >= 800:
        return "hard"
    return "medium"
