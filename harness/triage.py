"""Lightweight triage: cost bands for ReAct-only scheduling (no auto-solvers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from harness.estimate import COST_CODING_REACT, COST_SHORT_REACT, COST_SKIP
from harness.types import Task, TaskType

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


def triage(task: Task) -> TaskProfile:
    """Put every known task on the ReAct path; only unknown types are skipped."""
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
        )

    difficulty = _difficulty_proxy(task)
    if task.task_type == TaskType.CODING:
        cost = COST_CODING_REACT
        lane: Lane = "coding"
    else:
        cost = COST_SHORT_REACT
        lane = "react"

    return TaskProfile(
        task_type=task.task_type,
        difficulty_proxy=difficulty,
        feasible=True,
        fail_reason=None,
        cost_lo=cost,
        cost_hi=cost,
        expected_cost=cost,
        lane=lane,
    )


def _difficulty_proxy(task: Task) -> str:
    n = len(task.instruction)
    if n <= 200:
        return "easy"
    if n >= 800:
        return "hard"
    return "medium"
