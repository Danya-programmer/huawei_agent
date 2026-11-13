"""Single ReAct solver for every contest task type (no deterministic path)."""

from __future__ import annotations

from harness.agents.react_agent import CODING_SYSTEM_PROMPT, run_coding_react, run_task_react
from harness.domain.types import Result, Task, TaskType


class ReactSolver:
    """Always call the openJiuwen ReAct agent (or chat fallback)."""

    def __init__(self, *, max_iterations: int = 8) -> None:
        self.max_iterations = max_iterations

    def solve(self, task: Task) -> Result:
        if task.task_type == TaskType.CODING:
            return run_coding_react(task, max_iterations=self.max_iterations)
        return run_task_react(
            task,
            max_iterations=self.max_iterations,
            system_prompt=_prompt_for(task.task_type),
        )


def _prompt_for(task_type: TaskType) -> str:
    if task_type == TaskType.CODING:
        return CODING_SYSTEM_PROMPT
    # General prompt already covers math / csv / search.
    from harness.agents.react_agent import GENERAL_SYSTEM_PROMPT

    return GENERAL_SYSTEM_PROMPT
