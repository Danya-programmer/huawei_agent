"""Solver protocol and registry — extension points for parallel tracks."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from harness.types import Result, Status, Task, TaskType


@runtime_checkable
class Solver(Protocol):
    """Contract every task solver must implement.

    Later agents plug in concrete solvers by registering them on
    :class:`SolverRegistry` (or by replacing the factory used by the
    scheduler). Signature::

        def solve(self, task: Task) -> Result: ...

    ``Result`` carries status, answer, cost, model, usage, error, meta.
    """

    def solve(self, task: Task) -> Result:
        """Attempt to solve ``task`` and return a structured Result."""
        ...


class NotImplementedSolver:
    """Placeholder solver used until real solvers are wired in."""

    def __init__(self, task_type: Optional[TaskType] = None) -> None:
        self.task_type = task_type

    def solve(self, task: Task) -> Result:
        label = self.task_type.value if self.task_type else task.task_type.value
        return Result(
            status=Status.SKIPPED,
            answer=None,
            cost=0,
            model=None,
            error=f"solver for {label!r} not implemented",
            meta={"reason": "not_implemented"},
        )


class SolverRegistry:
    """Map task_type -> Solver. Missing types fall back to a stub."""

    def __init__(self) -> None:
        self._solvers: dict[TaskType, Solver] = {}
        self._default: Solver = NotImplementedSolver()

    def register(self, task_type: TaskType, solver: Solver) -> None:
        self._solvers[task_type] = solver

    def get(self, task_type: TaskType) -> Solver:
        return self._solvers.get(task_type, self._default)

    def solve(self, task: Task) -> Result:
        return self.get(task.task_type).solve(task)

    @classmethod
    def with_stubs(cls) -> SolverRegistry:
        """Registry pre-filled with no-op stubs for all known task types."""
        registry = cls()
        for tt in TaskType:
            registry.register(tt, NotImplementedSolver(tt))
        return registry
