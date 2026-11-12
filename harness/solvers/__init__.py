"""Solvers: ReAct-only registry (no deterministic auto-solvers)."""

from __future__ import annotations

from harness.solver import SolverRegistry
from harness.solvers.react_solver import ReactSolver
from harness.types import TaskType


def build_default_registry() -> SolverRegistry:
    """Every task type goes through the ReAct agent."""
    registry = SolverRegistry()
    solver = ReactSolver(max_iterations=8)
    for task_type in TaskType:
        registry.register(task_type, solver)
    return registry


__all__ = [
    "ReactSolver",
    "build_default_registry",
]
