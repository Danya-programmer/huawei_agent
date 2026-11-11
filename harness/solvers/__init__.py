"""Concrete solvers for competition task types."""

from __future__ import annotations

from harness.solver import SolverRegistry
from harness.solvers.coding import CodingSolver
from harness.solvers.csv_solver import CsvSolver
from harness.solvers.math_solver import MathSolver
from harness.solvers.search_solver import SearchSolver
from harness.types import TaskType


def build_default_registry() -> SolverRegistry:
    """Registry with all production solvers wired in."""
    registry = SolverRegistry()
    registry.register(TaskType.MATH, MathSolver())
    registry.register(TaskType.DATA_ANALYSIS, CsvSolver())
    registry.register(TaskType.SEARCH, SearchSolver())
    registry.register(TaskType.CODING, CodingSolver())
    return registry


__all__ = [
    "CodingSolver",
    "CsvSolver",
    "MathSolver",
    "SearchSolver",
    "build_default_registry",
]
