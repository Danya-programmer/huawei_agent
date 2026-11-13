"""ReactSolver registry wiring."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from harness.agents.solvers import ReactSolver, build_default_registry
from harness.domain.types import Result, Status, Task, TaskType


class TestReactSolver(unittest.TestCase):
    def test_registry_registers_all_types(self) -> None:
        registry = build_default_registry()
        for tt in TaskType:
            solver = registry.get(tt)
            self.assertIsInstance(solver, ReactSolver)

    def test_math_calls_task_react(self) -> None:
        solver = ReactSolver(max_iterations=3)
        fake = Result(status=Status.SOLVED, answer="4", cost=10, meta={"solver": "task_react"})
        task = Task(
            task_id="m",
            task_type=TaskType.MATH,
            instruction="2+2",
            reward=1,
            task_dir=Path("."),
        )
        with mock.patch(
            "harness.agents.solvers.react_solver.run_task_react", return_value=fake
        ) as react:
            out = solver.solve(task)
        react.assert_called_once()
        self.assertEqual(out.status, Status.SOLVED)
        self.assertEqual(out.answer, "4")

    def test_coding_calls_coding_react(self) -> None:
        solver = ReactSolver(max_iterations=3)
        fake = Result(status=Status.SOLVED, answer="DONE", cost=20, meta={"solver": "coding_react"})
        task = Task(
            task_id="c",
            task_type=TaskType.CODING,
            instruction="fix tests",
            reward=2,
            task_dir=Path("."),
        )
        with mock.patch(
            "harness.agents.solvers.react_solver.run_coding_react", return_value=fake
        ) as react:
            out = solver.solve(task)
        react.assert_called_once()
        self.assertEqual(out.answer, "DONE")


if __name__ == "__main__":
    unittest.main()
