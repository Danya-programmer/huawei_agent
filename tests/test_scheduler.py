"""Smoke test: stub scheduler discovers tasks and writes answers when solved."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.runtime.scheduler import SimpleScheduler
from harness.agents.solver import SolverRegistry
from harness.domain.types import Result, Status, Task, TaskType

import rank_isolation


class _EchoSolver:
    def solve(self, task: Task) -> Result:
        return Result(
            status=Status.SOLVED,
            answer="42",
            cost=0,
            model=None,
        )


_MATH_OK = (
    "The warehouse starts with 100 boxes. "
    "It ships 3 boxes per day for 10 days. "
    "It receives 2 deliveries of 5 boxes each."
)


class TestScheduler(unittest.TestCase):
    def test_writes_answer_on_solved(self) -> None:
        with rank_isolation.no_rank_models(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "math_x"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "math_x",
                        "task_type": "math",
                        "instruction": _MATH_OK,
                        "reward": 1,
                    }
                ),
                encoding="utf-8",
            )
            registry = SolverRegistry()
            registry.register(TaskType.MATH, _EchoSolver())
            summary = SimpleScheduler(solvers=registry, journal=None).run(root)
            self.assertEqual(summary.solved, 1)
            self.assertEqual(
                (task_dir / "answer.txt").read_text(encoding="utf-8"),
                "42\n",
            )

    def test_react_lane_calls_solver(self) -> None:
        calls: list[str] = []

        class _CountingSolver:
            def solve(self, task: Task) -> Result:
                calls.append(task.task_id)
                return Result(status=Status.SOLVED, answer="x", cost=0)

        with rank_isolation.no_rank_models(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "csv_any"
            work = task_dir / "workspace"
            work.mkdir(parents=True)
            (work / "table.csv").write_text("units\n1\n2\n", encoding="utf-8")
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "csv_any",
                        "task_type": "data_analysis",
                        "instruction": "Sum column units in table.csv",
                        "reward": 2,
                    }
                ),
                encoding="utf-8",
            )
            registry = SolverRegistry()
            registry.register(TaskType.DATA_ANALYSIS, _CountingSolver())
            summary = SimpleScheduler(solvers=registry, journal=None).run(root)
            self.assertEqual(calls, ["csv_any"])
            self.assertEqual(summary.solved, 1)


if __name__ == "__main__":
    unittest.main()
