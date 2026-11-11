"""Smoke test: stub scheduler discovers tasks and writes answers when solved."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.scheduler import SimpleScheduler
from harness.solver import SolverRegistry
from harness.types import Result, Status, Task, TaskType


class _EchoSolver:
    def solve(self, task: Task) -> Result:
        return Result(
            status=Status.SOLVED,
            answer="42",
            cost=0,
            model=None,
        )


class TestScheduler(unittest.TestCase):
    def test_writes_answer_on_solved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "math_x"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "math_x",
                        "task_type": "math",
                        "instruction": "x",
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


if __name__ == "__main__":
    unittest.main()
