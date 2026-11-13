"""Structural skips and the probability gate."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from harness.runtime.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from harness.runtime.rank import ConstantProba, assess, clear_cache
from harness.runtime.scheduler import SimpleScheduler
from harness.agents.solver import SolverRegistry
from harness.domain.tasks import load_task
from harness.runtime.triage import triage
from harness.domain.types import Result, Status, Task, TaskType


def _task(root: Path, task_id: str, task_type: str, instruction: str, reward: int) -> Path:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "task_type": task_type,
                "instruction": instruction,
                "reward": reward,
            }
        ),
        encoding="utf-8",
    )
    return task_dir


class TestStructuralSkips(unittest.TestCase):
    def test_math_without_numbers_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            task_dir = _task(
                Path(td),
                "m0",
                "math",
                "What is the final number?",
                4,
            )
            profile = triage(load_task(task_dir))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "react")
        self.assertIsNone(profile.fail_reason)

    def test_syntax_error_in_mylib_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            task_dir = _task(Path(td), "c0", "coding", "Fix mylib.py.", 3)
            work = task_dir / "workspace"
            work.mkdir()
            (work / "mylib.py").write_text("def f(\n    return\n", encoding="utf-8")
            (work / "test_mylib.py").write_text(
                "import mylib\nassert mylib.f() == 1\n",
                encoding="utf-8",
            )
            profile = triage(load_task(task_dir))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "coding")

    def test_foreign_import_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            task_dir = _task(Path(td), "c0", "coding", "Fix mylib.py.", 3)
            work = task_dir / "workspace"
            work.mkdir()
            (work / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
            (work / "test_mylib.py").write_text(
                "import otherlib as mylib\nassert mylib.f() == 1\n",
                encoding="utf-8",
            )
            profile = triage(load_task(task_dir))
        self.assertFalse(profile.feasible)
        self.assertIn("coding_foreign_import", profile.fail_reason or "")

    def test_empty_kb_does_not_write_answer(self) -> None:
        class _Solver:
            def solve(self, task: Task) -> Result:
                raise AssertionError("skipped task must not be solved")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = _task(root, "s0", "search", "Who founded Granite AI?", 2)
            registry = SolverRegistry()
            registry.register(TaskType.SEARCH, _Solver())
            summary = SimpleScheduler(solvers=registry, journal=None).run(root)
        self.assertEqual(summary.solved, 0)
        self.assertEqual(summary.skipped, 1)
        self.assertFalse((task_dir / "answer.txt").exists())


class TestProbabilityGate(unittest.TestCase):
    def test_zero_probability_skips_table_but_not_math(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = root / "models"
            models.mkdir()
            blob = {
                "model": ConstantProba(0.0),
                "feature_names": FEATURE_NAMES,
                "schema_version": FEATURE_SCHEMA_VERSION,
            }
            with (models / "solvability.joblib").open("wb") as fh:
                pickle.dump(blob, fh)
            (models / "tau.json").write_text(json.dumps({"tau": 0.5}), encoding="utf-8")
            clear_cache()

            math_dir = _task(root, "m0", "math", "What is the final number?", 5)
            data_dir = _task(root, "d0", "data_analysis", "Read sales.csv and sum it.", 5)
            work = data_dir / "workspace"
            work.mkdir()
            (work / "sales.csv").write_text("a\n1\n", encoding="utf-8")

            math_est = assess(load_task(math_dir), models=models)
            data_est = assess(load_task(data_dir), models=models)

        self.assertFalse(math_est.skip)
        self.assertEqual(math_est.p_solvable, 1.0)
        self.assertTrue(data_est.skip)
        self.assertEqual(data_est.fail_reason, "low_p")


if __name__ == "__main__":
    unittest.main()
