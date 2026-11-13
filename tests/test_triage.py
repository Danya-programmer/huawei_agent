"""Unit tests for ReAct-only triage lanes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.runtime.estimate import COST_SHORT_REACT
from harness.runtime.features import heuristic_cost
from harness.runtime.scheduler import plan
from harness.domain.tasks import load_task
from harness.runtime.triage import triage


def _write_task(
    root: Path,
    task_id: str,
    task_type: str,
    instruction: str,
    reward: int,
) -> Path:
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


def _valid_coding(task_dir: Path) -> None:
    work = task_dir / "workspace"
    work.mkdir(parents=True, exist_ok=True)
    (work / "mylib.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (work / "test_mylib.py").write_text("import mylib\nassert mylib.f() == 1\n", encoding="utf-8")


class TestTriageReactOnly(unittest.TestCase):
    def test_math_goes_to_react(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "m0", "math", "What is 2+2?", 3)
            profile = triage(load_task(root / "m0"))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "react")
        self.assertEqual(profile.expected_cost, COST_SHORT_REACT)

    def test_csv_without_table_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "d0", "data_analysis", "Sum column a in table.csv", 2)
            profile = triage(load_task(root / "d0"))
        self.assertFalse(profile.feasible)
        self.assertEqual(profile.lane, "skip")

    def test_csv_with_table_goes_to_react(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = _write_task(root, "d0", "data_analysis", "Sum column a in table.csv", 2)
            work = task_dir / "workspace"
            work.mkdir()
            (work / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            profile = triage(load_task(task_dir))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "react")

    def test_empty_kb_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "s0", "search", "Who founded Granite AI?", 1)
            profile = triage(load_task(root / "s0"))
        self.assertFalse(profile.feasible)
        self.assertEqual(profile.lane, "skip")

    def test_search_with_kb_goes_to_react(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = _write_task(root, "s0", "search", "Who founded Granite AI?", 1)
            kb = task_dir / "workspace" / "kb"
            kb.mkdir(parents=True)
            (kb / "note.md").write_text("Granite AI was founded in 2019.\n", encoding="utf-8")
            profile = triage(load_task(task_dir))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "react")

    def test_coding_lane(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = _write_task(root, "c0", "coding", "Fix mylib.py so tests pass.", 5)
            _valid_coding(task_dir)
            task = load_task(task_dir)
            profile = triage(task)
            self.assertTrue(profile.feasible)
            self.assertEqual(profile.lane, "coding")
            self.assertEqual(profile.expected_cost, heuristic_cost(task))

    def test_plan_orders_by_reward_per_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            math_dir = _write_task(root, "m0", "math", "x", 1)
            code_dir = _write_task(root, "c0", "coding", "y", 5)
            _valid_coding(code_dir)
            tasks = [load_task(math_dir), load_task(code_dir)]
            planned = plan(tasks)
        self.assertEqual([t.task_id for t in planned.phase_a], ["c0", "m0"])
        self.assertEqual(planned.phase_b, [])
        self.assertEqual(planned.skips, [])


if __name__ == "__main__":
    unittest.main()
