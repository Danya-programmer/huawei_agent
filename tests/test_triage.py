"""Unit tests for ReAct-only triage lanes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.estimate import COST_CODING_REACT, COST_SHORT_REACT
from harness.scheduler import plan
from harness.tasks import load_task
from harness.triage import triage
from harness.types import TaskType


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


class TestTriageReactOnly(unittest.TestCase):
    def test_math_goes_to_react(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "m0", "math", "What is 2+2?", 3)
            profile = triage(load_task(root / "m0"))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "react")
        self.assertEqual(profile.expected_cost, COST_SHORT_REACT)

    def test_csv_goes_to_react_even_without_sales(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "d0", "data_analysis", "Sum column a in table.csv", 2)
            profile = triage(load_task(root / "d0"))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "react")

    def test_search_goes_to_react(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "s0", "search", "Who founded Granite AI?", 1)
            profile = triage(load_task(root / "s0"))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "react")

    def test_coding_lane(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "c0", "coding", "Fix mylib.py so tests pass.", 5)
            profile = triage(load_task(root / "c0"))
        self.assertTrue(profile.feasible)
        self.assertEqual(profile.lane, "coding")
        self.assertEqual(profile.expected_cost, COST_CODING_REACT)

    def test_plan_puts_react_before_coding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_task(root, "m0", "math", "x", 1)
            _write_task(root, "c0", "coding", "y", 5)
            tasks = [load_task(root / "m0"), load_task(root / "c0")]
            planned = plan(tasks)
        self.assertEqual([t.task_id for t in planned.phase_a], ["m0"])
        self.assertEqual([t.task_id for t in planned.phase_b], ["c0"])
        self.assertEqual(planned.skips, [])


if __name__ == "__main__":
    unittest.main()
