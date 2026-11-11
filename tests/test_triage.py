"""Unit tests for triage lanes and two-phase ordering (synthetic tasks)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.estimate import (
    COST_CODING_HEURISTIC,
    COST_DETERMINISTIC,
    COST_SHORT_REACT,
    COST_SKIP,
)
from harness.scheduler import SimpleScheduler, plan
from harness.tasks import load_task
from harness.triage import triage
from harness.types import Task, TaskType


_MATH_OK = (
    "The warehouse starts with 100 boxes. "
    "It ships 3 boxes per day for 10 days. "
    "It receives 2 deliveries of 5 boxes each."
)


def _write_task(
    root: Path,
    task_id: str,
    task_type: str,
    instruction: str,
    reward: int,
    *,
    workspace: bool = True,
) -> Task:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    work = task_dir / "workspace" if workspace else task_dir
    if workspace:
        work.mkdir()
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
    return load_task(task_dir)


class TestTriageLanes(unittest.TestCase):
    def test_math_cheap_when_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(Path(tmp), "m1", "math", _MATH_OK, 2)
            profile = triage(task)
            self.assertEqual(profile.lane, "cheap")
            self.assertTrue(profile.feasible)
            self.assertEqual(profile.expected_cost, COST_DETERMINISTIC)
            self.assertIsNone(profile.fail_reason)

    def test_math_needs_react_when_unparsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(
                Path(tmp),
                "m2",
                "math",
                "Compute the mystic warehouse karma of 42.",
                5,
            )
            profile = triage(task)
            self.assertEqual(profile.lane, "needs_react_fallback")
            self.assertTrue(profile.feasible)
            self.assertEqual(profile.expected_cost, COST_SHORT_REACT)
            self.assertEqual(profile.fail_reason, "math_parse_failed")

    def test_csv_skip_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(
                Path(tmp),
                "c1",
                "data_analysis",
                "How many units of product 'X' were sold in months 1 through 2?",
                1,
            )
            profile = triage(task)
            self.assertEqual(profile.lane, "skip")
            self.assertFalse(profile.feasible)
            self.assertEqual(profile.expected_cost, COST_SKIP)
            self.assertEqual(profile.fail_reason, "missing_sales_csv")

    def test_csv_cheap_when_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(
                Path(tmp),
                "c2",
                "data_analysis",
                "How many units of product 'Widget' were sold in months 1 through 3?",
                3,
            )
            csv_path = task.work_dir / "sales.csv"
            csv_path.write_text(
                "order_id,region,product,month,units,price\n"
                "1,EU,Widget,1,4,1.0\n"
                "2,EU,Widget,2,6,1.0\n",
                encoding="utf-8",
            )
            profile = triage(task)
            self.assertEqual(profile.lane, "cheap")
            self.assertTrue(profile.feasible)
            self.assertEqual(profile.expected_cost, COST_DETERMINISTIC)

    def test_search_skip_empty_kb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(
                Path(tmp),
                "s1",
                "search",
                "In which year was Ada Lovelace born?",
                2,
            )
            (task.work_dir / "kb").mkdir()
            profile = triage(task)
            self.assertEqual(profile.lane, "skip")
            self.assertFalse(profile.feasible)
            self.assertEqual(profile.fail_reason, "kb_empty_or_missing")

    def test_search_cheap_born(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(
                Path(tmp),
                "s2",
                "search",
                "In which year was Ada Lovelace born?",
                2,
            )
            kb = task.work_dir / "kb"
            kb.mkdir()
            (kb / "person_ada.md").write_text(
                "# Ada Lovelace\nBorn: 1815\nEmployer: Analytical Engine Co\n",
                encoding="utf-8",
            )
            profile = triage(task)
            self.assertEqual(profile.lane, "cheap")
            self.assertTrue(profile.feasible)

    def test_coding_skip_missing_lib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(
                Path(tmp),
                "k1",
                "coding",
                "Fix mylib.py until tests pass.",
                4,
            )
            (task.work_dir / "test_mylib.py").write_text(
                "import unittest\nimport mylib as m\n"
                "class T(unittest.TestCase):\n"
                "    def test_total(self):\n"
                "        self.assertEqual(m.total([1,2]), 3)\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n",
                encoding="utf-8",
            )
            profile = triage(task)
            self.assertEqual(profile.lane, "skip")
            self.assertEqual(profile.fail_reason, "missing_mylib")

    def test_coding_lane_when_files_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = _write_task(
                Path(tmp),
                "k2",
                "coding",
                "Fix mylib.py until tests pass.",
                4,
            )
            (task.work_dir / "mylib.py").write_text(
                "def total(xs):\n    return 0\n",
                encoding="utf-8",
            )
            (task.work_dir / "test_mylib.py").write_text(
                "import unittest\nimport mylib as m\n"
                "class T(unittest.TestCase):\n"
                "    def test_total(self):\n"
                "        self.assertEqual(m.total([1,2]), 3)\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n",
                encoding="utf-8",
            )
            profile = triage(task)
            self.assertEqual(profile.lane, "coding")
            self.assertTrue(profile.feasible)
            self.assertEqual(profile.expected_cost, COST_CODING_HEURISTIC)


class TestTwoPhaseOrdering(unittest.TestCase):
    def test_plan_phase_a_before_b_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Expensive-looking coding (high reward) should still be phase B.
            coding = _write_task(
                root,
                "code_hi",
                "coding",
                "Fix mylib.py until tests pass.",
                5,
            )
            (coding.work_dir / "mylib.py").write_text(
                "def total(xs):\n    return 0\n", encoding="utf-8"
            )
            (coding.work_dir / "test_mylib.py").write_text(
                "import unittest\nimport mylib as m\n"
                "class T(unittest.TestCase):\n"
                "    def test_total(self):\n"
                "        self.assertEqual(m.total([1]), 1)\n"
                "if __name__ == '__main__':\n    unittest.main()\n",
                encoding="utf-8",
            )
            # Cheap math with lower reward than coding.
            math_lo = _write_task(root, "math_lo", "math", _MATH_OK, 1)
            # Unsolvable csv → skip.
            skip_csv = _write_task(
                root,
                "csv_bad",
                "data_analysis",
                "How many units of product 'Z' were sold in months 1 through 2?",
                5,
            )
            # Math fallback (needs react) → phase B.
            math_fb = _write_task(
                root,
                "math_fb",
                "math",
                "Totally unknown math puzzle.",
                4,
            )
            # Another cheap math with higher reward — same cost, higher reward first.
            math_hi = _write_task(
                root,
                "math_hi",
                "math",
                _MATH_OK.replace("100", "200"),
                3,
            )

            tasks = [coding, math_lo, skip_csv, math_fb, math_hi]
            planned = plan(tasks)

            self.assertEqual([t.task_id for t in planned.skips], ["csv_bad"])
            self.assertEqual(
                [t.task_id for t in planned.phase_a],
                ["math_hi", "math_lo"],  # same cost, higher reward first
            )
            # Phase B: sorted by -reward/expected_cost
            # coding: 5/1=5, math_fb: 4/800=0.005 → coding first
            self.assertEqual(
                [t.task_id for t in planned.phase_b],
                ["code_hi", "math_fb"],
            )
            ordered = SimpleScheduler().order_tasks(tasks)
            self.assertEqual(
                [t.task_id for t in ordered],
                ["math_hi", "math_lo", "code_hi", "math_fb"],
            )

    def test_budget_blocks_expensive_estimate(self) -> None:
        from harness.solver import SolverRegistry
        from harness.types import Result, Status

        class _OkSolver:
            def solve(self, task: Task) -> Result:
                return Result(status=Status.SOLVED, answer="1", cost=0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            math_ok = _write_task(root, "math_ok", "math", _MATH_OK, 1)
            math_fb = _write_task(
                root, "math_fb", "math", "Unknown warehouse riddle.", 5
            )
            registry = SolverRegistry()
            registry.register(TaskType.MATH, _OkSolver())
            # Budget too small for short react (800).
            summary = SimpleScheduler(
                solvers=registry,
                journal=None,
                budget=10,
            ).run(root)
            by_id = {t.task_id: r for t, r in summary.results}
            self.assertEqual(by_id["math_ok"].status, Status.SOLVED)
            self.assertEqual(by_id["math_fb"].status, Status.BUDGET)


class TestReactDefault(unittest.TestCase):
    def test_react_enabled_defaults_from_llm_env(self) -> None:
        from harness.solvers.coding import react_enabled
        import os

        old = {
            k: os.environ.get(k)
            for k in ("HARNESS_USE_REACT", "LLM_API_KEY", "LLM_BASE_URL")
        }
        try:
            os.environ.pop("HARNESS_USE_REACT", None)
            os.environ["LLM_API_KEY"] = "x"
            os.environ["LLM_BASE_URL"] = "http://localhost"
            self.assertTrue(react_enabled())
            os.environ["HARNESS_USE_REACT"] = "0"
            self.assertFalse(react_enabled())
            os.environ["HARNESS_USE_REACT"] = "false"
            self.assertFalse(react_enabled())
            os.environ.pop("HARNESS_USE_REACT", None)
            os.environ.pop("LLM_API_KEY", None)
            self.assertFalse(react_enabled())
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
