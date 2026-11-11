"""Validate deterministic solvers on official example tasks."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from harness.solvers import build_default_registry
from harness.tasks import load_task
from harness.types import Status

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "task-dir.zip"


@unittest.skipUnless(ZIP.is_file(), "task-dir.zip missing")
class TestExampleSolvers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(ZIP) as zf:
            zf.extractall(cls._tmp.name)
        cls.tasks_root = Path(cls._tmp.name) / "task-dir"
        cls.registry = build_default_registry()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _check(self, task_id: str) -> None:
        task_dir = self.tasks_root / task_id
        # Copy so coding solver can mutate mylib without poisoning other runs.
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / task_id
            shutil.copytree(task_dir, dest)
            task = load_task(dest)
            expected = task.extras.get("expected")
            result = self.registry.solve(task)
            self.assertEqual(result.status, Status.SOLVED, result.error)
            self.assertIsNotNone(result.answer)
            if expected == "tests_pass":
                self.assertEqual(result.answer, "DONE")
            else:
                self.assertEqual(result.answer, str(expected))

    def test_math_0000(self) -> None:
        self._check("math_0000")

    def test_math_0004(self) -> None:
        self._check("math_0004")

    def test_data_analysis_0002(self) -> None:
        self._check("data_analysis_0002")

    def test_data_analysis_0006(self) -> None:
        self._check("data_analysis_0006")

    def test_search_0001(self) -> None:
        self._check("search_0001")

    def test_search_0005(self) -> None:
        self._check("search_0005")

    def test_coding_0003(self) -> None:
        self._check("coding_0003")

    def test_coding_0007(self) -> None:
        self._check("coding_0007")


if __name__ == "__main__":
    unittest.main()
