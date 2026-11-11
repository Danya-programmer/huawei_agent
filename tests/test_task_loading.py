"""Unit tests for task.json loading and discovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.io import discover_tasks, load_task, resolve_tasks_dir
from harness.types import TaskType


class TestLoadTask(unittest.TestCase):
    def _write(self, directory: Path, payload: dict) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "task.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return directory

    def test_loads_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = self._write(
                Path(tmp) / "math_0000",
                {
                    "task_id": "math_0000",
                    "task_type": "math",
                    "instruction": "compute",
                    "reward": 5,
                },
            )
            task = load_task(task_dir)
            self.assertEqual(task.task_id, "math_0000")
            self.assertEqual(task.task_type, TaskType.MATH)
            self.assertEqual(task.instruction, "compute")
            self.assertEqual(task.reward, 5)
            self.assertEqual(task.extras, {})

    def test_ignores_example_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = self._write(
                Path(tmp) / "search_0001",
                {
                    "task_id": "search_0001",
                    "task_type": "search",
                    "instruction": "find",
                    "reward": 1,
                    "expected": "1983",
                    "check": "exact_str",
                },
            )
            task = load_task(task_dir)
            self.assertEqual(task.extras["expected"], "1983")
            self.assertEqual(task.extras["check"], "exact_str")

    def test_work_dir_prefers_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = self._write(
                Path(tmp) / "coding_0003",
                {
                    "task_id": "coding_0003",
                    "task_type": "coding",
                    "instruction": "fix",
                    "reward": 2,
                },
            )
            (task_dir / "workspace").mkdir()
            task = load_task(task_dir)
            self.assertEqual(task.work_dir, (task_dir / "workspace").resolve())

    def test_missing_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = self._write(
                Path(tmp) / "bad",
                {"task_id": "bad", "task_type": "math", "instruction": "x"},
            )
            with self.assertRaises(ValueError):
                load_task(task_dir)

    def test_discover_skips_non_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root / "math_0000",
                {
                    "task_id": "math_0000",
                    "task_type": "math",
                    "instruction": "a",
                    "reward": 1,
                },
            )
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            (root / "empty_dir").mkdir()
            tasks = discover_tasks(root)
            self.assertEqual([t.task_id for t in tasks], ["math_0000"])

    def test_resolve_tasks_dir_prefers_cli(self) -> None:
        path = resolve_tasks_dir("/data/tasks", env={"TASKS": "/other"})
        self.assertEqual(path, Path("/data/tasks"))

    def test_resolve_tasks_dir_env_fallback(self) -> None:
        path = resolve_tasks_dir(None, env={"TASKS": "/from/env"})
        self.assertEqual(path, Path("/from/env"))


if __name__ == "__main__":
    unittest.main()
