"""Quota and manifest checks for harness.generators (gen-100)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.generators import (
    TASK_TYPES,
    generate_training_set,
    plan_slots,
)


class TestGeneratorQuotas(unittest.TestCase):
    def test_plan_slots_n100(self) -> None:
        slots = plan_slots(100, 0.2)
        self.assertEqual(len(slots), 100)
        by_type = {t: 0 for t in TASK_TYPES}
        by_diff = {"easy": 0, "medium": 0, "hard": 0}
        unsolvable_by_type = {t: 0 for t in TASK_TYPES}
        for s in slots:
            by_type[s.task_type] += 1
            if s.solvable:
                by_diff[s.difficulty] += 1
            else:
                unsolvable_by_type[s.task_type] += 1
                self.assertIsNotNone(s.fail_mode)
        self.assertEqual(by_type, {t: 25 for t in TASK_TYPES})
        self.assertEqual(sum(unsolvable_by_type.values()), 20)
        self.assertEqual(unsolvable_by_type, {t: 5 for t in TASK_TYPES})
        self.assertEqual(by_diff, {"easy": 40, "medium": 25, "hard": 15})

    def test_generate_n100_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "train-100"
            generate_training_set(out, n=100, seed=42, unsolvable_ratio=0.2)

            manifest = out / "manifest.jsonl"
            self.assertTrue(manifest.is_file())
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 100)

            by_type = {t: 0 for t in TASK_TYPES}
            by_diff = {"easy": 0, "medium": 0, "hard": 0}
            solvable = 0
            unsolvable = 0
            required = {
                "task_id",
                "task_type",
                "difficulty",
                "solvable",
                "expected",
                "reward",
                "fail_mode",
            }
            for row in rows:
                self.assertEqual(set(row.keys()), required)
                by_type[row["task_type"]] += 1
                task_json = out / row["task_id"] / "task.json"
                self.assertTrue(task_json.is_file())
                payload = json.loads(task_json.read_text(encoding="utf-8"))
                for key in ("task_id", "task_type", "instruction", "reward"):
                    self.assertIn(key, payload)
                self.assertEqual(payload["expected"], row["expected"])
                if row["solvable"]:
                    solvable += 1
                    by_diff[row["difficulty"]] += 1
                    self.assertIsNone(row["fail_mode"])
                else:
                    unsolvable += 1
                    self.assertEqual(row["difficulty"], "unsolvable")
                    self.assertEqual(row["expected"], "UNSOLVABLE")
                    self.assertIsNotNone(row["fail_mode"])

            self.assertEqual(by_type, {t: 25 for t in TASK_TYPES})
            self.assertEqual(solvable, 80)
            self.assertEqual(unsolvable, 20)
            self.assertEqual(by_diff, {"easy": 40, "medium": 25, "hard": 15})

            # Idempotent: second run still yields exactly 100 task dirs + manifest.
            generate_training_set(out, n=100, seed=42, unsolvable_ratio=0.2)
            task_dirs = [p for p in out.iterdir() if p.is_dir()]
            self.assertEqual(len(task_dirs), 100)
            self.assertTrue((out / "manifest.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
