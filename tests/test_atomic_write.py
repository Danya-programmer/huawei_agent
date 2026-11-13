"""Unit tests for atomic answer.txt writes and cost journal."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.infra.io import CostJournal, atomic_write_answer
from harness.domain.types import Status


class TestAtomicWriteAnswer(unittest.TestCase):
    def test_writes_answer_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = atomic_write_answer(tmp, "389")
            self.assertEqual(target, Path(tmp) / "answer.txt")
            self.assertEqual(target.read_text(encoding="utf-8"), "389\n")

    def test_overwrites_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            atomic_write_answer(tmp, "old")
            atomic_write_answer(tmp, "DONE")
            self.assertEqual(
                (Path(tmp) / "answer.txt").read_text(encoding="utf-8"),
                "DONE\n",
            )

    def test_no_leftover_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            atomic_write_answer(tmp, "42")
            leftovers = [
                p for p in Path(tmp).iterdir() if p.name.startswith(".answer.")
            ]
            self.assertEqual(leftovers, [])


class TestCostJournal(unittest.TestCase):
    def test_records_schema(self) -> None:
        journal = CostJournal(None)
        entry = journal.record(
            task_id="math_0000",
            reward=5,
            cost=12,
            model="qwen3.6-35b-a3b-noreason",
            status=Status.SOLVED,
            task_type="math",
            answer_written=True,
        )
        for key in ("task_id", "reward", "cost", "model", "status"):
            self.assertIn(key, entry)
        self.assertEqual(entry["status"], "solved")
        self.assertEqual(journal.total_cost(), 12)

    def test_mirrors_to_out_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = CostJournal(tmp)
            journal.record(
                task_id="t1",
                reward=1,
                cost=3,
                model=None,
                status=Status.SKIPPED,
            )
            journal.close()
            path = Path(tmp) / "cost_journal.jsonl"
            self.assertTrue(path.is_file())
            line = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(line["task_id"], "t1")
            self.assertEqual(line["cost"], 3)


if __name__ == "__main__":
    unittest.main()
