"""Quota helpers for external dataset builder (no HF download)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_builder():
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_external_datasets.py"
    spec = importlib.util.spec_from_file_location("build_external_datasets", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ExternalQuotaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_builder()

    def test_plan_slots_100(self):
        slots = self.mod.plan_slots(100, 0.2)
        self.mod.assert_quotas(slots, 100, 0.2)
        self.assertEqual(len(slots), 100)

    def test_plan_slots_3000(self):
        slots = self.mod.plan_slots(3000, 0.2)
        self.mod.assert_quotas(slots, 3000, 0.2)
        self.assertEqual(sum(1 for s in slots if not s.solvable), 600)
        self.assertEqual(sum(1 for s in slots if s.task_type == "math"), 750)


if __name__ == "__main__":
    unittest.main()
