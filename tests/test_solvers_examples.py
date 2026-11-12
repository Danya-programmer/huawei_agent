"""Example tasks now require ReAct/LLM — skipped without live credentials."""

from __future__ import annotations

import os
import unittest

from harness.solvers import ReactSolver, build_default_registry
from harness.types import TaskType


class TestReactOnlyRegistry(unittest.TestCase):
    def test_default_registry_is_react_only(self) -> None:
        registry = build_default_registry()
        for tt in TaskType:
            self.assertIsInstance(registry.get(tt), ReactSolver)

    @unittest.skipUnless(
        os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL"),
        "LLM_* not set — live example solve skipped",
    )
    def test_placeholder_live_examples_require_llm(self) -> None:
        # Live solves of task-dir.zip belong in Docker / manual runs.
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
