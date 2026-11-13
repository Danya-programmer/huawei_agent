"""Unit tests for jury-like score_pack coding / answer normalization."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "score_pack", _ROOT / "scripts" / "score_pack.py"
)
assert _SPEC and _SPEC.loader
_score_pack = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_score_pack)
_norm = _score_pack._norm
_score_coding = _score_pack._score_coding
_verdict = _score_pack._verdict
score_pack = _score_pack.score_pack


def _write_coding_task(
    root: Path,
    tid: str,
    *,
    solvable: bool,
    mylib: str,
    tests: str,
    answer: str | None = None,
    reward: int = 3,
) -> None:
    task_dir = root / tid
    ws = task_dir / "workspace"
    ws.mkdir(parents=True)
    (ws / "mylib.py").write_text(mylib, encoding="utf-8")
    (ws / "test_mylib.py").write_text(tests, encoding="utf-8")
    payload = {
        "task_id": tid,
        "task_type": "coding",
        "instruction": "fix",
        "reward": reward,
        "expected": "DONE" if solvable else "UNSOLVABLE",
    }
    if not solvable:
        # score_pack uses manifest.solvable; also expected UNSOLVABLE
        pass
    (task_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    if answer is not None:
        (task_dir / "answer.txt").write_text(answer, encoding="utf-8")


def _manifest_row(
    tid: str,
    *,
    task_type: str = "coding",
    solvable: bool = True,
    expected: str = "DONE",
    reward: int = 3,
    difficulty: str = "easy",
) -> dict:
    return {
        "task_id": tid,
        "task_type": task_type,
        "solvable": solvable,
        "expected": expected,
        "reward": reward,
        "difficulty": difficulty,
    }


class NormQuotesTests(unittest.TestCase):
    def test_strip_outer_double_quotes(self) -> None:
        self.assertEqual(
            _norm('"Move Your Body"', '"Move Your Body"'),
            _norm("Move Your Body", '"Move Your Body"'),
        )

    def test_numeric_still_normalized(self) -> None:
        self.assertEqual(_norm("1,000", "1000"), "1000")


class ScoreCodingUnitTests(unittest.TestCase):
    def test_solvable_green_tests_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_coding_task(
                root,
                "c_ok",
                solvable=True,
                mylib="def add(a, b):\n    return a + b\n",
                tests=(
                    "import unittest\nimport mylib\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(mylib.add(1, 2), 3)\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
                answer=None,
            )
            v, ok, _ = _score_coding(
                root / "c_ok", solvable=True, expected="DONE", got=None
            )
            self.assertEqual(v, "correct")
            self.assertTrue(ok)

    def test_solvable_done_only_red_tests_not_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_coding_task(
                root,
                "c_fake",
                solvable=True,
                mylib="def add(a, b):\n    return a - b\n",
                tests=(
                    "import unittest\nimport mylib\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(mylib.add(1, 2), 3)\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
                answer="DONE\n",
            )
            v, ok, _ = _score_coding(
                root / "c_fake", solvable=True, expected="DONE", got="DONE"
            )
            self.assertEqual(v, "wrong")
            self.assertFalse(ok)

    def test_unsolvable_empty_answer_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_coding_task(
                root,
                "c_un",
                solvable=False,
                mylib="def add(a, b):\n    return a - b\n",
                tests=(
                    "import unittest\nimport mylib\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(mylib.add(1, 2), 3)\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
                answer=None,
            )
            v, ok, _ = _score_coding(
                root / "c_un",
                solvable=False,
                expected="UNSOLVABLE",
                got=None,
            )
            self.assertEqual(v, "unsolvable_ok")
            self.assertFalse(ok)


class ScorePackIntegrationTests(unittest.TestCase):
    def test_pack_coding_and_quote_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "pack"
            pack.mkdir()
            _write_coding_task(
                pack,
                "coding_ok",
                solvable=True,
                mylib="def f():\n    return 1\n",
                tests=(
                    "import unittest\nimport mylib\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_f(self):\n"
                    "        self.assertEqual(mylib.f(), 1)\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
            )
            _write_coding_task(
                pack,
                "coding_fake",
                solvable=True,
                mylib="def f():\n    return 0\n",
                tests=(
                    "import unittest\nimport mylib\n\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_f(self):\n"
                    "        self.assertEqual(mylib.f(), 1)\n\n"
                    "if __name__ == '__main__':\n"
                    "    unittest.main()\n"
                ),
                answer="DONE\n",
            )
            da = pack / "da_quote"
            da.mkdir()
            (da / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "da_quote",
                        "task_type": "data_analysis",
                        "instruction": "x",
                        "reward": 5,
                        "expected": '"Move Your Body"',
                    }
                ),
                encoding="utf-8",
            )
            (da / "answer.txt").write_text("Move Your Body\n", encoding="utf-8")

            rows = [
                _manifest_row("coding_ok", reward=2),
                _manifest_row("coding_fake", reward=2),
                _manifest_row(
                    "da_quote",
                    task_type="data_analysis",
                    expected='"Move Your Body"',
                    reward=5,
                ),
            ]
            (pack / "manifest.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8",
            )
            out_json = root / "report.json"
            out_md = root / "report.md"
            report = score_pack(pack, None, out_json, out_md)
            by_id = {t["task_id"]: t for t in report["tasks"]}
            self.assertEqual(by_id["coding_ok"]["verdict"], "correct")
            self.assertTrue(by_id["coding_ok"]["tests_ok"])
            self.assertEqual(by_id["coding_fake"]["verdict"], "wrong")
            self.assertFalse(by_id["coding_fake"]["tests_ok"])
            self.assertEqual(by_id["da_quote"]["verdict"], "correct")
            self.assertEqual(report["summary"]["correct"], 2)
            self.assertEqual(report["summary"]["wrong"], 1)


class VerdictQuoteTests(unittest.TestCase):
    def test_verdict_quotes(self) -> None:
        self.assertEqual(
            _verdict(True, '"Somewhere In Europe"', "Somewhere In Europe"),
            "correct",
        )


if __name__ == "__main__":
    unittest.main()
