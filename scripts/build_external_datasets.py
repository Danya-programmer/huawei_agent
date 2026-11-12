#!/usr/bin/env python3
"""Build contest-format test-100 / train-3000 packs from public HF datasets.

Usage:
  python scripts/build_external_datasets.py --out task_packs --test-n 100 --train-n 3000 --seed 42

Requires: datasets, huggingface_hub (build-time only; not in Docker image).
Caches upstream dumps under .cache/hf_datasets/ (gitignored).
Output dir must NOT be named ``datasets`` (shadows the HuggingFace package).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "hf_datasets"

TASK_TYPES = ("math", "data_analysis", "search", "coding")
DIFFICULTIES = ("easy", "medium", "hard")

MATH_FAIL_MODES = ("no_starts_with", "contradictory_ops", "non_integer_trap")
CSV_FAIL_MODES = ("missing_sales_csv", "broken_header", "missing_units_column")
SEARCH_FAIL_MODES = ("person_missing", "evidence_missing", "empty_kb")
CODING_FAIL_MODES = ("missing_mylib", "foreign_import", "empty_test")

# AbstentionBench (facebook/AbstentionBench) uses a loading script incompatible with
# datasets≥4. We pull the same underlying abstain-positive sources it wraps:
# FalseQA (false premises) + SQuAD 2.0 unanswerable.
FALSEQA_URL = (
    "https://raw.githubusercontent.com/thunlp/FalseQA/refs/heads/main/dataset/test.csv"
)

ANSWER_SUFFIX = (
    "Work only inside the current directory. When you are done, write ONLY the final answer "
    "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
)
CODING_SUFFIX = (
    "Work only inside the current directory. When you are done, write ONLY the word DONE "
    "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
)


@dataclass
class Candidate:
    task_type: str
    source: str
    source_id: str
    difficulty: str
    instruction: str
    expected: str
    workspace_files: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotSpec:
    task_type: str
    difficulty: str
    solvable: bool
    fail_mode: Optional[str]


@dataclass
class BuiltTask:
    task_id: str
    task_type: str
    difficulty: str
    solvable: bool
    expected: str
    reward: int
    fail_mode: Optional[str]
    source: str
    source_id: str
    path: Path


# ---------------------------------------------------------------------------
# Quotas
# ---------------------------------------------------------------------------


def plan_slots(n: int, unsolvable_ratio: float = 0.2) -> list[SlotSpec]:
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4 so types are equal")
    if not 0.0 <= unsolvable_ratio < 1.0:
        raise ValueError("unsolvable_ratio must be in [0, 1)")

    per_type = n // 4
    n_unsolvable = int(round(n * unsolvable_ratio))
    if n_unsolvable % 4 != 0:
        n_unsolvable = int(round(n_unsolvable / 4) * 4)
    n_unsolvable = min(n_unsolvable, n)
    unsolvable_per_type = n_unsolvable // 4
    solvable_per_type = per_type - unsolvable_per_type
    n_solvable = solvable_per_type * 4

    if n_solvable == 80:
        easy_n, medium_n, hard_n = 40, 25, 15
    else:
        easy_n = int(round(n_solvable * 40 / 80))
        hard_n = int(round(n_solvable * 15 / 80))
        medium_n = n_solvable - easy_n - hard_n
        if medium_n < 0:
            medium_n = 0
            while easy_n + hard_n > n_solvable:
                if easy_n >= hard_n and easy_n > 0:
                    easy_n -= 1
                elif hard_n > 0:
                    hard_n -= 1
                else:
                    break
            medium_n = n_solvable - easy_n - hard_n

    easy_per = _spread(easy_n, 4)
    medium_per = _spread(medium_n, 4)
    hard_per = _spread(hard_n, 4)
    for i in range(4):
        while easy_per[i] + medium_per[i] + hard_per[i] > solvable_per_type:
            if medium_per[i] > 0:
                medium_per[i] -= 1
            elif easy_per[i] > 0:
                easy_per[i] -= 1
            elif hard_per[i] > 0:
                hard_per[i] -= 1
        while easy_per[i] + medium_per[i] + hard_per[i] < solvable_per_type:
            medium_per[i] += 1

    fail_banks = {
        "math": MATH_FAIL_MODES,
        "data_analysis": CSV_FAIL_MODES,
        "search": SEARCH_FAIL_MODES,
        "coding": CODING_FAIL_MODES,
    }
    slots: list[SlotSpec] = []
    for ti, task_type in enumerate(TASK_TYPES):
        for d, count in (
            ("easy", easy_per[ti]),
            ("medium", medium_per[ti]),
            ("hard", hard_per[ti]),
        ):
            for _ in range(count):
                slots.append(SlotSpec(task_type, d, True, None))
        modes = fail_banks[task_type]
        for j in range(unsolvable_per_type):
            slots.append(SlotSpec(task_type, "unsolvable", False, modes[j % len(modes)]))
    return slots


def _spread(total: int, buckets: int) -> list[int]:
    base, rem = divmod(total, buckets)
    return [base + (1 if i < rem else 0) for i in range(buckets)]


def assert_quotas(slots: list[SlotSpec], n: int, unsolvable_ratio: float = 0.2) -> None:
    assert len(slots) == n
    by_type = {t: 0 for t in TASK_TYPES}
    by_diff = {"easy": 0, "medium": 0, "hard": 0}
    unsolvable = 0
    unsolvable_by_type = {t: 0 for t in TASK_TYPES}
    for s in slots:
        by_type[s.task_type] += 1
        if not s.solvable:
            unsolvable += 1
            unsolvable_by_type[s.task_type] += 1
        else:
            by_diff[s.difficulty] += 1
    assert by_type == {t: n // 4 for t in TASK_TYPES}, by_type
    expected_uns = int(round(n * unsolvable_ratio))
    if expected_uns % 4 != 0:
        expected_uns = int(round(expected_uns / 4) * 4)
    assert unsolvable == expected_uns, (unsolvable, expected_uns)
    assert all(v == expected_uns // 4 for v in unsolvable_by_type.values()), unsolvable_by_type
    n_solvable = n - expected_uns
    assert sum(by_diff.values()) == n_solvable


# ---------------------------------------------------------------------------
# HF loading helpers
# ---------------------------------------------------------------------------


def _ensure_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install with:\n"
            "  uv venv .venv && uv pip install --python .venv/bin/python "
            "'datasets>=2.19' 'huggingface_hub>=0.23'\n"
            "  # or: python3 -m venv .venv && .venv/bin/pip install datasets huggingface_hub"
        ) from exc


def _load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VAL from .env into os.environ (does not override existing)."""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if not key:
            continue
        os.environ.setdefault(key, val)
    # HuggingFace Hub accepts either name.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def _hf_load(
    path: str,
    name: Optional[str] = None,
    split: str = "train",
    revision: Optional[str] = None,
    **kwargs,
):
    from datasets import load_dataset

    _load_dotenv()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR / "hf_datasets"))
    load_kwargs = dict(kwargs)
    if revision is not None:
        load_kwargs["revision"] = revision
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and "token" not in load_kwargs:
        load_kwargs["token"] = token
    if name is None:
        return load_dataset(path, split=split, **load_kwargs)
    return load_dataset(path, name, split=split, **load_kwargs)


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (s or "doc")[:max_len]


def _reward(rng: random.Random) -> int:
    return rng.randint(1, 5)


def _extract_gsm8k_answer(raw: str) -> Optional[str]:
    if "####" in raw:
        ans = raw.split("####")[-1].strip().replace(",", "")
        return ans
    return None


def _extract_math_answer(solution: str) -> Optional[str]:
    # Prefer \\boxed{...}
    m = re.search(r"\\boxed\{([^{}]+)\}", solution)
    if not m:
        m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    if m:
        ans = m.group(1).strip()
        ans = ans.replace("\\,", "").replace(",", "").strip()
        # Keep short answers only
        if len(ans) <= 64 and "\n" not in ans:
            return ans
    return None


def _is_simple_math_answer(ans: str) -> bool:
    if re.fullmatch(r"-?\d+(\.\d+)?", ans):
        return True
    if re.fullmatch(r"-?\d+/\d+", ans):
        return True
    if len(ans) <= 24 and not re.search(r"\\(frac|begin|end|sum|int)", ans):
        return True
    return False


# ---------------------------------------------------------------------------
# Converters → Candidate pools
# ---------------------------------------------------------------------------


def load_math_candidates(limit: int = 8000) -> list[Candidate]:
    out: list[Candidate] = []
    gsm = _hf_load("openai/gsm8k", name="main", split="train")
    for i, row in enumerate(gsm):
        ans = _extract_gsm8k_answer(row["answer"])
        if not ans:
            continue
        q = row["question"].strip()
        instruction = (
            f"{q}\n\nCompute it with the bash tool (python3 -c). {ANSWER_SUFFIX}"
        )
        out.append(
            Candidate(
                task_type="math",
                source="gsm8k",
                source_id=f"gsm8k_train_{i}",
                difficulty="easy",
                instruction=instruction,
                expected=ans,
            )
        )
        if len(out) >= limit // 2:
            break

    try:
        math_ds = _hf_load("DigitalLearningGmbH/MATH-lighteval", split="train")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: MATH-lighteval failed ({exc}); trying EleutherAI/hendrycks_math")
        math_ds = None
        for subject in (
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ):
            try:
                part = _hf_load("EleutherAI/hendrycks_math", name=subject, split="train")
            except Exception:
                continue
            if math_ds is None:
                math_ds = part
            else:
                from datasets import concatenate_datasets

                math_ds = concatenate_datasets([math_ds, part])

    if math_ds is not None:
        for i, row in enumerate(math_ds):
            level_raw = str(row.get("level") or row.get("type") or "")
            lvl_m = re.search(r"(\d+)", level_raw)
            level = int(lvl_m.group(1)) if lvl_m else 3
            solution = row.get("solution") or row.get("answer") or ""
            ans = _extract_math_answer(str(solution))
            if not ans or not _is_simple_math_answer(ans):
                continue
            problem = str(row.get("problem") or row.get("question") or "").strip()
            if not problem or len(problem) > 2500:
                continue
            difficulty = "hard" if level >= 4 else "medium"
            instruction = (
                f"{problem}\n\nCompute it with the bash tool (python3 -c). {ANSWER_SUFFIX}"
            )
            sid = str(row.get("unique_id") or row.get("id") or f"math_{i}")
            out.append(
                Candidate(
                    task_type="math",
                    source="math",
                    source_id=f"math_{sid}",
                    difficulty=difficulty,
                    instruction=instruction,
                    expected=ans.replace("$", "").strip(),
                    meta={"level": level},
                )
            )
            if sum(1 for c in out if c.source == "math") >= limit // 2:
                break
    print(f"math candidates: {len(out)}")
    return out


def _bugify_python(src: str, rng: random.Random) -> str:
    """Apply a light deterministic mutation so tests fail until fixed."""
    lines = src.splitlines(keepends=True)
    if not lines:
        return "def broken():\n    return None\n"
    mutated = False
    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        if mutated or stripped.startswith("#") or stripped.startswith("def ") or stripped.startswith("class "):
            out_lines.append(line)
            continue
        if "return " in line and "None" not in line:
            # Flip +/− or append an off-by-one when possible
            if "+" in line and rng.random() < 0.7:
                out_lines.append(line.replace("+", "-", 1))
                mutated = True
                continue
            if "-" in line and "+-" not in line:
                out_lines.append(line.replace("-", "+", 1))
                mutated = True
                continue
            indent = line[: len(line) - len(line.lstrip())]
            expr = line.split("return ", 1)[1].rstrip("\n").rstrip()
            if expr.endswith(":"):
                out_lines.append(line)
                continue
            out_lines.append(f"{indent}return ({expr}) + 1  # bug\n")
            mutated = True
            continue
        out_lines.append(line)
    if not mutated:
        out_lines.append("\n# intentional bug marker\nBUG = True\n")
        # Also break first return-less function body by inserting wrong default
        text = "".join(out_lines)
        text = text.replace("True", "False", 1) if "True" in text else text + "\ndef _noop():\n    return 1\n"
        return text
    return "".join(out_lines)


def _humaneval_to_mylib(prompt: str, canonical: str, entry_point: str) -> str:
    # prompt usually contains imports + def signature + docstring; append solution body
    code = prompt.rstrip() + "\n" + canonical.lstrip()
    # Rename entry point stays; tests import mylib
    return code


def _humaneval_tests(test: str, entry_point: str) -> str:
    # HumanEval test defines `check(candidate)`; wrap into unittest importing mylib
    return (
        "import unittest\n"
        "import mylib\n\n"
        f"{test}\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_check(self):\n"
        f"        check(mylib.{entry_point})\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def _mbpp_tests(test_list: list[str]) -> str:
    body = "\n".join(f"        {t}" for t in test_list)
    return (
        "import unittest\n"
        "import mylib\n"
        # Make functions available unqualified for MBPP asserts like assert similar_elements(...)
        "from mylib import *  # noqa: F401,F403\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_cases(self):\n"
        f"{body}\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def _coding_difficulty(n_tests: int, code_len: int, prefer_hard: bool = False) -> str:
    if prefer_hard or n_tests >= 5 or code_len >= 600:
        return "hard"
    if n_tests <= 2 and code_len < 220:
        return "easy"
    return "medium"


def load_coding_candidates(limit: int = 4000, rng: Optional[random.Random] = None) -> list[Candidate]:
    rng = rng or random.Random(0)
    out: list[Candidate] = []
    seen: set[str] = set()
    instruction = (
        "Run `python3 test_mylib.py` in the current directory; some tests fail. "
        "Fix the bugs in mylib.py (do not edit the tests) until all tests pass. "
        + CODING_SUFFIX
    )

    def add_mbpp_row(row: dict[str, Any], cfg: str, split: str, i: int, bug_seed: int) -> None:
        code = str(row.get("code") or "").strip()
        tests = list(row.get("test_list") or [])
        if not code or not tests:
            return
        tid = row.get("task_id", i)
        source_id = f"mbpp_{cfg}_{split}_{tid}_b{bug_seed}"
        if source_id in seen:
            return
        seen.add(source_id)
        buggy = _bugify_python(code, random.Random(hash((tid, bug_seed, cfg)) & 0xFFFFFFFF))
        difficulty = _coding_difficulty(len(tests), len(code), prefer_hard=(bug_seed > 0 and len(code) > 300))
        out.append(
            Candidate(
                task_type="coding",
                source="mbpp",
                source_id=source_id,
                difficulty=difficulty,
                instruction=instruction,
                expected="DONE",
                workspace_files={
                    "mylib.py": buggy,
                    "test_mylib.py": _mbpp_tests(tests),
                },
                meta={"n_tests": len(tests), "cfg": cfg, "split": split},
            )
        )

    # Full + sanitized MBPP across splits → enough unique rows for train-3000 coding quota.
    for cfg in ("full", "sanitized"):
        for split in ("train", "test", "validation", "prompt"):
            try:
                mbpp = _hf_load("google-research-datasets/mbpp", name=cfg, split=split)
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: mbpp {cfg}/{split}: {exc}")
                continue
            for i, row in enumerate(mbpp):
                add_mbpp_row(row, cfg, split, i, bug_seed=0)
                # Second bug variant expands pool without inventing new problems.
                if len(out) < limit:
                    add_mbpp_row(row, cfg, split, i, bug_seed=1)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break

    he = _hf_load("openai/openai_humaneval", split="test")
    for i, row in enumerate(he):
        prompt = str(row["prompt"])
        canonical = str(row["canonical_solution"])
        entry = str(row["entry_point"])
        test = str(row["test"])
        good = _humaneval_to_mylib(prompt, canonical, entry)
        for bug_seed in (0, 1):
            source_id = f"{row.get('task_id') or f'HumanEval/{i}'}_b{bug_seed}"
            if source_id in seen:
                continue
            seen.add(source_id)
            buggy = _bugify_python(good, random.Random(hash((i, "he", bug_seed)) & 0xFFFFFFFF))
            n_assert = test.count("assert")
            difficulty = _coding_difficulty(n_assert, len(good), prefer_hard=True)
            out.append(
                Candidate(
                    task_type="coding",
                    source="humaneval",
                    source_id=source_id,
                    difficulty=difficulty,
                    instruction=instruction,
                    expected="DONE",
                    workspace_files={
                        "mylib.py": buggy,
                        "test_mylib.py": _humaneval_tests(test, entry),
                    },
                    meta={"n_assert": n_assert, "entry_point": entry},
                )
            )
    print(f"coding candidates: {len(out)}")
    return out


def _table_to_csv(table: Any) -> tuple[str, int]:
    """Convert WikiTQ-style table dict to CSV text; return (csv, nrows)."""
    if isinstance(table, dict):
        header = table.get("header") or table.get("column_header") or []
        rows = table.get("rows") or table.get("content") or []
    else:
        header, rows = [], []
    if not header and rows:
        width = max(len(r) for r in rows)
        header = [f"col_{i}" for i in range(width)]
    buf = io.StringIO()
    writer = csv.writer(buf)
    if header:
        writer.writerow(header)
    for r in rows:
        writer.writerow(list(r))
    return buf.getvalue(), len(rows)


def _normalize_wiki_answer(ans: Any) -> Optional[str]:
    if ans is None:
        return None
    if isinstance(ans, list):
        if not ans:
            return None
        ans = ans[0]
    s = str(ans).strip()
    if not s or len(s) > 120:
        return None
    return s


def load_data_analysis_candidates(limit: int = 5000) -> list[Candidate]:
    out: list[Candidate] = []
    # datasets>=4 removed loading scripts. Parquet lives on PR branch refs/pr/4.
    ds = None
    source_name = "stanfordnlp/wikitablequestions"
    for path, kwargs in (
        (
            "stanfordnlp/wikitablequestions",
            {"name": "random-split-1", "split": "train", "revision": "refs/pr/4"},
        ),
        (
            "stanfordnlp/wikitablequestions",
            {"split": "train", "revision": "refs/pr/4"},
        ),
    ):
        try:
            ds = _hf_load(path, **kwargs)
            source_name = path
            break
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: load {path} {kwargs} failed: {exc}")
    if ds is None:
        raise RuntimeError(
            "Could not load WikiTableQuestions parquet "
            "(tried stanfordnlp/wikitablequestions@refs/pr/4). "
            "datasets>=4 no longer supports the legacy loading script. "
            "Set HF_TOKEN in .env for higher Hub rate limits."
        )

    for i, row in enumerate(ds):
        table = row.get("table") or row.get("context")
        question = str(row.get("question") or "").strip()
        answers = row.get("answers") or row.get("answer") or row.get("target_text")
        expected = _normalize_wiki_answer(answers)
        if not question or expected is None or table is None:
            continue
        csv_text, nrows = _table_to_csv(table)
        if nrows < 2 or nrows > 400:
            continue
        if len(csv_text) > 80_000:
            continue
        if nrows <= 15:
            difficulty = "easy"
        elif nrows <= 60:
            difficulty = "medium"
        else:
            difficulty = "hard"
        instruction = (
            f"The file sales.csv is a data table ({nrows} rows, plus header). "
            f"Question: {question} "
            f"Use bash with python3 to compute it. {ANSWER_SUFFIX}"
        )
        sid = str(row.get("id") or row.get("question_id") or i)
        out.append(
            Candidate(
                task_type="data_analysis",
                source=source_name,
                source_id=f"wikitq_{sid}",
                difficulty=difficulty,
                instruction=instruction,
                expected=expected,
                workspace_files={"sales.csv": csv_text},
                meta={"nrows": nrows},
            )
        )
        if len(out) >= limit:
            break
    print(f"data_analysis candidates: {len(out)}")
    return out


def load_search_candidates(limit: int = 5000) -> list[Candidate]:
    out: list[Candidate] = []
    ds = None
    # huggingface_hub requires namespace/name (bare "hotpot_qa" → HfUriError).
    for path, kwargs in (
        ("hotpotqa/hotpot_qa", {"name": "distractor", "split": "train"}),
        (
            "hotpotqa/hotpot_qa",
            {"name": "distractor", "split": "train", "revision": "refs/convert/parquet"},
        ),
    ):
        try:
            ds = _hf_load(path, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: load {path} {kwargs} failed: {exc}")
    if ds is None:
        raise RuntimeError(
            "Could not load HotpotQA (tried hotpotqa/hotpot_qa distractor). "
            "Bare id 'hotpot_qa' is invalid for current huggingface_hub."
        )
    for i, row in enumerate(ds):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer or len(answer) > 80:
            continue
        ctx = row.get("context") or {}
        titles = list(ctx.get("title") or [])
        sentences = list(ctx.get("sentences") or [])
        if not titles or not sentences:
            continue
        # Build kb markdown files; cap to keep zip small
        files: dict[str, str] = {}
        used = 0
        for title, sents in zip(titles, sentences):
            if used >= 12:
                break
            body = " ".join(sents) if isinstance(sents, list) else str(sents)
            body = body.strip()
            if not body:
                continue
            if len(body) > 1200:
                body = body[:1200] + "..."
            slug = _slug(str(title))
            fname = f"kb/{slug}.md"
            # avoid collisions
            if fname in files:
                fname = f"kb/{slug}_{used}.md"
            files[fname] = f"# {title}\n\n{body}\n"
            used += 1
        if used < 2:
            continue
        level = str(row.get("level") or "medium").lower()
        qtype = str(row.get("type") or "").lower()
        if level == "easy" or qtype == "bridge":
            difficulty = "easy" if level == "easy" else "medium"
        elif level == "hard" or qtype == "comparison":
            difficulty = "hard"
        else:
            difficulty = "medium"
        if qtype == "bridge" and difficulty == "hard":
            difficulty = "medium"
        if qtype == "comparison":
            difficulty = "hard" if used >= 6 else "medium"

        instruction = (
            f"The directory kb/ contains a small knowledge base of markdown files. "
            f"{question} "
            f"Use the file tools or bash (ls, grep, cat) to find it. {ANSWER_SUFFIX}"
        )
        sid = str(row.get("id") or i)
        out.append(
            Candidate(
                task_type="search",
                source="hotpot_qa",
                source_id=f"hotpot_{sid}",
                difficulty=difficulty,
                instruction=instruction,
                expected=answer,
                workspace_files=files,
                meta={"level": level, "type": qtype, "n_docs": used},
            )
        )
        if len(out) >= limit:
            break
    print(f"search candidates: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# AbstentionBench-derived (should_abstain=True)
# ---------------------------------------------------------------------------


def _download_text(url: str, cache_name: str) -> str:
    import urllib.request

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / cache_name
    if path.is_file() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "huawei-agent-hackaton/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    path.write_text(data, encoding="utf-8")
    return data


def load_abstention_candidates(limit: int = 2000) -> list[Candidate]:
    """Build unsolvable contest tasks from AbstentionBench-aligned sources.

    facebook/AbstentionBench itself requires datasets≤3.6 + trust_remote_code;
    we mirror its FalseQA + SQuAD2 abstain-positive slices instead.
    """
    out: list[Candidate] = []

    # --- FalseQA (label=1 → false premise → abstain) ---
    try:
        raw = _download_text(FALSEQA_URL, "falseqa_test.csv")
        reader = csv.DictReader(io.StringIO(raw))
        for i, row in enumerate(reader):
            try:
                label = int(str(row.get("label", "0")).strip() or "0")
            except ValueError:
                continue
            if label != 1:
                continue
            q = str(row.get("question") or "").strip()
            if not q or len(q) > 500:
                continue
            instruction = (
                f"The directory kb/ contains a small knowledge base of markdown files. "
                f"{q} "
                f"Use the file tools or bash (ls, grep, cat) to find it. "
                f"If the question cannot be answered from the available files, write NOTHING "
                f"and stop. {ANSWER_SUFFIX}"
            )
            out.append(
                Candidate(
                    task_type="search",
                    source="abstentionbench_falseqa",
                    source_id=f"falseqa_{i}",
                    difficulty="unsolvable",
                    instruction=instruction,
                    expected="UNSOLVABLE",
                    workspace_files={
                        "kb/decoy.md": (
                            "# Decoy notes\n\n"
                            "General trivia that does not resolve the user question.\n"
                            "No authoritative answer is present in this knowledge base.\n"
                        )
                    },
                    meta={"fail_mode": "abstention_false_premise", "abstention": True},
                )
            )
            if sum(1 for c in out if c.source == "abstentionbench_falseqa") >= limit // 2:
                break
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: FalseQA download/parse failed: {exc}")

    # --- SQuAD 2.0 unanswerable (AbstentionBench squad2) ---
    try:
        ds = None
        for path, kwargs in (
            ("rajpurkar/squad_v2", {"split": "validation"}),
            ("squad_v2", {"split": "validation"}),
        ):
            try:
                ds = _hf_load(path, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: load {path} failed: {exc}")
        if ds is not None:
            n_sq = 0
            for i, row in enumerate(ds):
                answers = row.get("answers") or {}
                texts = answers.get("text") if isinstance(answers, dict) else None
                if texts is None:
                    continue
                if len(texts) != 0:
                    continue  # answerable
                q = str(row.get("question") or "").strip()
                ctx = str(row.get("context") or "").strip()
                if not q or not ctx or len(q) > 400:
                    continue
                if len(ctx) > 2500:
                    ctx = ctx[:2500] + "..."
                instruction = (
                    f"The directory kb/ contains a small knowledge base of markdown files. "
                    f"{q} "
                    f"Use the file tools or bash (ls, grep, cat) to find it. "
                    f"If the question cannot be answered from the available files, write NOTHING "
                    f"and stop. {ANSWER_SUFFIX}"
                )
                sid = str(row.get("id") or i)
                out.append(
                    Candidate(
                        task_type="search",
                        source="abstentionbench_squad2",
                        source_id=f"squad2_{sid}",
                        difficulty="unsolvable",
                        instruction=instruction,
                        expected="UNSOLVABLE",
                        workspace_files={
                            "kb/passage.md": f"# Passage\n\n{ctx}\n",
                        },
                        meta={"fail_mode": "abstention_unanswerable", "abstention": True},
                    )
                )
                n_sq += 1
                if n_sq >= limit // 2:
                    break
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: SQuAD2 abstention load failed: {exc}")

    # Underspecified math-style abstention (no numbers) — small synthetic slice
    # aligned with AbstentionBench UMWP / known-unknown spirit.
    for i in range(min(200, limit // 10)):
        instruction = (
            f"A quantity of interest is never numerically specified in the problem statement "
            f"(case {i}). After an unknown number of undocumented operations, what is the "
            f"exact final integer value? Compute it with the bash tool (python3 -c). "
            f"If the problem is underdetermined, write NOTHING and stop. {ANSWER_SUFFIX}"
        )
        out.append(
            Candidate(
                task_type="math",
                source="abstentionbench_underspec",
                source_id=f"abstain_math_{i}",
                difficulty="unsolvable",
                instruction=instruction,
                expected="UNSOLVABLE",
                workspace_files={},
                meta={"fail_mode": "abstention_underspecified", "abstention": True},
            )
        )

    print(f"abstention candidates: {len(out)}")
    return out


def materialize_abstention_pack(
    out_dir: Path,
    candidates: list[Candidate],
    n: int,
    seed: int,
) -> list[BuiltTask]:
    """Write a standalone AbstentionBench-derived pack (all expected=UNSOLVABLE)."""
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        if out_dir.exists():
            # Fallback if a concurrent writer left remnants.
            bak = out_dir.with_name(out_dir.name + ".bak")
            if bak.exists():
                shutil.rmtree(bak, ignore_errors=True)
            out_dir.rename(bak)
            shutil.rmtree(bak, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = [c for c in candidates if c.expected == "UNSOLVABLE"]
    if not pool:
        raise RuntimeError("no abstention candidates to materialize")
    rng.shuffle(pool)
    chosen = pool[: min(n, len(pool))]
    results: list[BuiltTask] = []
    for idx, cand in enumerate(chosen):
        task_id = f"abstention_{idx:04d}"
        reward = _reward(rng)
        task_dir = out_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "task_type": cand.task_type,
            "instruction": cand.instruction,
            "reward": reward,
            "expected": "UNSOLVABLE",
        }
        (task_dir / "task.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if cand.workspace_files:
            ws = task_dir / "workspace"
            for rel, content in cand.workspace_files.items():
                path = ws / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        results.append(
            BuiltTask(
                task_id=task_id,
                task_type=cand.task_type,
                difficulty="unsolvable",
                solvable=False,
                expected="UNSOLVABLE",
                reward=reward,
                fail_mode=str(cand.meta.get("fail_mode") or "abstention"),
                source=cand.source,
                source_id=cand.source_id,
                path=task_dir,
            )
        )
    results.sort(key=lambda r: r.task_id)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "task_id": r.task_id,
                        "task_type": r.task_type,
                        "difficulty": r.difficulty,
                        "solvable": False,
                        "expected": "UNSOLVABLE",
                        "reward": r.reward,
                        "fail_mode": r.fail_mode,
                        "source": r.source,
                        "source_id": r.source_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"abstention-pack: wrote {len(results)} tasks → {out_dir}")
    return results


# ---------------------------------------------------------------------------
# Mutations (unsolvable)
# ---------------------------------------------------------------------------


def mutate_unsolvable(cand: Candidate, fail_mode: str, rng: random.Random) -> Candidate:
    c = Candidate(
        task_type=cand.task_type,
        source=cand.source,
        source_id=f"{cand.source_id}__uns_{fail_mode}",
        difficulty="unsolvable",
        instruction=cand.instruction,
        expected="UNSOLVABLE",
        workspace_files=dict(cand.workspace_files),
        meta=dict(cand.meta),
    )
    c.meta["fail_mode"] = fail_mode
    c.meta["parent_source_id"] = cand.source_id

    if cand.task_type == "math":
        if fail_mode == "contradictory_ops":
            a, b = rng.randint(10, 99), rng.randint(100, 999)
            c.instruction = (
                f"A problem states the answer must be both {a} and {b} at once "
                f"(contradictory givens). Original prompt was: {cand.instruction[:400]} "
                f"How many is the final numeric answer? Compute it with the bash tool (python3 -c). "
                + ANSWER_SUFFIX
            )
        elif fail_mode == "non_integer_trap":
            c.instruction = (
                "Start with 100.5 items. Split equally into 3 groups and keep one share as a "
                "non-terminating float without rounding. Report the exact final count as a single "
                "integer in answer.txt (this is intentionally impossible). "
                + ANSWER_SUFFIX
            )
        else:  # no_starts_with
            c.instruction = (
                "A quantity is unknown (exact count never stated). After several undocumented "
                "operations it ends with some integer. What is the final number? "
                "Compute it with the bash tool (python3 -c). "
                + ANSWER_SUFFIX
            )
        c.workspace_files = {}
        return c

    if cand.task_type == "data_analysis":
        if fail_mode == "missing_sales_csv":
            c.workspace_files = {"readme.txt": "sales data was not shipped\n"}
        elif fail_mode == "missing_units_column":
            # Rewrite CSV without a key numeric column if possible
            raw = cand.workspace_files.get("sales.csv", "a,b\n1,2\n")
            reader = csv.reader(io.StringIO(raw))
            rows = list(reader)
            if rows:
                header = rows[0]
                if len(header) > 1:
                    drop = len(header) - 1
                    new_rows = [r[:drop] for r in rows]
                else:
                    new_rows = [["bad"], ["1"]]
            else:
                new_rows = [["bad"], ["1"]]
            buf = io.StringIO()
            csv.writer(buf).writerows(new_rows)
            c.workspace_files = {"sales.csv": buf.getvalue()}
        else:  # broken_header
            c.workspace_files = {"sales.csv": "bad,header\n1,2\n"}
        return c

    if cand.task_type == "search":
        files = dict(c.workspace_files)
        keys = sorted(k for k in files if k.startswith("kb/"))
        if fail_mode == "empty_kb":
            c.workspace_files = {"kb/.keep": ""}
        elif fail_mode == "evidence_missing" and keys:
            # Drop half of docs including first (often supporting)
            keep_n = max(1, len(keys) // 3)
            keep = set(keys[-keep_n:])
            c.workspace_files = {k: v for k, v in files.items() if k in keep}
            # Also scrub answer string from remaining text
            ans = cand.expected
            scrubbed = {}
            for k, v in c.workspace_files.items():
                scrubbed[k] = v.replace(ans, "[REDACTED]") if ans else v
            c.workspace_files = scrubbed
        else:  # person_missing / default: remove all docs, leave decoy
            c.workspace_files = {
                "kb/decoy.md": "# Decoy\n\nNo relevant facts are recorded here.\n"
            }
        return c

    if cand.task_type == "coding":
        test = c.workspace_files.get("test_mylib.py", "")
        if fail_mode == "foreign_import":
            test2 = test.replace("import mylib", "import otherlib as mylib")
            test2 = test2.replace("from mylib import", "from otherlib import")
            c.workspace_files = {
                "mylib.py": c.workspace_files.get("mylib.py", "def f():\n    return 0\n"),
                "test_mylib.py": test2,
            }
        elif fail_mode == "empty_test":
            c.workspace_files = {
                "mylib.py": c.workspace_files.get("mylib.py", "def f():\n    return 0\n"),
                "test_mylib.py": "",
            }
        else:  # missing_mylib
            c.workspace_files = {"test_mylib.py": test or "import mylib\n"}
        return c

    return c


# ---------------------------------------------------------------------------
# Sampling + writing
# ---------------------------------------------------------------------------


def build_pools(seed: int) -> dict[str, list[Candidate]]:
    rng = random.Random(seed)
    abstention = load_abstention_candidates()
    return {
        "math": load_math_candidates(),
        "coding": load_coding_candidates(rng=rng),
        "data_analysis": load_data_analysis_candidates(),
        "search": load_search_candidates(),
        "abstention": abstention,
    }


def _pick_solvable(
    pool: list[Candidate],
    difficulty: str,
    used: set[str],
    rng: random.Random,
) -> Candidate:
    # Prefer exact difficulty; fall back to any unused
    exact = [c for c in pool if c.difficulty == difficulty and c.source_id not in used]
    if exact:
        c = rng.choice(exact)
        used.add(c.source_id)
        return c
    # Adjacent difficulties
    order = {
        "easy": ("easy", "medium", "hard"),
        "medium": ("medium", "easy", "hard"),
        "hard": ("hard", "medium", "easy"),
    }[difficulty]
    for d in order:
        cand = [c for c in pool if c.difficulty == d and c.source_id not in used]
        if cand:
            c = rng.choice(cand)
            used.add(c.source_id)
            return c
    raise RuntimeError(f"exhausted pool for difficulty={difficulty}")


def _pick_for_unsolvable(pool: list[Candidate], used: set[str], rng: random.Random) -> Candidate:
    avail = [c for c in pool if c.source_id not in used]
    if not avail:
        # Allow reuse of parents for mutation-only ids (still unique via suffix)
        avail = list(pool)
    c = rng.choice(avail)
    used.add(c.source_id)
    return c


def materialize_set(
    name: str,
    out_dir: Path,
    n: int,
    pools: dict[str, list[Candidate]],
    used: set[str],
    seed: int,
    unsolvable_ratio: float = 0.2,
    abstention_mix: float = 0.5,
) -> list[BuiltTask]:
    rng = random.Random(seed)
    slots = plan_slots(n, unsolvable_ratio)
    assert_quotas(slots, n, unsolvable_ratio)
    rng.shuffle(slots)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counters = {t: 0 for t in TASK_TYPES}
    results: list[BuiltTask] = []
    abstention_pool = [
        c
        for c in pools.get("abstention", [])
        if c.expected == "UNSOLVABLE"
    ]

    for slot in slots:
        idx = counters[slot.task_type]
        counters[slot.task_type] = idx + 1
        task_id = f"{slot.task_type}_ext_{idx:04d}"
        pool = pools[slot.task_type]
        reward = _reward(rng)

        if slot.solvable:
            cand = _pick_solvable(pool, slot.difficulty, used, rng)
            expected = cand.expected
            instruction = cand.instruction
            files = cand.workspace_files
            difficulty = slot.difficulty
            fail_mode = None
            source, source_id = cand.source, cand.source_id
        else:
            # Prefer AbstentionBench-derived items for search/math unsolvable slots.
            use_abstention = (
                slot.task_type in {"search", "math"}
                and abstention_pool
                and rng.random() < abstention_mix
            )
            if use_abstention:
                typed = [c for c in abstention_pool if c.task_type == slot.task_type]
                bank = typed or abstention_pool
                avail = [c for c in bank if c.source_id not in used] or bank
                cand = rng.choice(avail)
                used.add(cand.source_id)
                expected = "UNSOLVABLE"
                instruction = cand.instruction
                files = cand.workspace_files
                difficulty = "unsolvable"
                fail_mode = str(cand.meta.get("fail_mode") or "abstention")
                source, source_id = cand.source, cand.source_id
            else:
                parent = _pick_for_unsolvable(pool, used, rng)
                mut = mutate_unsolvable(parent, slot.fail_mode or "no_starts_with", rng)
                expected = "UNSOLVABLE"
                instruction = mut.instruction
                files = mut.workspace_files
                difficulty = "unsolvable"
                fail_mode = slot.fail_mode
                source, source_id = mut.source, mut.source_id

        task_dir = out_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "task_type": slot.task_type,
            "instruction": instruction,
            "reward": reward,
            "expected": expected,
        }
        (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if files:
            ws = task_dir / "workspace"
            for rel, content in files.items():
                path = ws / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

        results.append(
            BuiltTask(
                task_id=task_id,
                task_type=slot.task_type,
                difficulty=difficulty,
                solvable=slot.solvable,
                expected=expected,
                reward=reward,
                fail_mode=fail_mode,
                source=source,
                source_id=source_id,
                path=task_dir,
            )
        )

    results.sort(key=lambda r: r.task_id)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "task_id": r.task_id,
                        "task_type": r.task_type,
                        "difficulty": r.difficulty,
                        "solvable": r.solvable,
                        "expected": r.expected,
                        "reward": r.reward,
                        "fail_mode": r.fail_mode,
                        "source": r.source,
                        "source_id": r.source_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    _print_counts(name, results)
    print(f"{name}: wrote {len(results)} tasks → {out_dir}")
    return results


def _print_counts(name: str, results: list[BuiltTask]) -> None:
    by_type: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    solvable = unsolvable = 0
    for r in results:
        by_type[r.task_type] = by_type.get(r.task_type, 0) + 1
        if r.solvable:
            solvable += 1
            by_diff[r.difficulty] = by_diff.get(r.difficulty, 0) + 1
        else:
            unsolvable += 1
    print(f"[{name}] total={len(results)}")
    print("  by_type: " + " ".join(f"{k}={by_type.get(k, 0)}" for k in TASK_TYPES))
    print(f"  solvable={solvable} unsolvable={unsolvable}")
    print(
        "  by_difficulty(solvable): "
        + " ".join(f"{k}={by_diff.get(k, 0)}" for k in DIFFICULTIES)
    )


def zip_pack(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.name == "answer.txt" or "__pycache__" in path.parts:
                continue
            arc = path.relative_to(src_dir.parent)
            zf.write(path, arcname=str(arc))
    print(f"zip: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "task_packs")
    p.add_argument("--test-n", type=int, default=100)
    p.add_argument("--train-n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unsolvable-ratio", type=float, default=0.2)
    p.add_argument("--skip-zip", action="store_true")
    p.add_argument("--test-only", action="store_true", help="Build only the test pack")
    p.add_argument(
        "--abstention-n",
        type=int,
        default=100,
        help="Size of standalone abstention-pack (0 to skip)",
    )
    p.add_argument(
        "--abstention-mix",
        type=float,
        default=0.5,
        help="Fraction of search/math unsolvable slots filled from AbstentionBench-derived items",
    )
    p.add_argument(
        "--abstention-only",
        action="store_true",
        help="Only build abstention-pack.zip (skip test/train)",
    )
    args = p.parse_args()

    _load_dotenv()
    _ensure_datasets()
    args.out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("HF_TOKEN detected (authenticated Hub downloads).")
    else:
        print("WARN: no HF_TOKEN in env/.env — Hub rate limits may be low.")

    if args.abstention_only:
        print("Loading AbstentionBench-derived pool only...")
        abstention = load_abstention_candidates()
        if args.abstention_n > 0:
            abs_dir = args.out / "abstention-pack"
            materialize_abstention_pack(
                abs_dir, abstention, n=args.abstention_n, seed=args.seed + 7
            )
            if not args.skip_zip:
                zip_pack(abs_dir, args.out / "abstention-pack.zip")
        print("done (abstention-only).")
        return

    print("Loading upstream pools (cached under .cache/hf_datasets)...")
    pools = build_pools(args.seed)
    for t, items in pools.items():
        if t == "abstention":
            continue
        if len(items) < args.train_n // 4 + args.test_n // 4:
            print(f"WARN: pool {t} has only {len(items)} candidates")

    if args.abstention_n > 0:
        abs_dir = args.out / "abstention-pack"
        materialize_abstention_pack(
            abs_dir,
            pools.get("abstention", []),
            n=args.abstention_n,
            seed=args.seed + 7,
        )
        if not args.skip_zip:
            zip_pack(abs_dir, args.out / "abstention-pack.zip")

    used: set[str] = set()
    test_dir = args.out / "test-100"
    materialize_set(
        "test-100",
        test_dir,
        args.test_n,
        pools,
        used,
        seed=args.seed,
        unsolvable_ratio=args.unsolvable_ratio,
        abstention_mix=args.abstention_mix,
    )
    if not args.skip_zip:
        zip_pack(test_dir, args.out / "test-100.zip")

    if not args.test_only:
        train_dir = args.out / "train-3000"
        materialize_set(
            "train-3000",
            train_dir,
            args.train_n,
            pools,
            used,
            seed=args.seed + 1,
            unsolvable_ratio=args.unsolvable_ratio,
            abstention_mix=args.abstention_mix,
        )
        if not args.skip_zip:
            zip_pack(train_dir, args.out / "train-3000.zip")

    print("done.")


if __name__ == "__main__":
    main()
