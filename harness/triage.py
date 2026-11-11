"""Zero-token task triage: feasibility, cost bands, and scheduler lanes."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from harness.estimate import (
    COST_CODING_HEURISTIC,
    COST_CODING_REACT,
    COST_DETERMINISTIC,
    COST_SHORT_REACT,
    COST_SKIP,
)
from harness.solvers.coding import preflight_coding
from harness.solvers.csv_solver import _REQUIRED_COLS, parse_csv_query
from harness.solvers.math_solver import math_is_unsolvable, parse_math_ops
from harness.solvers.search_solver import try_solve_search
from harness.types import Task, TaskType

Lane = Literal["cheap", "coding", "skip", "needs_react_fallback"]


@dataclass(frozen=True)
class TaskProfile:
    """Triage result for one task (no LLM)."""

    task_type: TaskType
    difficulty_proxy: str
    feasible: bool
    fail_reason: Optional[str]
    cost_lo: int
    cost_hi: int
    expected_cost: int
    lane: Lane


def triage(task: Task) -> TaskProfile:
    """Classify ``task`` into a lane with a cost band. Uses FS + parse probes only."""
    if task.task_type == TaskType.MATH:
        return _triage_math(task)
    if task.task_type == TaskType.DATA_ANALYSIS:
        return _triage_csv(task)
    if task.task_type == TaskType.SEARCH:
        return _triage_search(task)
    if task.task_type == TaskType.CODING:
        return _triage_coding(task)
    return TaskProfile(
        task_type=task.task_type,
        difficulty_proxy="unknown",
        feasible=False,
        fail_reason="unknown_task_type",
        cost_lo=COST_SKIP,
        cost_hi=COST_SKIP,
        expected_cost=COST_SKIP,
        lane="skip",
    )


def _difficulty_from_length(n: int, *, easy: int, hard: int) -> str:
    if n <= easy:
        return "easy"
    if n >= hard:
        return "hard"
    return "medium"


def _triage_math(task: Task) -> TaskProfile:
    bad = math_is_unsolvable(task.instruction)
    if bad:
        return _skip(TaskType.MATH, "unsolvable", bad)

    ops = parse_math_ops(task.instruction)
    difficulty = _difficulty_from_length(
        len(ops) if ops else len(task.instruction),
        easy=3,
        hard=6,
    )
    if ops is not None:
        return TaskProfile(
            task_type=TaskType.MATH,
            difficulty_proxy=difficulty,
            feasible=True,
            fail_reason=None,
            cost_lo=COST_DETERMINISTIC,
            cost_hi=COST_DETERMINISTIC,
            expected_cost=COST_DETERMINISTIC,
            lane="cheap",
        )
    # Unrecognized template: keep for short ReAct fallback at budget tail.
    return TaskProfile(
        task_type=TaskType.MATH,
        difficulty_proxy="hard",
        feasible=True,
        fail_reason="math_parse_failed",
        cost_lo=COST_SHORT_REACT,
        cost_hi=COST_SHORT_REACT,
        expected_cost=COST_SHORT_REACT,
        lane="needs_react_fallback",
    )


def _csv_column_issue(csv_path: Path) -> Optional[str]:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return "csv_no_header"
            cols = {c.strip().lower() for c in reader.fieldnames}
            missing = [c for c in _REQUIRED_COLS if c not in cols]
            if missing:
                return f"missing_columns:{missing}"
    except OSError:
        return "csv_unreadable"
    except csv.Error:
        return "csv_bad"
    return None


def _triage_csv(task: Task) -> TaskProfile:
    csv_path = task.work_dir / "sales.csv"
    rows = 0
    if csv_path.is_file():
        try:
            with csv_path.open("r", encoding="utf-8") as fh:
                rows = max(0, sum(1 for _ in fh) - 1)
        except OSError:
            rows = 0
    difficulty = _difficulty_from_length(rows, easy=50, hard=300)

    if not csv_path.is_file():
        return _skip(task.task_type, difficulty, "missing_sales_csv")

    col_issue = _csv_column_issue(csv_path)
    if col_issue:
        return _skip(task.task_type, difficulty, col_issue)

    query = parse_csv_query(task.instruction)
    if query is None:
        return TaskProfile(
            task_type=TaskType.DATA_ANALYSIS,
            difficulty_proxy=difficulty,
            feasible=True,
            fail_reason="csv_parse_failed",
            cost_lo=COST_SHORT_REACT,
            cost_hi=COST_SHORT_REACT,
            expected_cost=COST_SHORT_REACT,
            lane="needs_react_fallback",
        )

    return TaskProfile(
        task_type=TaskType.DATA_ANALYSIS,
        difficulty_proxy=difficulty,
        feasible=True,
        fail_reason=None,
        cost_lo=COST_DETERMINISTIC,
        cost_hi=COST_DETERMINISTIC,
        expected_cost=COST_DETERMINISTIC,
        lane="cheap",
    )


def _triage_search(task: Task) -> TaskProfile:
    kb_dir = task.work_dir / "kb"
    n_md = 0
    if kb_dir.is_dir():
        n_md = sum(1 for _ in kb_dir.glob("*.md"))
    difficulty = _difficulty_from_length(n_md, easy=40, hard=80)

    answer, reason = try_solve_search(task.instruction, kb_dir)
    if answer is not None:
        return TaskProfile(
            task_type=TaskType.SEARCH,
            difficulty_proxy=difficulty,
            feasible=True,
            fail_reason=None,
            cost_lo=COST_DETERMINISTIC,
            cost_hi=COST_DETERMINISTIC,
            expected_cost=COST_DETERMINISTIC,
            lane="cheap",
        )

    if reason in {"kb_empty_or_missing", "person_missing", "employer_missing", "company_missing", "born_missing", "hq_missing"}:
        return _skip(task.task_type, difficulty, reason)

    # Unrecognized question template → short react fallback.
    return TaskProfile(
        task_type=TaskType.SEARCH,
        difficulty_proxy=difficulty,
        feasible=True,
        fail_reason=reason or "search_parse_failed",
        cost_lo=COST_SHORT_REACT,
        cost_hi=COST_SHORT_REACT,
        expected_cost=COST_SHORT_REACT,
        lane="needs_react_fallback",
    )


_FOREIGN_IMPORT_RE = re.compile(
    r"^(?:from\s+(?!mylib\b|unittest\b)\w+|import\s+(?!mylib\b|unittest\b)\w+)",
    re.MULTILINE,
)


def _triage_coding(task: Task) -> TaskProfile:
    work = task.work_dir
    lib = work / "mylib.py"
    test = work / "test_mylib.py"
    nlines = 0
    if lib.is_file():
        try:
            nlines = len(lib.read_text(encoding="utf-8").splitlines())
        except OSError:
            nlines = 0
    difficulty = _difficulty_from_length(nlines, easy=30, hard=80)

    reason = preflight_coding(work)
    if reason:
        return _skip(task.task_type, difficulty, reason)

    # Empty / trivial tests or foreign imports → unsolvable without burning budget.
    try:
        test_src = test.read_text(encoding="utf-8")
    except OSError:
        return _skip(task.task_type, difficulty, "missing_tests")

    if not test_src.strip() or "assert" not in test_src.lower():
        return _skip(task.task_type, difficulty, "empty_or_useless_tests")

    if _FOREIGN_IMPORT_RE.search(test_src) and "import mylib" not in test_src and "from mylib" not in test_src:
        return _skip(task.task_type, difficulty, "foreign_test_import")

    # Heuristic first; reserve react band when source looks hard.
    if difficulty == "hard":
        expected = COST_CODING_REACT
        cost_lo, cost_hi = COST_CODING_HEURISTIC, COST_CODING_REACT
    else:
        expected = COST_CODING_HEURISTIC
        cost_lo, cost_hi = COST_CODING_HEURISTIC, COST_CODING_REACT

    return TaskProfile(
        task_type=TaskType.CODING,
        difficulty_proxy=difficulty,
        feasible=True,
        fail_reason=None,
        cost_lo=cost_lo,
        cost_hi=cost_hi,
        expected_cost=expected,
        lane="coding",
    )


def _skip(task_type: TaskType, difficulty: str, reason: str) -> TaskProfile:
    return TaskProfile(
        task_type=task_type,
        difficulty_proxy=difficulty,
        feasible=False,
        fail_reason=reason,
        cost_lo=COST_SKIP,
        cost_hi=COST_SKIP,
        expected_cost=COST_SKIP,
        lane="skip",
    )
