"""Expected-cost estimates for budget-aware scheduling."""

from __future__ import annotations

from pathlib import Path

from harness.types import Task, TaskType

# Median-ish costs in output-token units from dry-run heuristics.
# Deterministic solvers cost ~0; coding/react is the expensive path.
DEFAULT_COST: dict[TaskType, int] = {
    TaskType.MATH: 1,
    TaskType.DATA_ANALYSIS: 1,
    TaskType.SEARCH: 1,
    TaskType.CODING: 2500,
}


def feature_adjust(task: Task) -> float:
    """Simple multiplicative adjustment from task features."""
    instr_len = len(task.instruction)
    factor = 1.0
    if instr_len > 800:
        factor *= 1.2
    work = task.work_dir
    if task.task_type == TaskType.SEARCH:
        kb = work / "kb"
        if kb.is_dir():
            n = sum(1 for _ in kb.glob("*.md"))
            if n > 100:
                factor *= 1.1
            if n == 0:
                factor *= 0.2  # likely unsolvable / cheap skip
    elif task.task_type == TaskType.DATA_ANALYSIS:
        csv_path = work / "sales.csv"
        if csv_path.is_file():
            try:
                lines = sum(1 for _ in csv_path.open("r", encoding="utf-8"))
            except OSError:
                lines = 0
            if lines > 500:
                factor *= 1.1
        else:
            factor *= 0.2
    elif task.task_type == TaskType.CODING:
        lib = work / "mylib.py"
        if lib.is_file():
            try:
                nlines = len(lib.read_text(encoding="utf-8").splitlines())
            except OSError:
                nlines = 50
            factor *= 1.0 + min(nlines, 200) / 200.0
    return factor


def expected_cost(task: Task) -> int:
    base = DEFAULT_COST.get(task.task_type, 100)
    return max(1, int(base * feature_adjust(task)))


def value_score(task: Task) -> float:
    """Higher is better: reward per expected token."""
    return task.reward / max(expected_cost(task), 1)
