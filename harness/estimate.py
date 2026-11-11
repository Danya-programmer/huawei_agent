"""Expected-cost estimates for budget-aware scheduling."""

from __future__ import annotations

from harness.types import Task, TaskType

# Output-token cost bands (medians; fit_estimates.py can refresh from journal).
COST_DETERMINISTIC = 1
COST_SKIP = 0
COST_SHORT_REACT = 800
COST_CODING_HEURISTIC = 1
COST_CODING_REACT = 2500
COST_CODING_REACT_REASON = 6000

# Legacy per-type defaults (used only if triage is unavailable).
DEFAULT_COST: dict[TaskType, int] = {
    TaskType.MATH: COST_DETERMINISTIC,
    TaskType.DATA_ANALYSIS: COST_DETERMINISTIC,
    TaskType.SEARCH: COST_DETERMINISTIC,
    TaskType.CODING: COST_CODING_REACT,
}

COST_BANDS: dict[str, int] = {
    "deterministic": COST_DETERMINISTIC,
    "skip": COST_SKIP,
    "short_react": COST_SHORT_REACT,
    "coding_heuristic": COST_CODING_HEURISTIC,
    "coding_react": COST_CODING_REACT,
    "coding_react_reason": COST_CODING_REACT_REASON,
}


def feature_adjust(task: Task) -> float:
    """Simple multiplicative adjustment from task features (legacy helper)."""
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
    """Prefer triage band; fall back to type defaults."""
    try:
        from harness.triage import triage

        return int(triage(task).expected_cost)
    except Exception:  # noqa: BLE001
        base = DEFAULT_COST.get(task.task_type, 100)
        return max(0, int(base * feature_adjust(task)))


def value_score(task: Task) -> float:
    """Higher is better: reward per expected token (phase-B style)."""
    cost = max(expected_cost(task), 1)
    return task.reward / cost
