"""Tabular features and structural skip flags. No LLM, no expected/reward.

Standalone: notebooks load this file by path, so it must not import the harness
package (that pulls openJiuwen).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Optional

FEATURE_SCHEMA_VERSION = 1

# Bands match harness.estimate. The heuristic stays here so Kaggle and the
# runtime share one formula when no cost model is loaded.
COST_SHORT = 800
COST_CODING = 2500

FEATURE_NAMES: tuple[str, ...] = (
    "is_math",
    "is_data_analysis",
    "is_search",
    "is_coding",
    "instr_chars",
    "instr_words",
    "instr_lines",
    "digit_ratio",
    "n_numbers",
    "n_questions",
    "has_latex",
    "has_code_fence",
    "mentions_answer_txt",
    "da_named_file_missing",
    "da_no_csv",
    "da_broken_header",
    "da_column_missing",
    "search_kb_empty",
    "search_kb_tiny",
    "search_token_overlap",
    "coding_no_mylib",
    "coding_empty_test",
    "coding_foreign_import",
    "coding_ast_fail",
    "math_arith_steps",
    "csv_lines",
    "csv_cols",
    "kb_files",
    "kb_chars",
    "code_lines",
    "code_defs",
    "code_asserts",
)

# Hard skip only when a missing or wrong file makes the task impossible.
# A syntax error in mylib.py is a feature, not a skip: it may be the bug to fix.
# Tiny kb is a feature, not a skip: a short real note would be a false skip.
_HARD_SKIP_FLAGS: dict[str, tuple[str, ...]] = {
    "data_analysis": (
        "da_named_file_missing",
        "da_no_csv",
    ),
    "search": ("search_kb_empty",),
    "coding": (
        "coding_no_mylib",
        "coding_empty_test",
        "coding_foreign_import",
    ),
}

_FILE_RE = re.compile(r"\b([\w./-]+\.(?:csv|tsv|json|txt|md|py))\b", re.IGNORECASE)
_COL_RE = re.compile(r"\bcolumn\s+([A-Za-z_][\w]*)", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)",
    re.MULTILINE,
)
_ARITH_RE = re.compile(
    r"\b(?:split|double|doubled|add|added|receive|receives|ship|ships|"
    r"merge|merged|minus|plus|times|divided|subtract|multiply)\b",
    re.IGNORECASE,
)
_DEF_RE = re.compile(r"^\s*def\s+\w+", re.MULTILINE)
_ASSERT_RE = re.compile(r"\bassert\b|\bdef\s+test_\w+", re.IGNORECASE)

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "you", "your", "only", "file", "answer", "when", "done", "write", "final",
    "current", "directory", "example", "bash", "echo", "python", "compute",
    "work", "inside", "using", "use", "what", "which", "who", "how", "many",
    "does", "have", "has", "into", "then", "than", "after", "before", "about",
}

_CSV_LINE_CAP = 5000
_KB_CHAR_CAP = 400_000
_KB_TINY_CHARS = 200


def _task_type(task: Any) -> str:
    raw = getattr(task, "task_type", "")
    return raw.value if hasattr(raw, "value") else str(raw)


def _work_dir(task: Any) -> Path:
    work = getattr(task, "work_dir", None)
    if work is not None:
        return Path(work)
    return Path(getattr(task, "task_dir"))


def _read_text(path: Path, cap: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return data[:cap]


def _named_files(instruction: str) -> list[str]:
    found: list[str] = []
    for match in _FILE_RE.findall(instruction):
        if Path(match).name.lower() == "answer.txt":
            continue
        found.append(match)
    return found


def _file_exists(work: Path, name: str) -> bool:
    if (work / name).is_file():
        return True
    return (work / Path(name).name).is_file()


def _table_paths(work: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.csv", "*.tsv", "*.json"):
        paths.extend(p for p in work.glob(pattern) if p.is_file())
    return sorted(paths)


def _csv_shape(path: Path) -> tuple[int, int, bool, list[str]]:
    """Return line count (capped), column count, broken-header flag, header cells."""
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return 0, 0, True, []
    lines = 0
    header: list[str] = []
    with fh:
        for line in fh:
            if lines == 0:
                header = [cell.strip() for cell in line.rstrip("\n").split(",")]
            lines += 1
            if lines >= _CSV_LINE_CAP:
                break
    broken = lines == 0 or not any(header)
    cols = 0 if broken else len(header)
    return lines, cols, broken, header


def _kb_stats(work: Path) -> tuple[int, int, str]:
    kb = work / "kb"
    if not kb.is_dir():
        return 0, 0, ""
    files = sorted(p for p in kb.glob("*.md") if p.is_file())
    chunks: list[str] = []
    total = 0
    for path in files:
        text = _read_text(path, _KB_CHAR_CAP - total)
        if not text:
            continue
        chunks.append(text)
        total += len(text)
        if total >= _KB_CHAR_CAP:
            break
    body = "\n".join(chunks)
    nonempty = sum(1 for chunk in chunks if chunk.strip())
    return nonempty, len(body), body


def _token_overlap(instruction: str, kb_text: str) -> float:
    question = {w.lower() for w in _WORD_RE.findall(instruction) if w.lower() not in _STOP}
    if not question or not kb_text:
        return 0.0
    kb_words = {w.lower() for w in _WORD_RE.findall(kb_text)}
    return len(question & kb_words) / len(question)


def _coding_stats(work: Path) -> dict[str, float]:
    lib = work / "mylib.py"
    test = work / "test_mylib.py"
    lib_text = _read_text(lib, 200_000) if lib.is_file() else ""
    test_text = _read_text(test, 200_000) if test.is_file() else ""
    ast_fail = 0.0
    if lib.is_file():
        try:
            ast.parse(lib_text)
        except SyntaxError:
            ast_fail = 1.0
    imports = {name for name in _IMPORT_RE.findall(test_text)}
    foreign = 1.0 if imports and "mylib" not in imports else 0.0
    return {
        "coding_no_mylib": 0.0 if lib.is_file() else 1.0,
        "coding_empty_test": 0.0 if test_text.strip() else 1.0,
        "coding_foreign_import": foreign,
        "coding_ast_fail": ast_fail,
        "code_lines": float(lib_text.count("\n") + (1 if lib_text.strip() else 0)),
        "code_defs": float(len(_DEF_RE.findall(lib_text))),
        "code_asserts": float(len(_ASSERT_RE.findall(test_text))),
    }


def extract(task: Any) -> dict[str, float]:
    """Feature dict keyed by FEATURE_NAMES. Reward is intentionally absent."""
    task_type = _task_type(task)
    instruction = str(getattr(task, "instruction", "") or "")
    work = _work_dir(task)
    chars = len(instruction)
    words = re.findall(r"\S+", instruction)

    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats["is_math"] = 1.0 if task_type == "math" else 0.0
    feats["is_data_analysis"] = 1.0 if task_type == "data_analysis" else 0.0
    feats["is_search"] = 1.0 if task_type == "search" else 0.0
    feats["is_coding"] = 1.0 if task_type == "coding" else 0.0
    feats["instr_chars"] = float(chars)
    feats["instr_words"] = float(len(words))
    feats["instr_lines"] = float(instruction.count("\n") + (1 if instruction else 0))
    feats["digit_ratio"] = (sum(ch.isdigit() for ch in instruction) / chars) if chars else 0.0
    feats["n_numbers"] = float(len(_NUM_RE.findall(instruction)))
    feats["n_questions"] = float(instruction.count("?"))
    feats["has_latex"] = 1.0 if ("$" in instruction or "\\frac" in instruction) else 0.0
    feats["has_code_fence"] = 1.0 if "```" in instruction else 0.0
    feats["mentions_answer_txt"] = 1.0 if "answer.txt" in instruction.lower() else 0.0
    feats["math_arith_steps"] = float(len(_ARITH_RE.findall(instruction)))

    if task_type == "data_analysis":
        named = _named_files(instruction)
        if named and any(not _file_exists(work, name) for name in named):
            feats["da_named_file_missing"] = 1.0
        tables = _table_paths(work)
        if not tables:
            feats["da_no_csv"] = 1.0
        else:
            preferred = tables[0]
            for name in named:
                candidate = work / Path(name).name
                if candidate.is_file():
                    preferred = candidate
                    break
            if preferred.suffix.lower() in {".csv", ".tsv"}:
                lines, cols, broken, header = _csv_shape(preferred)
                feats["csv_lines"] = float(lines)
                feats["csv_cols"] = float(cols)
                feats["da_broken_header"] = 1.0 if broken else 0.0
                header_key = {cell.casefold() for cell in header if cell}
                missing_col = [
                    col for col in _COL_RE.findall(instruction)
                    if col.casefold() not in header_key
                ]
                if missing_col:
                    feats["da_column_missing"] = 1.0

    if task_type == "search":
        n_files, n_chars, body = _kb_stats(work)
        feats["kb_files"] = float(n_files)
        feats["kb_chars"] = float(n_chars)
        feats["search_kb_empty"] = 1.0 if n_files == 0 or not body.strip() else 0.0
        feats["search_kb_tiny"] = 1.0 if 0 < n_chars < _KB_TINY_CHARS else 0.0
        feats["search_token_overlap"] = _token_overlap(instruction, body)

    if task_type == "coding":
        feats.update(_coding_stats(work))

    return feats


def vector(task: Any) -> list[float]:
    feats = extract(task)
    return [float(feats[name]) for name in FEATURE_NAMES]


def structural_skip_reason(task: Any, feats: Optional[dict[str, float]] = None) -> Optional[str]:
    """Comma-joined hard-fail flags, or None. Math never hard-fails."""
    task_type = _task_type(task)
    flags = _HARD_SKIP_FLAGS.get(task_type)
    if not flags:
        return None
    row = feats if feats is not None else extract(task)
    hit = [name for name in flags if row.get(name, 0.0) >= 1.0]
    if not hit:
        return None
    return ",".join(hit)


def heuristic_cost(task: Any, feats: Optional[dict[str, float]] = None) -> int:
    """Same bands as harness.estimate.feature_adjust, from the feature row."""
    row = feats if feats is not None else extract(task)
    base = COST_CODING if row["is_coding"] >= 1.0 else COST_SHORT
    factor = 1.0
    if row["instr_chars"] > 800:
        factor *= 1.2
    if row["is_search"] >= 1.0 and row["kb_files"] > 100:
        factor *= 1.1
    if row["is_data_analysis"] >= 1.0 and row["csv_lines"] > 500:
        factor *= 1.1
    if row["is_coding"] >= 1.0 and row["coding_no_mylib"] < 1.0:
        factor *= 1.0 + min(row["code_lines"], 200.0) / 200.0
    return max(1, int(base * factor))
