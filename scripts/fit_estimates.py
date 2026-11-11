#!/usr/bin/env python3
"""Fit cost-band medians from a cost journal JSONL.

Safe no-op when the journal (or train-100) is missing:

  python scripts/fit_estimates.py
  python scripts/fit_estimates.py --journal trajectories/cost_journal.jsonl
  python scripts/fit_estimates.py --journal OUT/cost_journal.jsonl --write

Prints suggested medians; with ``--write``, updates constants in harness/estimate.py
when enough samples exist.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.estimate import (  # noqa: E402
    COST_BANDS,
    COST_CODING_HEURISTIC,
    COST_CODING_REACT,
    COST_CODING_REACT_REASON,
    COST_DETERMINISTIC,
    COST_SHORT_REACT,
    COST_SKIP,
)

ESTIMATE_PATH = ROOT / "harness" / "estimate.py"

# Map journal fields → band keys used in estimate.py
_BAND_KEYS = {
    "deterministic": "COST_DETERMINISTIC",
    "skip": "COST_SKIP",
    "short_react": "COST_SHORT_REACT",
    "coding_heuristic": "COST_CODING_HEURISTIC",
    "coding_react": "COST_CODING_REACT",
    "coding_react_reason": "COST_CODING_REACT_REASON",
}


def _default_journal_candidates() -> list[Path]:
    env_out = __import__("os").environ.get("OUT")
    cands: list[Path] = []
    if env_out:
        p = Path(env_out)
        cands.append(p if p.suffix in {".jsonl", ".json"} else p / "cost_journal.jsonl")
    cands.extend(
        [
            ROOT / "trajectories" / "cost_journal.jsonl",
            ROOT / "train-100" / "cost_journal.jsonl",
            ROOT / "out" / "cost_journal.jsonl",
        ]
    )
    return cands


def _resolve_journal(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    for cand in _default_journal_candidates():
        if cand.is_file():
            return cand
    return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _classify_band(row: dict[str, Any]) -> Optional[str]:
    status = str(row.get("status") or "")
    task_type = str(row.get("task_type") or "")
    cost = int(row.get("cost") or 0)
    method = str(row.get("method") or "")
    solver = str(row.get("solver") or "")
    lane = str(row.get("lane") or "")

    if status in {"unsolvable", "skipped", "budget"} or cost == 0 and lane == "skip":
        if status in {"unsolvable", "skipped"}:
            return "skip"
    if cost == 0 or method == "deterministic" or (
        task_type in {"math", "data_analysis", "search"} and cost <= 5
    ):
        if method in {"deterministic", "heuristic"} or (
            task_type != "coding" and cost <= 5 and status == "solved"
        ):
            return "deterministic" if task_type != "coding" else "coding_heuristic"
    if method == "heuristic" or (task_type == "coding" and cost <= 5):
        return "coding_heuristic"
    if "reason" in str(row.get("model") or "") or cost >= COST_CODING_REACT_REASON * 0.5:
        if task_type == "coding" or solver == "coding":
            return "coding_react_reason"
    if task_type == "coding" or solver == "coding" or method == "react":
        if cost >= COST_SHORT_REACT:
            return "coding_react"
        return "coding_heuristic"
    if lane == "needs_react_fallback" or COST_SHORT_REACT * 0.5 <= cost < COST_CODING_REACT:
        return "short_react"
    return None


def _median(values: list[int]) -> Optional[int]:
    if not values:
        return None
    return int(statistics.median(values))


def fit(records: list[dict[str, Any]]) -> dict[str, Optional[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in records:
        band = _classify_band(row)
        if band is None:
            continue
        buckets[band].append(int(row.get("cost") or 0))

    suggested = dict(COST_BANDS)
    for band in _BAND_KEYS:
        med = _median(buckets.get(band, []))
        if med is None:
            continue
        # Journal often records 0 for local solvers; keep a 1-unit schedule stub.
        if band in {"deterministic", "coding_heuristic"} and med <= 0:
            med = COST_DETERMINISTIC if band == "deterministic" else COST_CODING_HEURISTIC
        suggested[band] = med
    # Always keep skip at 0
    suggested["skip"] = COST_SKIP
    return {k: suggested.get(k) for k in _BAND_KEYS}


def _print_report(
    journal: Optional[Path],
    n_records: int,
    suggested: dict[str, Optional[int]],
) -> None:
    print("fit_estimates")
    if journal is None:
        print("  journal: (missing) — no-op, keeping defaults")
    else:
        print(f"  journal: {journal} ({n_records} records)")
    print("  suggested medians:")
    defaults = {
        "deterministic": COST_DETERMINISTIC,
        "skip": COST_SKIP,
        "short_react": COST_SHORT_REACT,
        "coding_heuristic": COST_CODING_HEURISTIC,
        "coding_react": COST_CODING_REACT,
        "coding_react_reason": COST_CODING_REACT_REASON,
    }
    for band, default in defaults.items():
        val = suggested.get(band, default)
        print(f"    {band:22s} {val}  (default {default})")


def _write_estimate_py(suggested: dict[str, Optional[int]]) -> bool:
    """Rewrite COST_* integer assignments when we have non-default suggestions."""
    if not ESTIMATE_PATH.is_file():
        print(f"  skip write: {ESTIMATE_PATH} missing", file=sys.stderr)
        return False
    text = ESTIMATE_PATH.read_text(encoding="utf-8")
    mapping = {
        "COST_DETERMINISTIC": suggested.get("deterministic"),
        "COST_SKIP": suggested.get("skip"),
        "COST_SHORT_REACT": suggested.get("short_react"),
        "COST_CODING_HEURISTIC": suggested.get("coding_heuristic"),
        "COST_CODING_REACT": suggested.get("coding_react"),
        "COST_CODING_REACT_REASON": suggested.get("coding_react_reason"),
    }
    changed = False
    for name, value in mapping.items():
        if value is None:
            continue
        pattern = re.compile(rf"^({name}\s*=\s*)\d+", re.MULTILINE)
        new_text, n = pattern.subn(rf"\g<1>{int(value)}", text, count=1)
        if n:
            text = new_text
            changed = True
    if changed:
        ESTIMATE_PATH.write_text(text, encoding="utf-8")
        print(f"  updated {ESTIMATE_PATH}")
    else:
        print("  write: no constant lines updated")
    return changed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal",
        help="Path to cost_journal.jsonl (optional; auto-discover if omitted)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update harness/estimate.py COST_* constants from medians",
    )
    args = parser.parse_args(argv)

    journal = _resolve_journal(args.journal)
    if journal is None:
        _print_report(None, 0, {k: COST_BANDS[k] for k in _BAND_KEYS})
        if args.journal:
            print(f"  warning: journal not found: {args.journal}", file=sys.stderr)
        return 0

    records = _load_records(journal)
    if not records:
        _print_report(journal, 0, {k: COST_BANDS[k] for k in _BAND_KEYS})
        print("  warning: journal empty — keeping defaults")
        return 0

    suggested = fit(records)
    _print_report(journal, len(records), suggested)
    if args.write:
        _write_estimate_py(suggested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
