#!/usr/bin/env python3
"""Evaluate harness on a generated train set against manifest.jsonl.

Usage:
  python scripts/eval_trainset.py --tasks-dir train-100 --budget 50000
  python scripts/eval_trainset.py --tasks-dir train-100 --budget 50000 --with-react

Without ``--with-react``, sets ``HARNESS_USE_REACT=0`` so the deterministic /
heuristic path is measured (no LLM).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.io import CostJournal  # noqa: E402
from harness.scheduler import run_tasks  # noqa: E402
from harness.solvers import build_default_registry  # noqa: E402
from harness.types import Status  # noqa: E402


def _load_manifest(tasks_dir: Path) -> dict[str, dict[str, Any]]:
    path = tasks_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"manifest.jsonl not found in {tasks_dir}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = str(row["task_id"])
            rows[tid] = row
    return rows


def _read_answer(task_dir: Path) -> Optional[str]:
    for candidate in (task_dir / "answer.txt", task_dir / "workspace" / "answer.txt"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return None


def _ensure_trainset(tasks_dir: Path, *, n: int = 100, seed: int = 42) -> None:
    """Regenerate train set if missing or incomplete."""
    manifest = tasks_dir / "manifest.jsonl"
    if tasks_dir.is_dir() and manifest.is_file():
        # Quick sanity: at least n task dirs besides manifest.
        n_dirs = sum(1 for p in tasks_dir.iterdir() if p.is_dir())
        if n_dirs >= n:
            return
    print(
        f"regenerating {tasks_dir.name}: "
        f"python3 -m harness.generators --out {tasks_dir} --n {n} --seed {seed} "
        f"--unsolvable-ratio 0.2",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "harness.generators",
            "--out",
            str(tasks_dir),
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--unsolvable-ratio",
            "0.2",
        ],
        cwd=str(ROOT),
        check=True,
    )


def _answers_match(got: Optional[str], expected: Any) -> bool:
    if got is None:
        return False
    exp = "" if expected is None else str(expected).strip()
    return got.strip() == exp


def evaluate(
    tasks_dir: Path,
    *,
    budget: int,
    with_react: bool,
    out_dir: Optional[Path],
) -> dict[str, Any]:
    if not with_react:
        os.environ["HARNESS_USE_REACT"] = "0"

    manifest = _load_manifest(tasks_dir)
    journal_path = (out_dir or tasks_dir) / "cost_journal.jsonl"
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Truncate previous journal for a clean run.
    if journal_path.is_file():
        journal_path.unlink()

    journal = CostJournal(journal_path)
    t0 = time.perf_counter()
    summary = run_tasks(
        tasks_dir,
        solvers=build_default_registry(),
        journal=journal,
        budget=budget,
    )
    elapsed_s = time.perf_counter() - t0

    results_by_id = {task.task_id: (task, result) for task, result in summary.results}

    solved = 0
    solvable_miss = 0
    unsolvable_correct_skip = 0
    false_skip = 0
    false_attempt = 0
    false_attempt_cost = 0
    total_reward = 0
    total_cost = 0
    cost_by_difficulty: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "cost": 0}
    )
    details: list[dict[str, Any]] = []

    for tid, meta in sorted(manifest.items()):
        solvable = bool(meta.get("solvable"))
        expected = meta.get("expected")
        difficulty = str(meta.get("difficulty") or "unknown")
        reward = int(meta.get("reward") or 0)

        pair = results_by_id.get(tid)
        if pair is None:
            status = "missing"
            cost = 0
            answer = _read_answer(tasks_dir / tid)
            result_status = None
            error = "not in run summary"
        else:
            task, result = pair
            status = result.status.value
            cost = int(result.cost or 0)
            answer = result.answer
            if answer is None:
                answer = _read_answer(task.task_dir)
            result_status = result.status
            error = result.error

        total_cost += cost
        cost_by_difficulty[difficulty]["count"] += 1
        cost_by_difficulty[difficulty]["cost"] += cost

        row: dict[str, Any] = {
            "task_id": tid,
            "solvable": solvable,
            "difficulty": difficulty,
            "status": status,
            "cost": cost,
            "expected": expected,
            "answer": answer,
            "error": error,
        }

        if solvable:
            ok = (
                result_status == Status.SOLVED
                and _answers_match(answer, expected)
            )
            # Coding expected is DONE; others compare to manifest expected.
            if ok:
                solved += 1
                total_reward += reward
                row["verdict"] = "solved"
            else:
                solvable_miss += 1
                if result_status in (Status.SKIPPED, Status.UNSOLVABLE, Status.BUDGET):
                    false_skip += 1
                    row["verdict"] = "false_skip"
                else:
                    row["verdict"] = "miss"
        else:
            # Correct: marked unsolvable/skipped at cost 0 (no burn).
            if (
                result_status in (Status.UNSOLVABLE, Status.SKIPPED)
                and cost == 0
            ):
                unsolvable_correct_skip += 1
                row["verdict"] = "unsolvable_correct_skip"
            else:
                false_attempt += 1
                false_attempt_cost += cost
                row["verdict"] = "false_attempt"

        details.append(row)

    n_solvable = sum(1 for m in manifest.values() if m.get("solvable"))
    n_unsolvable = len(manifest) - n_solvable

    cost_by_diff_out = {
        d: {
            "count": int(v["count"]),
            "total_cost": int(v["cost"]),
            "avg_cost": round(v["cost"] / v["count"], 3) if v["count"] else 0.0,
        }
        for d, v in sorted(cost_by_difficulty.items())
    }

    acceptance = {
        "all_solvable_closed": solved == n_solvable and solvable_miss == 0,
        "unsolvable_skip_ge_18": unsolvable_correct_skip >= 18,
        "under_5_min": elapsed_s < 300.0,
    }
    acceptance["passed"] = all(acceptance.values())

    report = {
        "tasks_dir": str(tasks_dir),
        "budget": budget,
        "with_react": with_react,
        "n_tasks": len(manifest),
        "n_solvable": n_solvable,
        "n_unsolvable": n_unsolvable,
        "solved": solved,
        "solvable_miss": solvable_miss,
        "unsolvable_correct_skip": unsolvable_correct_skip,
        "false_skip": false_skip,
        "false_attempt": false_attempt,
        "false_attempt_cost": false_attempt_cost,
        "total_reward": total_reward,
        "total_cost": total_cost,
        "harness_total_reward": summary.total_reward,
        "harness_total_cost": summary.total_cost,
        "elapsed_s": round(elapsed_s, 3),
        "cost_by_difficulty": cost_by_diff_out,
        "journal": str(journal_path) if journal_path.is_file() else None,
        "acceptance": acceptance,
        "misses": [d for d in details if d["verdict"] not in {"solved", "unsolvable_correct_skip"}],
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("=== eval_trainset summary ===")
    print(
        f"tasks={report['n_tasks']} solvable={report['n_solvable']} "
        f"unsolvable={report['n_unsolvable']} budget={report['budget']} "
        f"react={report['with_react']}"
    )
    print(
        f"solved={report['solved']}/{report['n_solvable']} "
        f"unsolvable_correct_skip={report['unsolvable_correct_skip']}/{report['n_unsolvable']} "
        f"false_skip={report['false_skip']} false_attempt={report['false_attempt']} "
        f"(false_attempt_cost={report['false_attempt_cost']})"
    )
    print(
        f"total_reward={report['total_reward']} total_cost={report['total_cost']} "
        f"elapsed_s={report['elapsed_s']}"
    )
    print("cost_by_difficulty:")
    for diff, stats in report["cost_by_difficulty"].items():
        print(
            f"  {diff}: count={stats['count']} "
            f"total={stats['total_cost']} avg={stats['avg_cost']}"
        )
    acc = report["acceptance"]
    print(
        "acceptance: "
        f"all_solvable_closed={acc['all_solvable_closed']} "
        f"unsolvable_skip_ge_18={acc['unsolvable_skip_ge_18']} "
        f"under_5_min={acc['under_5_min']} "
        f"PASSED={acc['passed']}"
    )
    misses = report.get("misses") or []
    if misses:
        print(f"misses ({len(misses)}):")
        for m in misses[:40]:
            print(
                f"  {m['task_id']}: verdict={m['verdict']} status={m['status']} "
                f"cost={m['cost']} expected={m['expected']!r} answer={m['answer']!r} "
                f"error={m.get('error')!r}"
            )
        if len(misses) > 40:
            print(f"  ... and {len(misses) - 40} more")
    print("--- JSON ---")
    # Omit full miss list duplication beyond what's useful; keep compact JSON.
    compact = {k: v for k, v in report.items() if k != "misses"}
    compact["n_misses"] = len(misses)
    compact["miss_ids"] = [m["task_id"] for m in misses]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-dir",
        default="train-100",
        help="Directory with generated tasks + manifest.jsonl",
    )
    parser.add_argument("--budget", type=int, default=50000)
    parser.add_argument(
        "--with-react",
        action="store_true",
        help="Allow ReAct/LLM (default: disable via HARNESS_USE_REACT=0)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for cost_journal.jsonl (default: tasks-dir)",
    )
    parser.add_argument(
        "--no-regen",
        action="store_true",
        help="Do not regenerate train set if incomplete",
    )
    args = parser.parse_args(argv)

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_absolute():
        tasks_dir = ROOT / tasks_dir
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    if not args.no_regen:
        _ensure_trainset(tasks_dir)

    report = evaluate(
        tasks_dir,
        budget=args.budget,
        with_react=args.with_react,
        out_dir=out_dir,
    )
    _print_report(report)
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
