#!/usr/bin/env python3
"""Score harness answers against pack manifest.jsonl.

Writes out/eval_report.json + out/eval_report.md (per-task table + fails).
Does not call the LLM.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]


def _norm(text: str, expected: str) -> str:
    s = (text or "").strip()
    exp = (expected or "").strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", exp.replace(",", "")):
        return s.replace(",", "").replace(" ", "")
    return s


def _read_answer(task_dir: Path) -> Optional[str]:
    for rel in ("answer.txt", "workspace/answer.txt"):
        p = task_dir / rel
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
    return None


def _load_manifest(tasks_dir: Path) -> list[dict[str, Any]]:
    path = tasks_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"manifest.jsonl not found in {tasks_dir}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_journal(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        tid = str(row.get("task_id") or "")
        if tid:
            out[tid] = row
    return out


def _verdict(solvable: bool, expected: str, got: Optional[str]) -> str:
    if not solvable or expected == "UNSOLVABLE":
        if got is None or not str(got).strip():
            return "unsolvable_ok"
        return "unsolvable_fp"
    if got is None or not str(got).strip():
        return "missing"
    if _norm(got, expected) == _norm(expected, expected):
        return "correct"
    return "wrong"


def _esc(cell: Any) -> str:
    s = "" if cell is None else str(cell)
    return s.replace("|", "\\|").replace("\n", " ").strip()


def score_pack(
    tasks_dir: Path,
    journal_path: Optional[Path],
    out_json: Path,
    out_md: Path,
) -> dict[str, Any]:
    manifest = _load_manifest(tasks_dir)
    journal = _load_journal(journal_path)
    tasks: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_diff: dict[str, Counter[str]] = defaultdict(Counter)
    reward_claimed = 0
    reward_correct = 0
    total_cost = 0

    for row in manifest:
        tid = str(row["task_id"])
        task_dir = tasks_dir / tid
        expected = str(row.get("expected") or "")
        solvable = bool(row.get("solvable"))
        got_raw = _read_answer(task_dir)
        got = got_raw.strip() if got_raw is not None else None
        v = _verdict(solvable, expected, got)
        counts[v] += 1
        ttype = str(row.get("task_type") or "?")
        diff = str(row.get("difficulty") or "?")
        by_type[ttype][v] += 1
        by_diff[diff][v] += 1
        j = journal.get(tid, {})
        cost = int(j.get("cost") or 0)
        total_cost += cost
        reward = int(row.get("reward") or 0)
        status = j.get("status")
        if status == "solved":
            reward_claimed += reward
        if v == "correct":
            reward_correct += reward
        tasks.append(
            {
                "task_id": tid,
                "task_type": ttype,
                "difficulty": diff,
                "solvable": solvable,
                "verdict": v,
                "expected": expected,
                "got": got if got is not None else "",
                "harness_status": status,
                "cost": cost,
                "model": j.get("model"),
                "reward": reward,
                "source": row.get("source"),
                "fail_mode": row.get("fail_mode"),
            }
        )

    summary = {
        "tasks": len(tasks),
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "missing": counts["missing"],
        "unsolvable_ok": counts["unsolvable_ok"],
        "unsolvable_fp": counts["unsolvable_fp"],
        "reward_claimed": reward_claimed,
        "reward_correct": reward_correct,
        "total_cost": total_cost,
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "by_difficulty": {k: dict(v) for k, v in sorted(by_diff.items())},
    }
    report = {"summary": summary, "tasks": tasks}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Eval report\n")
    lines.append(f"Tasks dir: `{tasks_dir}`\n")
    lines.append("## Summary\n")
    lines.append(
        f"- tasks={summary['tasks']} correct={summary['correct']} wrong={summary['wrong']} "
        f"missing={summary['missing']} unsolvable_ok={summary['unsolvable_ok']} "
        f"unsolvable_fp={summary['unsolvable_fp']}"
    )
    lines.append(
        f"- reward_claimed={summary['reward_claimed']} reward_correct={summary['reward_correct']} "
        f"total_cost={summary['total_cost']}"
    )
    lines.append("\n### By type\n")
    for t, c in summary["by_type"].items():
        lines.append(f"- `{t}`: {c}")
    lines.append("\n### By difficulty\n")
    for d, c in summary["by_difficulty"].items():
        lines.append(f"- `{d}`: {c}")

    lines.append("\n## Per-task\n")
    lines.append(
        "| task_id | type | difficulty | solvable | verdict | expected | got | "
        "harness_status | cost | model | reward |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in tasks:
        lines.append(
            "| "
            + " | ".join(
                [
                    _esc(t["task_id"]),
                    _esc(t["task_type"]),
                    _esc(t["difficulty"]),
                    "yes" if t["solvable"] else "no",
                    _esc(t["verdict"]),
                    _esc(t["expected"])[:40],
                    _esc(t["got"])[:40],
                    _esc(t["harness_status"]),
                    _esc(t["cost"]),
                    _esc(t["model"]),
                    _esc(t["reward"]),
                ]
            )
            + " |"
        )

    fails = [t for t in tasks if t["verdict"] in {"wrong", "missing", "unsolvable_fp"}]
    lines.append("\n## Fails\n")
    if not fails:
        lines.append("_none_\n")
    else:
        for t in fails:
            lines.append(
                f"- `{t['task_id']}` [{t['verdict']}] expected=`{t['expected']}` "
                f"got=`{t['got']}` status=`{t['harness_status']}` cost={t['cost']}"
            )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"tasks={summary['tasks']} correct={summary['correct']} wrong={summary['wrong']} "
        f"missing={summary['missing']} unsolvable_ok={summary['unsolvable_ok']} "
        f"unsolvable_fp={summary['unsolvable_fp']}"
    )
    print(
        f"reward_claimed={summary['reward_claimed']} reward_correct={summary['reward_correct']} "
        f"cost={summary['total_cost']}"
    )
    print(f"report: {out_md}")
    print(f"json:   {out_json}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-dir", type=Path, default=ROOT / "task_packs" / "test-100")
    p.add_argument("--journal", type=Path, default=ROOT / "out" / "cost_journal.jsonl")
    p.add_argument("--out-json", type=Path, default=ROOT / "out" / "eval_report.json")
    p.add_argument("--out-md", type=Path, default=ROOT / "out" / "eval_report.md")
    args = p.parse_args()
    score_pack(args.tasks_dir, args.journal, args.out_json, args.out_md)


if __name__ == "__main__":
    main()
