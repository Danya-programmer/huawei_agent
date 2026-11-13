#!/usr/bin/env python3
"""Jury entrypoint: ``python /app/run_agent.py --tasks-dir /data/tasks``.

Also accepts env ``TASKS`` as a fallback for the tasks directory.
API keys must come from the environment (``LLM_API_KEY`` / ``LLM_BASE_URL``);
nothing is hardcoded here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python run_agent.py`` from the repo root without installing the package.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os  # noqa: E402

from harness.infra.io import CostJournal, resolve_tasks_dir  # noqa: E402
from harness.runtime.scheduler import run_tasks  # noqa: E402
from harness.agents.solvers import build_default_registry  # noqa: E402


def _load_dotenv(path: Path | None = None) -> None:
    """Lightweight .env loader (no dependency). Does not override existing env."""
    env_path = path or (_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Budget-aware openJiuwen harness",
    )
    parser.add_argument(
        "--tasks-dir",
        default=None,
        help="Directory with task subfolders (fallback: env TASKS)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_arg_parser().parse_args(argv)
    tasks_dir = resolve_tasks_dir(args.tasks_dir)

    solvers = build_default_registry()
    journal = CostJournal.from_env()
    max_workers = int(os.environ.get("MAX_WORKERS", "2") or "2")

    summary = run_tasks(
        tasks_dir,
        solvers=solvers,
        journal=journal,
        max_workers=max_workers,
    )

    print(
        f"done: seen={summary.tasks_seen} solved={summary.solved} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"reward={summary.total_reward} cost={summary.total_cost}"
    )
    if journal.out_path is not None:
        print(f"journal: {journal.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
