"""Load task.json directories into Task objects."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from harness.domain.types import Task, TaskType


REQUIRED_TASK_FIELDS = ("task_id", "task_type", "instruction", "reward")


def load_task(task_dir: Path) -> Task:
    """Load a Task from ``task_dir/task.json``.

    Requires task_id, task_type, instruction, reward.
    Does not require expected/check (example-only fields).
    """
    task_dir = Path(task_dir).resolve()
    path = task_dir / "task.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing task.json in {task_dir}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    missing = [k for k in REQUIRED_TASK_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required fields: {missing}")

    try:
        task_type = TaskType(raw["task_type"])
    except ValueError as exc:
        raise ValueError(f"{path}: unknown task_type {raw['task_type']!r}") from exc

    reward = raw["reward"]
    if not isinstance(reward, int) or isinstance(reward, bool):
        raise ValueError(f"{path}: reward must be an int, got {type(reward).__name__}")

    known = set(REQUIRED_TASK_FIELDS)
    extras = {k: v for k, v in raw.items() if k not in known}

    return Task(
        task_id=str(raw["task_id"]),
        task_type=task_type,
        instruction=str(raw["instruction"]),
        reward=reward,
        task_dir=task_dir,
        extras=extras,
    )


def discover_tasks(tasks_dir: Path | str) -> list[Task]:
    """Discover and load every subdirectory that contains task.json."""
    root = Path(tasks_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"tasks directory not found: {root}")

    tasks: list[Task] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith("."):
            continue
        if not child.is_dir():
            continue
        if not (child / "task.json").is_file():
            continue
        tasks.append(load_task(child))
    return tasks


def resolve_tasks_dir(
    cli_tasks_dir: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> Path:
    """Prefer ``--tasks-dir``, fall back to env ``TASKS``."""
    environ = env if env is not None else os.environ
    if cli_tasks_dir:
        return Path(cli_tasks_dir)
    tasks = environ.get("TASKS")
    if tasks:
        return Path(tasks)
    raise SystemExit(
        "tasks directory required: pass --tasks-dir or set env TASKS"
    )
