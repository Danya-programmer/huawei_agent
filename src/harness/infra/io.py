"""Atomic answer.txt writes and cost journal."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from harness.domain.tasks import discover_tasks, load_task, resolve_tasks_dir
from harness.domain.types import Result, Status, Task

__all__ = [
    "CostJournal",
    "atomic_write_answer",
    "discover_tasks",
    "load_task",
    "resolve_tasks_dir",
]


def atomic_write_answer(task_dir: Path | str, answer: str) -> Path:
    """Atomically write ``answer.txt`` into the task directory.

    Uses a same-directory temp file + os.replace so readers never see a
    partial answer. Trailing newlines are normalized to a single ``\\n``.
    """
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    target = task_dir / "answer.txt"
    text = "" if answer is None else str(answer)
    if text and not text.endswith("\n"):
        text = text + "\n"

    fd, tmp_name = tempfile.mkstemp(
        prefix=".answer.",
        suffix=".tmp",
        dir=str(task_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CostJournal:
    """Append-only cost journal: task_id, reward, cost, model, status, ...

    Always keeps an in-memory log. If ``OUT`` is set and writable, also
    mirrors JSONL records there (file or directory).
    """

    def __init__(self, out_path: Optional[Path | str] = None) -> None:
        self.records: list[dict[str, Any]] = []
        self._fh: Optional[TextIO] = None
        self.out_path: Optional[Path] = None
        if out_path is not None:
            self._open_mirror(Path(out_path))

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> CostJournal:
        environ = env if env is not None else os.environ
        out = environ.get("OUT")
        if not out:
            return cls(None)
        return cls(out)

    def _open_mirror(self, out: Path) -> None:
        """Mirror JSONL to OUT if the path is writable.

        Competition scripts often ``mkdir -p "$OUT"`` (directory). A path
        ending in ``.jsonl`` / ``.json`` is treated as a file. Failures are
        silent — jury delivery is ``answer.txt`` in each task dir.
        """
        try:
            if out.suffix in {".jsonl", ".json"}:
                out.parent.mkdir(parents=True, exist_ok=True)
                path = out
            else:
                out.mkdir(parents=True, exist_ok=True)
                path = out / "cost_journal.jsonl" if out.is_dir() else out

            self._fh = path.open("a", encoding="utf-8")
            self.out_path = path
        except OSError as exc:
            print(f"WARN: cost journal not writable at {out}: {exc}", flush=True)
            self._fh = None
            self.out_path = None

    def record(
        self,
        *,
        task_id: str,
        reward: int,
        cost: int,
        model: Optional[str],
        status: Status | str,
        task_type: Optional[str] = None,
        answer_written: bool = False,
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "task_id": task_id,
            "reward": reward,
            "cost": int(cost),
            "model": model,
            "status": status.value if isinstance(status, Status) else str(status),
            "ts": _utc_now_iso(),
        }
        if task_type is not None:
            entry["task_type"] = task_type
        entry["answer_written"] = bool(answer_written)
        if error:
            entry["error"] = error
        if extra:
            entry.update(extra)

        self.records.append(entry)
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._fh.flush()
            except OSError:
                pass
        return entry

    def record_result(self, task: Task, result: Result, *, answer_written: bool) -> dict[str, Any]:
        return self.record(
            task_id=task.task_id,
            reward=task.reward,
            cost=result.cost,
            model=result.model,
            status=result.status,
            task_type=task.task_type.value,
            answer_written=answer_written,
            error=result.error,
            extra=dict(result.meta) if result.meta else None,
        )

    def total_cost(self) -> int:
        return sum(int(r.get("cost", 0)) for r in self.records)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def __enter__(self) -> CostJournal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
