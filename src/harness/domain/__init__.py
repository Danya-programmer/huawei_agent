"""Task contract: types, loading, answer sanitization."""

from harness.domain.answer import looks_like_garbage, sanitize_answer
from harness.domain.tasks import discover_tasks, load_task, resolve_tasks_dir
from harness.domain.types import (
    DEFAULT_MODEL,
    ModelName,
    Result,
    Status,
    Task,
    TaskType,
    TokenUsage,
    resolve_model,
)

__all__ = [
    "DEFAULT_MODEL",
    "ModelName",
    "Result",
    "Status",
    "Task",
    "TaskType",
    "TokenUsage",
    "discover_tasks",
    "load_task",
    "looks_like_garbage",
    "resolve_model",
    "resolve_tasks_dir",
    "sanitize_answer",
]
