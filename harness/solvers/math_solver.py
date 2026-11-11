"""Deterministic warehouse-arithmetic solver (no LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from harness.types import Result, Status, Task


@dataclass(frozen=True)
class MathOp:
    kind: str  # start | split | double | ship | receive
    a: int = 0
    b: int = 0


_START_RE = re.compile(
    r"warehouse starts with\s+(\d+)\s+boxes",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(
    r"split equally into\s+(\d+)\s+warehouses.*?keeps one share\s*\(rounded down\)",
    re.IGNORECASE | re.DOTALL,
)
_DOUBLE_RE = re.compile(
    r"stock is doubled by merging with a sister warehouse",
    re.IGNORECASE,
)
_SHIP_RE = re.compile(
    r"ships\s+(\d+)\s+boxes per day for\s+(\d+)\s+days",
    re.IGNORECASE,
)
_RECEIVE_RE = re.compile(
    r"receives\s+(\d+)\s+deliver(?:y|ies)\s+of\s+(\d+)\s+boxes each",
    re.IGNORECASE,
)


def parse_math_ops(instruction: str) -> Optional[list[MathOp]]:
    """Parse known warehouse templates into an op list. None = unrecognized."""
    start = _START_RE.search(instruction)
    if not start:
        return None

    ops: list[MathOp] = [MathOp("start", int(start.group(1)))]
    # Walk the instruction left-to-right collecting ops by earliest match.
    pos = start.end()
    while pos < len(instruction):
        candidates: list[tuple[int, MathOp, int]] = []
        for regex, builder in (
            (_SPLIT_RE, lambda m: MathOp("split", int(m.group(1)))),
            (_DOUBLE_RE, lambda m: MathOp("double")),
            (_SHIP_RE, lambda m: MathOp("ship", int(m.group(1)), int(m.group(2)))),
            (_RECEIVE_RE, lambda m: MathOp("receive", int(m.group(1)), int(m.group(2)))),
        ):
            m = regex.search(instruction, pos)
            if m:
                candidates.append((m.start(), builder(m), m.end()))
        if not candidates:
            break
        candidates.sort(key=lambda t: t[0])
        _, op, end = candidates[0]
        ops.append(op)
        pos = end

    if len(ops) < 2:
        return None
    return ops


def eval_math_ops(ops: list[MathOp]) -> int:
    value = 0
    for op in ops:
        if op.kind == "start":
            value = op.a
        elif op.kind == "split":
            if op.a <= 0:
                raise ValueError("split divisor must be positive")
            value = value // op.a
        elif op.kind == "double":
            value = value * 2
        elif op.kind == "ship":
            value = value - op.a * op.b
        elif op.kind == "receive":
            value = value + op.a * op.b
        else:
            raise ValueError(f"unknown op {op.kind}")
    return value


def try_solve_math(instruction: str) -> Optional[int]:
    ops = parse_math_ops(instruction)
    if ops is None:
        return None
    try:
        return eval_math_ops(ops)
    except (ValueError, ZeroDivisionError):
        return None


class MathSolver:
    """Solve warehouse math templates without calling an LLM."""

    def solve(self, task: Task) -> Result:
        answer = try_solve_math(task.instruction)
        if answer is None:
            return Result(
                status=Status.UNSOLVABLE,
                cost=0,
                error="unrecognized math template",
                meta={"reason": "parse_failed", "solver": "math"},
            )
        return Result(
            status=Status.SOLVED,
            answer=str(answer),
            cost=0,
            model=None,
            meta={"solver": "math", "method": "deterministic"},
        )
