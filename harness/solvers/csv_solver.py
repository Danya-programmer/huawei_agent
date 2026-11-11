"""Deterministic sales.csv aggregation solver (no LLM)."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from harness.types import Result, Status, Task

_PRODUCT_RE = re.compile(
    r"units of product\s+'([^']+)'\s+were sold in months\s+(\d+)\s+through\s+(\d+)",
    re.IGNORECASE,
)
_REQUIRED_COLS = ("order_id", "region", "product", "month", "units", "price")


def parse_csv_query(instruction: str) -> Optional[tuple[str, int, int]]:
    m = _PRODUCT_RE.search(instruction)
    if not m:
        return None
    product = m.group(1)
    lo = int(m.group(2))
    hi = int(m.group(3))
    if lo > hi:
        lo, hi = hi, lo
    return product, lo, hi


def sum_units(csv_path: Path, product: str, month_lo: int, month_hi: int) -> int:
    total = 0
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("csv has no header")
        cols = {c.strip().lower() for c in reader.fieldnames}
        missing = [c for c in _REQUIRED_COLS if c not in cols]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        for row in reader:
            if row.get("product") != product:
                continue
            try:
                month = int(row["month"])
                units = int(row["units"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"bad row: {row}") from exc
            if month_lo <= month <= month_hi:
                total += units
    return total


class CsvSolver:
    def solve(self, task: Task) -> Result:
        query = parse_csv_query(task.instruction)
        if query is None:
            return Result(
                status=Status.UNSOLVABLE,
                cost=0,
                error="unrecognized data_analysis template",
                meta={"reason": "parse_failed", "solver": "csv"},
            )

        csv_path = task.work_dir / "sales.csv"
        if not csv_path.is_file():
            return Result(
                status=Status.UNSOLVABLE,
                cost=0,
                error="sales.csv missing",
                meta={"reason": "missing_file", "solver": "csv"},
            )

        product, lo, hi = query
        try:
            total = sum_units(csv_path, product, lo, hi)
        except ValueError as exc:
            return Result(
                status=Status.UNSOLVABLE,
                cost=0,
                error=str(exc),
                meta={"reason": "bad_csv", "solver": "csv"},
            )

        return Result(
            status=Status.SOLVED,
            answer=str(total),
            cost=0,
            model=None,
            meta={
                "solver": "csv",
                "method": "deterministic",
                "product": product,
                "months": [lo, hi],
            },
        )
