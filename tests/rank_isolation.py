"""Keep unit tests from loading repo ``artifacts/models`` solvability weights."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager

from harness.runtime.rank import clear_cache


@contextmanager
def no_rank_models() -> Iterator[Path]:
    """Point RANK_MODELS at an empty dir so assess() uses p=1 / heuristic cost."""
    prev = os.environ.get("RANK_MODELS")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        os.environ["RANK_MODELS"] = str(path)
        clear_cache()
        try:
            yield path
        finally:
            clear_cache()
            if prev is None:
                os.environ.pop("RANK_MODELS", None)
            else:
                os.environ["RANK_MODELS"] = prev
