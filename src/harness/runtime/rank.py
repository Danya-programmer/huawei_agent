"""Solvability gate and value-density priority. Weights are optional."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from harness.runtime.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract,
    heuristic_cost,
    structural_skip_reason,
    vector,
)

_CACHE: dict[str, "_Bundle"] = {}


@dataclass(frozen=True)
class RankEstimate:
    """One task scored before any LLM call."""

    p_solvable: float
    cost_hat: int
    priority: float
    skip: bool
    fail_reason: Optional[str]


@dataclass
class _Bundle:
    clf: Any
    cost: Any
    tau: float
    feature_names: tuple[str, ...]


class ConstantProba:
    """Picklable stand-in so tests can force a probability without sklearn."""

    def __init__(self, p_solvable: float) -> None:
        self.p_solvable = float(p_solvable)
        self.classes_ = [0, 1]

    def predict_proba(self, rows: Any) -> list[list[float]]:
        p = self.p_solvable
        return [[1.0 - p, p] for _ in rows]


def models_dir() -> Path:
    """Prefer RANK_MODELS, then artifacts/models, then /app/models (Docker)."""
    raw = os.environ.get("RANK_MODELS")
    if raw:
        return Path(raw)
    # src/harness/runtime/rank.py → repo root is parents[3] (/app in the image)
    root = Path(__file__).resolve().parents[3]
    artifacts = root / "artifacts" / "models"
    if (artifacts / "solvability.joblib").is_file() or (artifacts / "tau.json").is_file():
        return artifacts
    # Dockerfile copies artifacts/models → /app/models
    return root / "models"


def clear_cache() -> None:
    _CACHE.clear()


def assess(task: Any, *, models: Optional[Path] = None) -> RankEstimate:
    """p=1 for math. Skip on a structural fail or on p < tau for the other types."""
    feats = extract(task)
    task_type = _type_name(task)
    hard = structural_skip_reason(task, feats)
    bundle = _bundle(models if models is not None else models_dir())
    cost = _predict_cost(bundle, task, feats)

    if task_type == "math":
        p = 1.0
        skip = False
        reason = None
    elif hard:
        p = 0.0
        skip = True
        reason = hard
    else:
        p = _predict_p(bundle, task)
        if p < bundle.tau:
            skip = True
            reason = "low_p"
        else:
            skip = False
            reason = None

    priority = 0.0 if skip else (p * _reward(task) / max(cost, 1))
    return RankEstimate(
        p_solvable=p,
        cost_hat=0 if skip else cost,
        priority=priority,
        skip=skip,
        fail_reason=reason,
    )


def _type_name(task: Any) -> str:
    raw = getattr(task, "task_type", "")
    return raw.value if hasattr(raw, "value") else str(raw)


def _reward(task: Any) -> float:
    try:
        return float(task.reward)
    except (TypeError, ValueError):
        return 0.0


def _bundle(path: Path) -> _Bundle:
    key = str(path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    clf_blob = _read_blob(path / "solvability.joblib")
    cost_blob = _read_blob(path / "cost.joblib")
    names = FEATURE_NAMES
    clf = None
    cost = None
    if isinstance(clf_blob, dict) and clf_blob.get("schema_version") == FEATURE_SCHEMA_VERSION:
        saved = tuple(clf_blob.get("feature_names") or ())
        if saved == FEATURE_NAMES:
            names = saved
            clf = clf_blob.get("model")
    if isinstance(cost_blob, dict) and cost_blob.get("schema_version") == FEATURE_SCHEMA_VERSION:
        saved = tuple(cost_blob.get("feature_names") or ())
        if saved == FEATURE_NAMES:
            cost = cost_blob.get("model")
    bundle = _Bundle(clf=clf, cost=cost, tau=_read_tau(path), feature_names=names)
    _CACHE[key] = bundle
    return bundle


def _read_tau(path: Path) -> float:
    """Missing tau means never skip on probability. Structural skips still apply."""
    file = path / "tau.json"
    if not file.is_file():
        return 0.0
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
        return float(raw.get("tau", 0.0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0


def _read_blob(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None


def _row(task: Any) -> list[float]:
    return vector(task)


def _predict_p(bundle: _Bundle, task: Any) -> float:
    model = bundle.clf
    if model is None:
        return 1.0
    try:
        proba = model.predict_proba([_row(task)])[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else -1
        return float(min(1.0, max(0.0, proba[idx])))
    except Exception:
        return 1.0


def _predict_cost(bundle: _Bundle, task: Any, feats: dict[str, float]) -> int:
    model = bundle.cost
    if model is None:
        return heuristic_cost(task, feats)
    try:
        pred = float(model.predict([_row(task)])[0])
        if pred != pred:  # NaN
            raise ValueError("nan cost")
        return max(1, int(round(pred)))
    except Exception:
        return heuristic_cost(task, feats)
