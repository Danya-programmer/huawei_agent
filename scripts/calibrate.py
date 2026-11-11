#!/usr/bin/env python3
"""Live calibration against the contest LLM proxy.

Requires env:
  LLM_BASE_URL, LLM_API_KEY
Optional:
  TASKS  (default: extracted task-dir)
  OUT    (directory for trajectories JSONL)

Usage:
  python scripts/calibrate.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.cost import compute_call_cost
from harness.types import ModelName

MODELS = [m.value for m in ModelName]


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
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


def chat(model: str, prompt: str) -> dict:
    base = os.environ["LLM_BASE_URL"].strip().rstrip("/")
    key = os.environ["LLM_API_KEY"].strip()

    url = f"{base}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reasoning_tokens = int(
        details.get("reasoning_tokens")
        or usage.get("reasoning_tokens")
        or 0
    )
    cost = compute_call_cost(prompt_tokens, completion_tokens, reasoning_tokens)
    return {
        "model": model,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost": cost,
        "content": ((payload.get("choices") or [{}])[0].get("message") or {}).get(
            "content"
        ),
    }


def main() -> int:
    _load_dotenv()
    if not os.environ.get("LLM_API_KEY") or not os.environ.get("LLM_BASE_URL"):
        print("LLM_API_KEY and LLM_BASE_URL required", file=sys.stderr)
        return 2

    out_dir = Path(os.environ.get("OUT") or (ROOT / "trajectories"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibrate_models.jsonl"

    rows = []
    with path.open("w", encoding="utf-8") as fh:
        for model in MODELS:
            try:
                row = chat(model, "Reply with exactly: pong")
            except urllib.error.URLError as exc:
                row = {"model": model, "error": str(exc)}
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row)

    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
