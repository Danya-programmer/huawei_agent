#!/usr/bin/env bash
# Build + run harness the jury way (local helper: loads .env, mounts TASKS/OUT).
# Canonical container flags match the contest регламент.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "missing .env — copy from .env.example and fill LLM_*" >&2
  exit 1
fi

# Keep CLI overrides (sudo TASKS=... ./scripts/run_docker.sh) — source .env must not clobber them.
_preset_tasks="${TASKS-}"
_preset_out="${OUT-}"
_preset_workers="${MAX_WORKERS-}"
_preset_model="${LLM_MODEL-}"
_preset_budget="${BUDGET-}"

# shellcheck disable=SC1091
set -a
source .env
set +a

if [[ -n "$_preset_tasks" ]]; then TASKS="$_preset_tasks"; fi
if [[ -n "$_preset_out" ]]; then OUT="$_preset_out"; fi
if [[ -n "$_preset_workers" ]]; then MAX_WORKERS="$_preset_workers"; fi
if [[ -n "$_preset_model" ]]; then LLM_MODEL="$_preset_model"; fi
if [[ -n "$_preset_budget" ]]; then BUDGET="$_preset_budget"; fi

# Strip accidental whitespace
LLM_BASE_URL="$(echo -n "${LLM_BASE_URL:-}" | xargs)"
LLM_API_KEY="$(echo -n "${LLM_API_KEY:-}" | xargs)"
LLM_MODEL="$(echo -n "${LLM_MODEL:-qwen3.8-27b-noreason}" | xargs)"

if [[ -z "${LLM_BASE_URL}" || -z "${LLM_API_KEY}" ]]; then
  echo "LLM_BASE_URL and LLM_API_KEY required in .env" >&2
  exit 1
fi

# Invite page is not an API base
if [[ "$LLM_BASE_URL" == *"hub.neuraldeep.ru"* ]]; then
  echo "LLM_BASE_URL looks like hub invite page. Use https://api.neuraldeep.ru/v1" >&2
  exit 1
fi

TASKS_HOST="${TASKS:-$ROOT/task-dir}"
# Resolve relative TASKS against repo root
if [[ "$TASKS_HOST" != /* ]]; then
  TASKS_HOST="$ROOT/${TASKS_HOST#./}"
fi

# Unpack official examples only when targeting task-dir.
if [[ "$TASKS_HOST" == "$ROOT/task-dir" ]]; then
  unzip -qo task-dir.zip
fi

if [[ ! -d "$TASKS_HOST" ]]; then
  echo "TASKS directory missing: $TASKS_HOST" >&2
  echo "For test-100: unzip -qo task_packs/test-100.zip -d task_packs" >&2
  exit 1
fi

n_tasks="$(find "$TASKS_HOST" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
mkdir -p "${OUT:-$ROOT/out}"
OUT_HOST="$(cd "${OUT:-$ROOT/out}" && pwd)"

# Docker Desktop/WSL often remaps container root ≠ host uid. World-writable
# mounts match the jury's `chmod 777 "$OUT"` and let the agent write answers.
chmod -R a+rwX "$OUT_HOST" 2>/dev/null || true
chmod -R a+rwX "$TASKS_HOST" 2>/dev/null || true
# Recreate journal so append isn't blocked by a host-owned 0644 file.
: >"$OUT_HOST/cost_journal.jsonl"
chmod a+rw "$OUT_HOST/cost_journal.jsonl" 2>/dev/null || true

echo "TASKS=$TASKS_HOST ($n_tasks task dirs)"
echo "OUT=$OUT_HOST"
echo "LLM_MODEL=$LLM_MODEL"
echo "BUDGET=${BUDGET:-<none>}"
echo "MAX_WORKERS=${MAX_WORKERS:-2}"

echo "building image hack:latest ..."
docker build -t hack:latest .
docker images --format '{{.Repository}}:{{.Tag}} -> {{.Size}}' | grep '^hack:latest'

echo "running (canonical flags) on $TASKS_HOST ..."
# Mirrors contest docker run: read-only rootfs, dropped caps, cpu/mem/pids limits.
# Local extras vs jury:
#   - OUT volume + LLM_MODEL/BUDGET/MAX_WORKERS
#   - tmpfs /app/logs: openjiuwen defaults to ./logs/ under cwd=/app; without
#     this, --read-only makes every task fail with FrameworkError 183001.
docker run --rm --name hack-run \
  -e LLM_BASE_URL="$LLM_BASE_URL" \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -e LLM_PROVIDER="${LLM_PROVIDER:-OpenAI}" \
  -e LLM_SSL_VERIFY="${LLM_SSL_VERIFY:-false}" \
  -e LLM_MODEL="$LLM_MODEL" \
  -e BUDGET="${BUDGET:-}" \
  -e MAX_WORKERS="${MAX_WORKERS:-2}" \
  -e OUT=/data/out \
  -v "$TASKS_HOST:/data/tasks:rw" \
  -v "$OUT_HOST:/data/out:rw" \
  --read-only \
  --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --tmpfs /app/logs:rw,size=256m \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  hack:latest \
  /app/run_agent.py --tasks-dir /data/tasks

echo "answers: $(find "$TASKS_HOST" -name answer.txt | wc -l | tr -d ' ') files"
find "$TASKS_HOST" -name answer.txt -printf '%P: ' -exec head -c 120 {} \; -exec printf '\n' \;
echo "journal: $OUT_HOST/cost_journal.jsonl"
