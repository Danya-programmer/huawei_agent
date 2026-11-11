#!/usr/bin/env bash
# Build + run harness via Docker, loading secrets from .env
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "missing .env — copy from .env.example and fill LLM_*" >&2
  exit 1
fi

# shellcheck disable=SC1091
set -a
source .env
set +a

# Strip accidental whitespace
LLM_BASE_URL="$(echo -n "${LLM_BASE_URL:-}" | xargs)"
LLM_API_KEY="$(echo -n "${LLM_API_KEY:-}" | xargs)"

if [[ -z "${LLM_BASE_URL}" || -z "${LLM_API_KEY}" ]]; then
  echo "LLM_BASE_URL and LLM_API_KEY required in .env" >&2
  exit 1
fi

# Invite page is not an API base
if [[ "$LLM_BASE_URL" == *"hub.neuraldeep.ru"* ]]; then
  echo "LLM_BASE_URL looks like hub invite page. Use https://api.neuraldeep.ru/v1" >&2
  exit 1
fi

unzip -qo task-dir.zip
TASKS_HOST="${TASKS:-$ROOT/task-dir}"
# Resolve relative TASKS against repo root
if [[ "$TASKS_HOST" != /* ]]; then
  TASKS_HOST="$ROOT/${TASKS_HOST#./}"
fi
mkdir -p "${OUT:-$ROOT/out}"
OUT_HOST="$(cd "${OUT:-$ROOT/out}" && pwd)"

echo "building image hack:latest ..."
docker build -t hack:latest .
docker images --format '{{.Repository}}:{{.Tag}} -> {{.Size}}' | grep '^hack:latest'

echo "running on $TASKS_HOST ..."
docker run --rm --name hack-run \
  --env-file "$ROOT/.env" \
  -e LLM_BASE_URL="$LLM_BASE_URL" \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -e OUT=/data/out \
  -v "$TASKS_HOST:/data/tasks:rw" \
  -v "$OUT_HOST:/data/out:rw" \
  --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --workdir /app --entrypoint python \
  hack:latest \
  /app/run_agent.py --tasks-dir /data/tasks

echo "answers:"
find "$TASKS_HOST" -name answer.txt -printf '%P: ' -exec cat {} \;
echo "journal: $OUT_HOST/cost_journal.jsonl"
