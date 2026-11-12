# Budget-aware openJiuwen harness

Харнесс для openJiuwen Challenge: максимум баллов на ~100 задачах при общем бюджете токенов.

## Что делает

1. Читает все `task.json` из `--tasks-dir` / `TASKS`.
2. Triage: оценка стоимости; coding отдельно от коротких ReAct-задач.
3. Решает до 5 задач параллельно (coding/ReAct concurrency ≤2).
4. Каждую задачу решает **только ReAct-агент** (openJiuwen + инструменты fs/shell/code).
5. Пишет `answer.txt` в каталог задачи сразу после успеха.
6. Ведёт журнал стоимости (`OUT/cost_journal.jsonl`, если `OUT` задан).

| type | стратегия |
|------|-----------|
| `math` / `data_analysis` / `search` | ReAct (`run_task_react`) |
| `coding` | ReAct (`run_coding_react`) + проверка `test_mylib.py` |

## Учёт токенов

```
cost = completion_tokens + reasoning_tokens + ceil(prompt_tokens / 10)
```

Модели: `qwen3.8-27b`, `qwen3.8-27b-noreason`, `qwen3.6-35b-a3b`, `qwen3.6-35b-a3b-noreason`.
По умолчанию — `qwen3.6-35b-a3b-noreason`.

Переменные:

- `LLM_BASE_URL`, `LLM_API_KEY` — прокси организаторов (обязательны)
- `TASKS` — каталог задач (альтернатива `--tasks-dir`)
- `OUT` — опциональный журнал
- `BUDGET` — мягкий потолок в output-token units
- `MAX_WORKERS` — параллельность (≤5)

Ключи в репозиторий не кладутся.

## Локальный запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
unzip -qo task-dir.zip
export LLM_BASE_URL=... LLM_API_KEY=...
python run_agent.py --tasks-dir task-dir
python -m unittest discover -s tests -q
```

Калибровка моделей:

```bash
python scripts/calibrate.py
```

## Docker

```bash
sudo ./scripts/run_docker.sh
```

Или вручную:

```bash
docker build -t hack:latest .
mkdir -p "$OUT" && chmod 777 "$OUT"
docker run --rm \
  -e LLM_BASE_URL -e LLM_API_KEY \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cpus 8 --memory 16g \
  --workdir /app --entrypoint python \
  hack:latest \
  /app/run_agent.py --tasks-dir /data/tasks
```

Версии: `python:3.11.10-slim-bookworm`, `openjiuwen==0.1.19`. Образ ранее собирался ~1.41GB (лимит 4GB).

## Структура

```
run_agent.py
harness/
  types.py cost.py usage.py tasks.py io.py estimate.py triage.py scheduler.py
  react_agent.py
  solvers/  react_solver.py
scripts/calibrate.py fit_estimates.py run_docker.sh
Dockerfile
```

## Материалы

См. `полезные статьи.md` (BAGEN, TRIAGE, FrugalGPT, SWE-agent, openJiuwen docs).
