# openJiuwen Challenge — budget-aware ReAct harness

Решение для финала: максимум `reward` на ~100 задачах при мягком бюджете `BUDGET≈100000` output-token units.

**Стек:** openJiuwen ReAct + proxy LLM (`qwen3.*`) + классический solvability-классификатор (HistGradientBoosting). Локальных нейросетей нет.

Точка входа жюри:

```text
python /app/run_agent.py --tasks-dir /data/tasks
```

Ответ: `answer.txt` **рядом с** `task.json` (не в `workspace/`). За заведомо нерешаемые задачи файл **не создаётся**.

---

## Для проверяющих (быстрый старт)

### Сборка и запуск (как у жюри)

```bash
# В окружении должны быть LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, BUDGET
docker build -t hack:latest .

docker run --rm --name "run-${TEAM}" \
  -e LLM_BASE_URL -e LLM_API_KEY -e LLM_MODEL \
  -e BUDGET="${BUDGET:-100000}" \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --tmpfs /app/logs:rw,size=256m \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  hack:latest \
  /app/run_agent.py --tasks-dir /data/tasks
```

Образ: `python:3.13.18-slim-bookworm`, `openjiuwen==0.1.20.post3`, `scikit-learn==1.9.1`.  
Веса ранжирования копируются в образ: `artifacts/models/` → `/app/models/`. Runtime-загрузок нет.

### Локальная проверка (команда)

```bash
cp .env.example .env   # LLM_* + BUDGET=100000 + TASKS=./task_packs/test-100
unzip -qo task_packs/test-100.zip -d task_packs
find task_packs/test-100 -name answer.txt -delete
sudo make docker-test BUDGET=100000
# отчёт: out/eval_report.md
```

Юнит-тесты (без LLM):

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

---

## Как устроено решение

```text
discover tasks
    → triage / assess (без LLM)
         · structural skip (битые файлы, cue-фразы unsolvable math, …)
         · solvability clf + tau  (coding / data_analysis / search)
         · math: p=1, кроме cue-skip
         · priority = p × reward / cost_hat
    → одна очередь по priority (потом reward, task_id)
    → ReAct (openJiuwen) под BUDGET
    → atomic write answer.txt рядом с task.json
```

| `task_type` | решение |
|-------------|---------|
| `math` | ReAct; hard-skip по cue-фразам underspec / contradictory |
| `data_analysis` | ReAct по таблицам в `workspace/` |
| `search` | ReAct по `kb/*.md` |
| `coding` | ReAct правит `mylib.py`; жюри проверяет `test_mylib.py` |

### Ранжирование и бюджет

- `priority = p_solvable × reward / cost_hat`
- `cost_hat` — эвристика (~800 short / ~2500 coding), если нет `cost.joblib`
- Перед запуском: если estimate не влезает в остаток `BUDGET` → статус `budget`, без LLM
- Параллельность по умолчанию **1** ReAct-сессия (`CODING_REACT_CONCURRENCY=1`), чтобы не ловить 429 у proxy; `MAX_WORKERS` — потолок (обычно 2)

### Solvability-модель

| файл | роль |
|------|------|
| `artifacts/models/solvability.joblib` | HistGradientBoosting, schema v1, 32 фичи |
| `artifacts/models/tau.json` | порог skip: `p < tau` |

Переопределение каталога: `RANK_MODELS=/path/to/dir` (ожидает `solvability.joblib` + `tau.json`).

Обучение (вне образа, Kaggle): `notebooks/train_solvability.ipynb`.

### Учёт токенов

```text
cost = completion_tokens + reasoning_tokens + ceil(prompt_tokens / 10)
```

Журнал (локально): `OUT/cost_journal.jsonl`.

---

## Переменные окружения

| переменная | назначение |
|------------|------------|
| `LLM_BASE_URL` | OpenAI-compatible proxy (**обязательно**) |
| `LLM_API_KEY` | ключ proxy (**обязательно**) |
| `LLM_MODEL` | `qwen3.8-27b-noreason` (default) / `qwen3.8-27b` / `qwen3.6-35b-a3b[-noreason]` |
| `LLM_PROVIDER` | `OpenAI` |
| `LLM_SSL_VERIFY` | `false` / `true` |
| `BUDGET` | мягкий потолок output-token units |
| `TASKS` | каталог задач, если нет `--tasks-dir` |
| `OUT` | каталог/`cost_journal.jsonl` |
| `MAX_WORKERS` | 1…5 (default **2**) |
| `RANK_MODELS` | опционально: каталог с весами |

Ключи в git не кладутся (`.env` в `.gitignore`). Пример: `.env.example`.

---

## Структура репозитория

```text
run_agent.py              # entrypoint жюри
src/harness/
  domain/                 # Task, Status, answer sanitize
  infra/                  # journal, cost, HTTP retry
  runtime/                # features, rank, triage, scheduler
  agents/                 # openJiuwen ReAct solvers
artifacts/models/         # solvability.joblib + tau.json  →  /app/models
scripts/                  # run_docker.sh, score_pack.py, packs
notebooks/                # обучение clf (не в образе)
tests/
task_packs/*.zip          # test-100 / train-3000 / abstention (SOURCES.md)
Dockerfile Makefile requirements.txt
```

`PYTHONPATH=src` (как в Makefile/Docker) или `pip install -e .`.

---

## Локальная разработка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
unzip -qo task_packs/test-100.zip -d task_packs
PYTHONPATH=src python run_agent.py --tasks-dir task_packs/test-100
```

Оценка coding как жюри (`test_mylib.py` + сверка ответов):

```bash
make score-test
```

Пересборка своих packs (нужен HF): см. [`task_packs/README.md`](task_packs/README.md).

---

## Материалы

- [`docs/Регламент соревнования.md`](docs/Регламент%20соревнования.md)
- [`docs/полезные статьи.md`](docs/полезные%20статьи.md)
- [`AGENTS.md`](AGENTS.md) — FAQ с установочной встречи
