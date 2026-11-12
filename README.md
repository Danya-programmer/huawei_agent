# Budget-aware openJiuwen harness

Харнесс для openJiuwen Challenge: набрать максимум баллов на наборе из ~100 задач при общем бюджете токенов.

Все задачи решаются **только через ReAct** (openJiuwen + инструменты чтения/записи файлов, shell и Python). Шаблонных автосолверов нет.

## Пайплайн

1. Читает все каталоги с `task.json` из `--tasks-dir` / `TASKS`.
2. Triage оценивает ожидаемую стоимость (без LLM): короткие ReAct vs coding.
3. Планировщик сначала гоняет math / data_analysis / search, затем coding.
4. Параллельность: до 5 сессий; coding/ReAct ограничен сильнее (≤2).
5. Успех → сразу `answer.txt` в каталоге задачи.
6. Журнал стоимости: `OUT/cost_journal.jsonl` (если задан `OUT`).

| `task_type` | как решается |
|-------------|--------------|
| `math` | ReAct → ответ в `answer.txt` |
| `data_analysis` | ReAct по CSV/таблицам в workspace |
| `search` | ReAct по `kb/*.md` |
| `coding` | ReAct правит `mylib.py`, затем проверка `test_mylib.py` |

Точка входа жюри: `python /app/run_agent.py --tasks-dir /data/tasks`.

## Учёт токенов

```
cost = completion_tokens + reasoning_tokens + ceil(prompt_tokens / 10)
```

Доступные модели прокси:

- `qwen3.8-27b`
- `qwen3.8-27b-noreason`
- `qwen3.6-35b-a3b`
- `qwen3.6-35b-a3b-noreason` ← по умолчанию

## Переменные окружения

| переменная | назначение |
|------------|------------|
| `LLM_BASE_URL` | OpenAI-compatible proxy (обязательно) |
| `LLM_API_KEY` | ключ proxy (обязательно) |
| `LLM_PROVIDER` | по умолчанию `OpenAI` |
| `LLM_SSL_VERIFY` | `false` / `true` |
| `TASKS` | каталог задач (если нет `--tasks-dir`) |
| `OUT` | куда писать `cost_journal.jsonl` |
| `BUDGET` | мягкий потолок в output-token units |
| `MAX_WORKERS` | параллельность 1…5 |

Ключи в git не кладутся. Пример: `.env.example` → `.env`.

Для NeuralDeep: `LLM_BASE_URL=https://api.neuraldeep.ru/v1` (не invite-страница hub).

## Локальный запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполни LLM_*
unzip -qo task-dir.zip

python run_agent.py --tasks-dir task-dir
python -m unittest discover -s tests -q
```

Калибровка четырёх моделей:

```bash
python scripts/calibrate.py
```

Подгонка cost-band по журналу:

```bash
python scripts/fit_estimates.py --journal out/cost_journal.jsonl
```

## Docker

Быстрый путь (читает `.env`, собирает образ, гоняет `TASKS`):

```bash
sudo ./scripts/run_docker.sh
```

Вручную (как у жюри):

```bash
docker build -t hack:latest .
mkdir -p "$OUT" && chmod 777 "$OUT"
docker run --rm \
  -e LLM_BASE_URL -e LLM_API_KEY \
  -v "$TASKS:/data/tasks:rw" \
  -v "$OUT:/data/out:rw" -e OUT=/data/out \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cpus 8 --memory 16g \
  --workdir /app --entrypoint python \
  hack:latest \
  /app/run_agent.py --tasks-dir /data/tasks
```

База: `python:3.11.10-slim-bookworm`, зависимость: `openjiuwen==0.1.19`.  
Образ собирается без extras; runtime-загрузки запрещены. Лимит размера образа — 4GB.

## Наборы задач

Примеры организаторов: `task-dir.zip`.

Свои packs из публичных датасетов (формат как у конкурса):

| файл | содержимое |
|------|----------|
| [`task_packs/SOURCES.md`](task_packs/SOURCES.md) | источники и лицензии |
| `task_packs/test-100.zip` | 100 задач (25×4 типа, ~20% unsolvable) |
| `task_packs/train-3000.zip` | 3000 задач, те же квоты |

```bash
unzip -qo task_packs/test-100.zip -d task_packs
python run_agent.py --tasks-dir task_packs/test-100
```

Пересборка packs (нужен интернет к HuggingFace + PyPI):

```bash
pip install -r requirements-datasets.txt
python scripts/build_external_datasets.py --out task_packs --test-n 100 --train-n 3000 --seed 42
```

Подробнее: [`task_packs/README.md`](task_packs/README.md).

## Структура репозитория

```
run_agent.py                 # entrypoint
harness/
  types.py cost.py usage.py
  tasks.py io.py
  estimate.py triage.py scheduler.py
  react_agent.py             # openJiuwen ReAct + chat fallback
  solvers/react_solver.py    # единый солвер на все типы
scripts/
  run_docker.sh calibrate.py fit_estimates.py
  build_external_datasets.py
task_packs/                  # zip + SOURCES.md
tests/
Dockerfile requirements.txt .env.example
```

## Материалы

См. [`полезные статьи.md`](полезные%20статьи.md) и [`Регламент соревнования.md`](Регламент%20соревнования.md).
