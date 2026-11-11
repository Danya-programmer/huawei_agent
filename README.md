# Budget-aware openJiuwen harness

Харнесс для openJiuwen Challenge: максимум баллов на ~100 задачах при общем бюджете токенов.

## Что делает

1. Читает все `task.json` из `--tasks-dir` / `TASKS`.
2. Сортирует задачи по `reward / expected_cost`.
3. Решает до 5 задач параллельно.
4. Пишет `answer.txt` в каталог задачи сразу после успеха.
5. Ведёт журнал стоимости (`OUT/cost_journal.jsonl`, если `OUT` задан).

Типы задач:

| type | стратегия |
|------|-----------|
| `math` | детерминированный парсер warehouse-арифметики (0 токенов) |
| `data_analysis` | сумма `units` по `sales.csv` (0 токенов) |
| `search` | 1–2 hop lookup по `kb/*.md` (0 токенов) |
| `coding` | эвристический фикс `mylib.py` по тестам; опционально ReAct (`HARNESS_USE_REACT=1`) |

Невыполнимые (нет файла/сущности/шаблона) помечаются `UNSOLVABLE` и не жгут бюджет.

## Учёт токенов

```
cost = completion_tokens + reasoning_tokens + ceil(prompt_tokens / 10)
```

Модели: `qwen3.8-27b`, `qwen3.8-27b-noreason`, `qwen3.6-35b-a3b`, `qwen3.6-35b-a3b-noreason`.
По умолчанию для ReAct — `qwen3.6-35b-a3b-noreason`.

Переменные:

- `LLM_BASE_URL`, `LLM_API_KEY` — прокси организаторов
- `TASKS` — каталог задач (альтернатива `--tasks-dir`)
- `OUT` — опциональный журнал
- `BUDGET` — мягкий потолок в output-token units
- `MAX_WORKERS` — параллельность (≤5)
- `HARNESS_USE_REACT=1` — включить LLM fallback для coding

Ключи в репозиторий не кладутся.

## Локальный запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
unzip -qo task-dir.zip
python run_agent.py --tasks-dir task-dir
python -m unittest discover -s tests -v
python -m harness.generators --out train-tasks --n 40
```

Калибровка моделей (нужен ключ):

```bash
export LLM_BASE_URL=... LLM_API_KEY=...
python scripts/calibrate.py
```

## Docker

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

Версии зафиксированы: `python:3.11.10-slim-bookworm`, `openjiuwen==0.1.19` без extras.
При сборке всё нужное попадает в образ; runtime-загрузки запрещены.

Проверено локально (`sudo ./scripts/run_docker.sh`):

- образ `hack:latest` → **1.41GB** (лимит 4GB)
- 8/8 примеров из `task-dir.zip`, reward=20, cost=0 (детерминированные/heuristic решатели)
- калибровка API (`scripts/calibrate.py`): все 4 модели отвечают; `*-noreason` заметно дешевле (cost≈4 vs 66–75 на коротком ping)

Быстрый Docker-запуск с `.env`:

```bash
sudo ./scripts/run_docker.sh
```

## Структура

```
run_agent.py
harness/
  types.py cost.py tasks.py io.py estimate.py scheduler.py
  solvers/  math_solver.py csv_solver.py search_solver.py coding.py
  react_agent.py generators.py
scripts/calibrate.py
Dockerfile
```

## Материалы

См. `полезные статьи.md` (BAGEN, TRIAGE, FrugalGPT, SWE-agent, openJiuwen docs).
