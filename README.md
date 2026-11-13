# openJiuwen Challenge — budget-aware ReAct harness

Решение для финала: максимум `reward` на ~100 задачах при мягком бюджете `BUDGET≈100000` output-token units.

**Стек:** openJiuwen ReAct + proxy LLM (`qwen3.*`) + классический solvability-классификатор (HistGradientBoosting). Локальных нейросетей нет.

Точка входа жюри:

```text
python /app/run_agent.py --tasks-dir /data/tasks
```

Ответ: `answer.txt` **рядом с** `task.json` (не в `workspace/`). За заведомо нерешаемые задачи файл **не создаётся**.

---

## Содержание

1. [Для проверяющих (быстрый старт)](#для-проверяющих-быстрый-старт)
2. [Пайплайн целиком](#пайплайн-целиком)
3. [Как устроен агент](#как-устроен-агент)
4. [Отсечение нерешаемых и ранжирование](#отсечение-нерешаемых-и-ранжирование)
5. [Учёт токенов и бюджет](#учёт-токенов-и-бюджет)
6. [Переменные окружения](#переменные-окружения)
7. [Структура репозитория](#структура-репозитория)
8. [Локальная разработка](#локальная-разработка)
9. [Планы улучшений](#планы-улучшений)
10. [Материалы](#материалы)

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

## Пайплайн целиком

```text
discover tasks  (каталоги с task.json)
      │
      ▼
 triage / assess  ← без LLM
      │  · structural hard-skip (битые файлы, cue-фразы unsolvable math, …)
      │  · solvability clf + tau  (coding / data_analysis / search)
      │  · math: p = 1, кроме cue-skip
      │  · priority = p × reward / cost_hat
      ▼
 одна очередь по priority ↓  (затем reward, task_id)
      │
      ▼
 ReAct (openJiuwen)  под остаток BUDGET
      │  tools: read/write file, execute_code, execute_cmd
      │  sandbox = workspace задачи
      ▼
 atomic write  answer.txt  рядом с task.json
```

| `task_type` | как решается |
|-------------|--------------|
| `math` | ReAct (bash/python); hard-skip по cue-фразам underspec / contradictory |
| `data_analysis` | ReAct по CSV/таблицам в `workspace/` |
| `search` | ReAct по `kb/*.md` |
| `coding` | ReAct правит только `mylib.py`; жюри гоняет `test_mylib.py` |

Шаблонных автосолверов нет: **все четыре типа** идут через один `ReactSolver`.

---

## Как устроен агент

Код: `src/harness/agents/` (`ReactSolver` → `run_task_react` / `run_coding_react` → openJiuwen `ReActAgent`).

### Цикл ReAct

1. На задачу создаётся свежий `ReActAgent` с `AgentCard` и `SysOperationCard` в режиме `LOCAL`.
2. Sandbox ограничен каталогом задачи (`sandbox_root` + `restrict_to_sandbox=True`); CWD — `workspace/` (или корень задачи).
3. К агенту подвешиваются sysop-tools:
   - `read_file` / `write_file`
   - `execute_code` (Python)
   - `execute_cmd` (shell)
4. Модель с proxy (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`) крутит до **`max_iterations=8`** шагов «мысль → tool → наблюдение».
5. После прогона харнесс читает `answer.txt`, санитизирует и атомарно кладёт рядом с `task.json`. Для coding дополнительно проверяется, что локальные тесты зелёные, перед фиксацией `DONE`.

Между задачами на том же worker-потоке: **persistent event loop** (не `asyncio.run` на каждую задачу) + `OpenAIModelClient.aclose()`, иначе openjiuwen держит httpx-клиент на закрытом loop / keep-alive и следующий вызов зависает.

### Два system prompt

**General** (`math` / `data_analysis` / `search`): исследовать файлы, посчитать ответ, записать **только финальное значение** в `answer.txt` через redirect (`echo … > answer.txt`). Если задача нерешаема при данных файлах — **не создавать** `answer.txt`.

**Coding**: править только `mylib.py`, не трогать тесты; цель — `python3 test_mylib.py` → OK; затем ровно `DONE` в `answer.txt`.

### Планировщик вокруг агента

`SimpleScheduler` сначала закрывает triage-skips статусом `UNSOLVABLE` (cost 0, без LLM), затем гоняет runnable-очередь. Перед каждым вызовом агента: влезает ли `cost_hat` в остаток `BUDGET`. Параллельность по умолчанию **1** ReAct-сессия (`CODING_REACT_CONCURRENCY=1`), чтобы не ловить 429 у proxy.

---

## Отсечение нерешаемых и ранжирование

Цель: **не тратить бюджет** на заведомо провальные задачи и раньше брать задачи с лучшим ожидаемым `reward` на токен.

### 1. Structural hard-skip (правила, без модели)

Срабатывает до классификатора. Примеры:

| тип | условие |
|-----|---------|
| `data_analysis` | нет CSV / указанный в инструкции файл отсутствует |
| `search` | пустой `kb/` |
| `coding` | нет `mylib.py`, пустой тест, импорт не `mylib` |
| `math` | cue-фразы underspec / contradiction (`underdetermined`, `never numerically specified`, `write NOTHING`, `must be both … and … at once`, …) |

Причина уходит в journal как `fail_reason` (например `math_unsolvable_cue`, `search_kb_empty`).

### 2. Solvability-классификатор

Для `coding` / `data_analysis` / `search` (не для обычного math):

- из задачи считаются **32 tabular-фичи** (`src/harness/runtime/features.py`): тип, длина инструкции, признаки таблиц/kb/кода, флаги «битых» артефактов и т.д.;
- `HistGradientBoostingClassifier` из `artifacts/models/solvability.joblib` (schema v1) выдаёт `p = P(solvable)`;
- если `p < tau` (`artifacts/models/tau.json`, сейчас ≈ **0.50**) → skip с причиной `low_p`, LLM не вызывается.

Классификатор обучен на своих packs (`train-3000` / abstention), ноутбук: `notebooks/train_solvability.ipynb`. В образе только готовые веса — дообучения на оценке нет.

Math намеренно **не** гоняется через clf: `p = 1`, кроме structural cue-skip (иначе легко ложные пропуски коротких арифметических задач).

### 3. Value-density очередь

Для каждой runnable-задачи:

```text
priority = p_solvable × reward / max(cost_hat, 1)
```

Сортировка: выше `priority` → выше `reward` → `task_id`.

`cost_hat` сейчас — **эвристика** (~800 short ReAct, ~2500 coding + надбавки за длину/размер данных), если нет `cost.joblib`. Чем выше баллы и ниже ожидаемая цена при высокой `p`, тем раньше задача попадёт к агенту, пока бюджет не кончился.

Переопределение каталога весов: `RANK_MODELS=/path/to/dir` (нужны `solvability.joblib` + `tau.json`).

---

## Учёт токенов и бюджет

```text
cost = completion_tokens + reasoning_tokens + ceil(prompt_tokens / 10)
```

`BUDGET` — мягкий потолок в этих единицах: харнесс сам ведёт ledger; при исчерпании новые задачи получают `budget` без вызова LLM. Оставшийся бюджет proxy **не сообщает**.

Локальный журнал: `OUT/cost_journal.jsonl` (task_id, status, cost, model, …).

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
| `OUT` | каталог / `cost_journal.jsonl` |
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
notebooks/                # обучение clf / cost (не в образе)
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

## Планы улучшений

Сейчас очередь опирается на **эвристический** `cost_hat`. В улучшениях хотим **умный cost-ранкер** (классическая модель, без локальных нейросетей):

- те же tabular-признаки задачи (`features.py`);
- таргет — **фактический расход токенов LLM** с прошлых прогонов (`cost_journal.jsonl` / калибровочные трассы);
- подставить `cost.joblib` в `assess`, чтобы `priority = p × reward / cost_pred` лучше отражал реальную «цену» ReAct на math vs coding vs длинный search;
- заготовка обучения: `notebooks/train_cost.ipynb`.

Идея: сначала отсечь нерешаемое (structural + solvability), затем точнее оценивать, **сколько токенов съест** оставшаяся задача, и тратить бюджет на самые выгодные reward/token.

---

## Материалы

- [`docs/Регламент соревнования.md`](docs/Регламент%20соревнования.md)
- [`docs/полезные статьи.md`](docs/полезные%20статьи.md)
- [`AGENTS.md`](AGENTS.md) — FAQ с установочной встречи
- [`notebooks/README.md`](notebooks/README.md) — обучение solvability / cost
