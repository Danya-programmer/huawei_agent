# Kaggle notebooks

Обучение не входит в образ конкурса. На Kaggle запускаются ноутбуки; обратно в репозиторий кладутся только веса и `tau`.

## Файлы

- `train_solvability.ipynb` — классификатор решаемости → `solvability.joblib`, `tau.json`
- `train_cost.ipynb` — модель стоимости / симуляция бюджета → `cost.joblib` (если есть journal)

## Самодостаточность

Нужен только датасет с паками `train-3000`, `abstention-pack`, `test-100` (Kaggle может вложить папку дважды — ноутбук сам найдёт `manifest.jsonl`).

- `features.py` в датасет класть не нужно: копия вшита в ячейку.
- `scikit-learn==1.9.1` ставится первой ячейкой.
- LLM и openJiuwen не используются.

Порядок: сначала solvability, потом cost (cost может подхватить `solvability.joblib` / `tau.json` из Output первого ноутбука).

## Куда положить веса в репозитории

Только сюда:

```
artifacts/models/solvability.joblib
artifacts/models/tau.json
artifacts/models/cost.joblib   # опционально
```

`src/harness/runtime/rank.py` читает `artifacts/models/` (или `RANK_MODELS=...`).
Docker копирует этот каталог в `/app/models`.

Не дублируй в корневой `models/` и не клади сырые HF-дампы в `datasets/` (см. `task_packs/SOURCES.md`).

Версия `scikit-learn` в ноутбуке и в `requirements.txt`: `1.9.1`.
