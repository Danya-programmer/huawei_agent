# Contest task packs

| File | Contents |
|------|----------|
| [SOURCES.md](SOURCES.md) | Upstream dataset links, licenses, mapping |
| `test-100.zip` | 100 tasks (25 per type, 20% unsolvable) + `manifest.jsonl` |
| `train-3000.zip` | 3000 tasks, same quotas scaled |
| `abstention-pack.zip` | 100 AbstentionBench-derived unsolvable-only tasks |

## Unpack

```bash
cd /home/dan/projects/huawei_agent_hacaton
unzip -qo task_packs/test-100.zip -d task_packs
unzip -qo task_packs/train-3000.zip -d task_packs
unzip -qo task_packs/abstention-pack.zip -d task_packs
# → task_packs/test-100/  train-3000/  abstention-pack/
```

Each task dir: `task.json` + optional `workspace/`. Labels for local eval: `manifest.jsonl`
(`solvable`, `difficulty`, `expected`, `source`, `fail_mode`).

## Rebuild from HuggingFace

Needs network once (HF + PyPI):

```bash
uv venv .venv && source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements-datasets.txt
# full packs + abstention-pack.zip
python scripts/build_external_datasets.py --out task_packs --test-n 100 --train-n 3000 --seed 42
# abstention only:
python scripts/build_external_datasets.py --out task_packs --abstention-only --abstention-n 100 --seed 42
```

Smoke-only (faster): add `--test-only`.
