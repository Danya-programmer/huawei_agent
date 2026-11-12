# Dataset sources (contest task pack)

Converted contest tasks only are shipped in `test-100.zip` / `train-3000.zip`.
Original upstream dumps are **not** committed; the build script caches them under `.cache/hf_datasets/`.

## Task-type mapping

| Contest `task_type` | Upstream dataset | License (upstream) | What we take | Difficulty heuristic |
|---------------------|------------------|--------------------|--------------|----------------------|
| `math` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (`main`) | MIT | word problem + numeric final answer | easy |
| `math` | [DigitalLearningGmbH/MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) | see dataset card | problems with short extractable answers | medium if level ≤ 3, hard if level ≥ 4 |
| `coding` | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (`sanitized`) | CC-BY-4.0 | prompt + reference code + `test_list` | easy / medium by test count |
| `coding` | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | MIT | `prompt` + `canonical_solution` + `test` | medium / hard by test count |
| `data_analysis` | [stanfordnlp/wikitablequestions](https://huggingface.co/datasets/stanfordnlp/wikitablequestions) (`refs/convert/parquet`, config `random-split-1`) | see dataset card | table + question + answer (parquet; script loaders unsupported in datasets≥4) | easy/med/hard by row count |
| `search` | [hotpot_qa](https://huggingface.co/datasets/hotpot_qa) (`distractor`) | CC-BY-SA-4.0 | question + context paragraphs → `workspace/kb/*.md` | easy=`bridge`, hard=`comparison` (else medium) |

Unsolvable items (~20%) are **mutations** of the same upstream rows (missing evidence / broken schema / contradictory wording), not separate public “impossible” corpora.

## Rebuild

```bash
cd /home/dan/projects/huawei_agent_hacaton
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements-datasets.txt
python scripts/build_external_datasets.py --out task_packs --test-n 100 --train-n 3000 --seed 42
```

Upstream dumps land in `.cache/hf_datasets/` (gitignored). Only converted zips are meant for git.
Do not name the output folder `datasets/` — it shadows the HuggingFace `datasets` package on `sys.path`.

## Citation

If you redistribute the zip packs, keep this file and attribute the upstream authors listed above.
