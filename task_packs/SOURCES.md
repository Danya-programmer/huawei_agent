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
| `data_analysis` | [stanfordnlp/wikitablequestions](https://huggingface.co/datasets/stanfordnlp/wikitablequestions) (`revision=refs/pr/4`, config `random-split-1`) | CC-BY-4.0 | table + question + answer (parquet PR; script loaders unsupported in datasets≥4) | easy/med/hard by row count |
| `search` | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) (`distractor`) | CC-BY-SA-4.0 | question + context paragraphs → `workspace/kb/*.md` (namespaced id required by huggingface_hub) | easy=`bridge`, hard=`comparison` (else medium) |
| unsolvable / abstention | [facebook/AbstentionBench](https://huggingface.co/datasets/facebook/AbstentionBench) (paper [arXiv:2506.09038](https://arxiv.org/abs/2506.09038)) | CC-BY-NC-4.0 | `should_abstain=True` slices via underlying **FalseQA** + **SQuAD 2.0** unanswerable (+ underspecified math); HF script needs datasets≤3.6 so we load those sources directly | all `expected=UNSOLVABLE` |
| (FalseQA raw) | [thunlp/FalseQA](https://github.com/thunlp/FalseQA) `dataset/test.csv` | see repo | false-premise questions (`label=1`) → search tasks with decoy kb | unsolvable |
| (SQuAD 2.0) | [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) | CC-BY-SA-4.0 | empty-answer validation items → search tasks with passage kb | unsolvable |

Unsolvable items (~20%) in test/train are a **mix**:
1. **AbstentionBench-derived** (~50% of search/math unsolvable slots) — false premises / unanswerable / underspecified.
2. **Mutations** of the same upstream rows (missing evidence / broken schema / contradictory wording).

Standalone pack: `abstention-pack.zip` (100 abstain-only tasks).

## Rebuild

```bash
cd /home/dan/projects/huawei_agent_hacaton
# Put HF_TOKEN=hf_... into .env (optional but recommended for Hub rate limits)
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements-datasets.txt
python scripts/build_external_datasets.py --out task_packs --test-n 100 --train-n 3000 --seed 42
# abstention-pack only:
python scripts/build_external_datasets.py --out task_packs --abstention-only --abstention-n 100
```

Upstream dumps land in `.cache/hf_datasets/` (gitignored). Only converted zips are meant for git.
Do not name the output folder `datasets/` — it shadows the HuggingFace `datasets` package on `sys.path`.

## Citation

If you redistribute the zip packs, keep this file and attribute the upstream authors listed above.
