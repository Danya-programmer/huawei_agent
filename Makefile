# Local eval helpers for contest task packs.
PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
TASKS ?= task_packs/test-100
OUT ?= out
BUDGET ?=
MAX_WORKERS ?= 2
export OUT
export MAX_WORKERS

.PHONY: unpack-test clean-answers clean-journal run-test score-test eval-test unpack-abstention docker-test

unpack-test:
	unzip -qo task_packs/test-100.zip -d task_packs

unpack-abstention:
	unzip -qo task_packs/abstention-pack.zip -d task_packs

clean-answers:
	@find $(TASKS) -name answer.txt -type f -delete 2>/dev/null || true

# Fresh journal for this eval (old root-owned file silently blocked writes).
clean-journal:
	@mkdir -p $(OUT)
	@rm -f $(OUT)/cost_journal.jsonl 2>/dev/null || \
		(echo "WARN: cannot remove $(OUT)/cost_journal.jsonl (try: sudo rm $(OUT)/cost_journal.jsonl)"; exit 1)
	@touch $(OUT)/cost_journal.jsonl

# Do not re-unpack here — eval-test already unpacked + cleaned answers.
run-test:
	@mkdir -p $(OUT)
	@if [ -n "$(BUDGET)" ]; then export BUDGET="$(BUDGET)"; fi; \
	OUT=$(OUT) MAX_WORKERS=$(MAX_WORKERS) $(PYTHON) run_agent.py --tasks-dir $(TASKS)

score-test:
	@mkdir -p $(OUT)
	$(PYTHON) scripts/score_pack.py \
		--tasks-dir $(TASKS) \
		--journal $(OUT)/cost_journal.jsonl \
		--out-json $(OUT)/eval_report.json \
		--out-md $(OUT)/eval_report.md

# Full local eval: unpack → wipe answers → fresh journal → run → score
eval-test: unpack-test clean-answers clean-journal run-test score-test
	@echo "Open $(OUT)/eval_report.md for per-task results."

# Canonical Docker eval (jury-like flags via scripts/run_docker.sh).
docker-test: unpack-test clean-answers clean-journal
	sudo TASKS="$(CURDIR)/$(TASKS)" OUT="$(CURDIR)/$(OUT)" ./scripts/run_docker.sh
	@$(MAKE) score-test TASKS=$(TASKS) OUT=$(OUT)
