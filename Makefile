# Antibody aggregation pipeline — production runbook.
# Phases and modules are described in docs/superpowers/specs/2026-07-14-plm-scaleup-design.md
# and the handoff in HANDOFF_NEXT.md.

PY ?= python3
PROGRESS_EVERY ?= 0

.PHONY: help install test test-fast cohort head-a head-b head-b-plm embeddings ledgers gdpa3 clean

help:
	@echo "Targets:"
	@echo "  make install       install the Phase-2 stack (torch/transformers/ablang2/lightgbm/sklearn)"
	@echo "  make test          run the full test suite (stdlib-only, no training)"
	@echo "  make cohort        assemble + summarize the antibody cohort (dry run)"
	@echo "  make embeddings SEQS=seqs.txt   precompute ESM-2 + AbLang2 embedding cache"
	@echo "  make head-a        train the general ProteinGym fitness head (Head A)"
	@echo "  make head-b        Head B grouped-CV eval, biophysical features only"
	@echo "  make head-b-plm    Head B grouped-CV eval, PLM + biophysical features"
	@echo "  make ledgers       emit conflicts.tsv / exclusions.tsv"
	@echo "  make gdpa3 FILE=... HASH=...   single-shot GDPa3 external eval (frozen)"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m unittest discover -s tests -p "test_*.py"

# One quick self-check that the serve path matches the train path.
test-fast:
	$(PY) -m unittest tests.test_train_serve_parity tests.test_head_b_gbm

cohort:
	$(PY) -c "from data.cohort import build_full_cohort; build_full_cohort()"

embeddings:
	@test -n "$(SEQS)" || (echo "usage: make embeddings SEQS=path/to/sequences.txt" && exit 1)
	$(PY) -m features.plm --precompute $(SEQS) --model both

head-a:
	$(PY) -m model.pretrain_proteingym_fitness

head-b:
	$(PY) -m model.head_b_gbm --ledgers --progress-every $(PROGRESS_EVERY) --out results/head_b_report.json

head-b-plm:
	$(PY) -m model.head_b_gbm --use-plm --ledgers --progress-every $(PROGRESS_EVERY) --out results/head_b_report.json

ledgers:
	$(PY) -c "from data.cohort import build_antibody_cohort, emit_ledgers; emit_ledgers(build_antibody_cohort())"

gdpa3:
	@test -n "$(FILE)" -a -n "$(HASH)" || (echo "usage: make gdpa3 FILE=data/external/gdpa/GDPa3.csv HASH=<sha256>" && exit 1)
	$(PY) -m model.evaluate_gdpa3 --run-once --gdpa3-file $(FILE) --expected-data-hash $(HASH)

clean:
	rm -rf results/head_b_report.json results/conflicts.tsv results/exclusions.tsv
