# Routed Multi-Expert Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and honestly evaluate an opt-in three-property antibody platform that preserves known-family specialists while improving or retaining source-held-out generalization.

**Architecture:** Reuse the existing registry, feature builders, overlap stripping, calibration, applicability, uncertainty, and policy. Add deterministic routing and simple nonnegative blends, evaluate selection inside nested source holdouts, then train a separate experimental artifact; do not touch the current serving artifact.

**Tech Stack:** Python 3, standard library, existing LightGBM, scikit-learn, NumPy, SQLite, pickle, JSON, CSV, unittest.

## Global Constraints

- Aggregation, hydrophobicity, and polyreactivity remain separate outputs.
- Higher risk always means worse developability.
- Use seeds 42, 43, and 44.
- Strip exact VH, exact VL, and equal-length VH identity >=0.90 from every held-source training split.
- Never use a source name as an inference input.
- Never load, inspect, count, train on, or tune against GDPa3.
- Reconstructed rows may train only when eligible and may never be holdouts.
- Never zero-fill missing PLM embeddings and call the result a PLM prediction.
- Add no production dependency.
- Do not modify model/head_platform.py or model/saved/platform.pkl.
- Preserve unrelated dirty work.
- Specialist and unseen-source AUCs remain separate; never pool them.

## File Map

- Create model/developability_experts.py: routing and equal-weight blend selection.
- Create tests/test_developability_experts.py: pure expert tests.
- Modify model/developability_matrix.py: reusable lgbm, histgb, and ridge fit path.
- Modify tests/test_developability_matrix.py: candidate-fit regression checks.
- Create model/developability_multi_expert.py: nested evaluation, specialists, artifact, inference CLI.
- Create tests/test_developability_multi_expert.py: leakage and artifact checks.
- Create tools/multi_expert_gate.py and tests/test_multi_expert_gate.py: baseline/trial gate.
- Modify PLATFORM_SCORECARD.md only after the real run.

---

### Task 1: Pure routing and blend selection

**Files:**
- Create: model/developability_experts.py
- Create: tests/test_developability_experts.py

**Interfaces:**
- positional_identity(query: str, reference: str) -> float
- choose_specialist(vh: str, references: dict[str, list[str]], threshold: float = 0.90) -> dict | None
- candidate_combinations(names: list[str]) -> list[tuple[str, ...]]
- select_blend(predictions: dict[str, dict[str, dict[str, tuple[int, float]]]], min_coverage: float = 0.80) -> dict

- [ ] **Step 1: Write failing tests**

~~~python
import unittest
from model.developability_experts import (
    candidate_combinations, choose_specialist, positional_identity, select_blend,
)

class TestDevelopabilityExperts(unittest.TestCase):
    def test_identity_and_router(self):
        self.assertEqual(positional_identity("AAAA", "AAAT"), 0.75)
        self.assertEqual(positional_identity("AAAA", "AAA"), 0.0)
        refs = {"zeta": ["AAAAC"], "alpha": ["AAAAT"]}
        self.assertEqual(
            choose_specialist("AAAAG", refs, 0.80),
            {"name": "alpha", "identity": 0.8},
        )
        self.assertIsNone(choose_specialist("AAAAG", refs, 0.90))

    def test_combinations_are_stable(self):
        self.assertEqual(
            candidate_combinations(["b", "a"]),
            [("a",), ("b",), ("a", "b")],
        )

    def test_selects_macro_source_winner(self):
        predictions = {
            "a": {
                "weak": {"0": (0, .8), "1": (1, .2)},
                "strong": {"0": (0, .1), "1": (1, .9)},
            },
            "b": {
                "weak": {"0": (0, .7), "1": (1, .3)},
                "strong": {"0": (0, .2), "1": (1, .8)},
            },
        }
        out = select_blend(predictions)
        self.assertEqual(out["members"], ["strong"])
        self.assertEqual(out["macro_source_auc"], 1.0)

if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~bash
~/Biostuff/.venv/bin/python -m unittest tests.test_developability_experts -v
~~~

Expected: ModuleNotFoundError for model.developability_experts.

- [ ] **Step 3: Implement the minimal pure module**

~~~python
from __future__ import annotations
import itertools
import statistics

def positional_identity(query: str, reference: str) -> float:
    query = str(query or "").strip().upper()
    reference = str(reference or "").strip().upper()
    if not query or len(query) != len(reference):
        return 0.0
    return sum(a == b for a, b in zip(query, reference)) / len(query)

def choose_specialist(vh, references, threshold=0.90):
    eligible = []
    for name in sorted(references):
        identity = max((positional_identity(vh, ref) for ref in references[name]), default=0.0)
        if identity >= threshold:
            eligible.append((identity, name))
    if not eligible:
        return None
    identity, name = min(eligible, key=lambda item: (-item[0], item[1]))
    return {"name": name, "identity": identity}

def candidate_combinations(names):
    names = sorted(set(names))
    return [
        combo
        for size in range(1, len(names) + 1)
        for combo in itertools.combinations(names, size)
    ]

def select_blend(predictions, min_coverage=0.80):
    from sklearn.metrics import roc_auc_score
    common_names = sorted(set.intersection(*(set(items) for items in predictions.values())))
    eligible = []
    for name in common_names:
        if all(
            len(items[name]) >= min_coverage * max(len(rows) for rows in items.values())
            for items in predictions.values()
        ):
            eligible.append(name)
    if not eligible:
        raise ValueError("no candidate has sufficient common prediction coverage")
    ranked = []
    for members in candidate_combinations(eligible):
        source_auc = {}
        for source, items in predictions.items():
            ids = sorted(set.intersection(*(set(items[name]) for name in members)))
            labels = [items[members[0]][record_id][0] for record_id in ids]
            if len(set(labels)) != 2:
                continue
            scores = [
                statistics.fmean(items[name][record_id][1] for name in members)
                for record_id in ids
            ]
            source_auc[source] = float(roc_auc_score(labels, scores))
        if source_auc:
            ranked.append((statistics.fmean(source_auc.values()), len(members), members, source_auc))
    if not ranked:
        raise ValueError("no two-class source predictions")
    macro, _, members, source_auc = min(ranked, key=lambda item: (-item[0], item[1], item[2]))
    return {
        "members": list(members),
        "macro_source_auc": macro,
        "source_auc": source_auc,
        "eligible_candidates": eligible,
    }
~~~

- [ ] **Step 4: Run tests and commit**

~~~bash
~/Biostuff/.venv/bin/python -m unittest tests.test_developability_experts -v
git add model/developability_experts.py tests/test_developability_experts.py
git commit -m "feat: add deterministic developability expert routing"
~~~

Expected: 3 tests pass.

---

### Task 2: Reusable comparable learners

**Files:**
- Modify: model/developability_matrix.py near _model and _source_result
- Modify: tests/test_developability_matrix.py

**Interfaces:**
- fit_candidate(learner, fit_featured, calibration_featured, test_featured, seed) -> dict
- Learners: lgbm, histgb, ridge
- Return keys: model, calibrator, raw_scores, scores, calibration_method
- fit_applicability_index(train_featured) -> dict and score_applicability(index, test_featured) -> list[dict]

- [ ] **Step 1: Add the failing learner test**

~~~python
from model.developability_matrix import fit_candidate

class TestCandidateFitting(unittest.TestCase):
    def test_supported_learners_return_probabilities(self):
        featured = [
            (row(f"r{i}", "train", i % 2, target=i % 2), [float(i % 2), float(i)])
            for i in range(40)
        ]
        for learner in ("lgbm", "histgb", "ridge"):
            out = fit_candidate(learner, featured[:30], featured[30:35], featured[35:], 42)
            self.assertEqual(len(out["scores"]), 5)
            self.assertTrue(all(0 <= value <= 1 for value in out["scores"]))

    def test_unknown_learner_fails(self):
        with self.assertRaisesRegex(ValueError, "unsupported learner"):
            fit_candidate("forest", [], [], [], 42)

    def test_calibrator_is_pickleable(self):
        import pickle
        featured = [
            (row(f"r{i}", "train", i % 2, target=i % 2), [float(i % 2), float(i)])
            for i in range(40)
        ]
        out = fit_candidate("lgbm", featured[:30], featured[30:35], featured[35:], 42)
        restored = pickle.loads(pickle.dumps(out["calibrator"]))
        self.assertEqual(len(restored(out["raw_scores"])), 5)
~~~

- [ ] **Step 2: Run the focused failure**

~~~bash
~/Biostuff/.venv/bin/python -m unittest tests.test_developability_matrix.TestCandidateFitting -v
~~~

Expected: ImportError for fit_candidate.

- [ ] **Step 3: Implement classifier construction and shared calibration**

Add _candidate_model with:
- lgbm returning existing _model(seed)
- histgb returning HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=seed)
- ridge returning make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=seed))
- unknown learner raising ValueError("unsupported learner: ...")

Replace the closure returned by _platt_calibrator with top-level pickleable
IdentityCalibrator and PlattCalibrator dataclasses. Each implements
__call__(scores) and PlattCalibrator stores only float coefficient/intercept.

Add fit_candidate that:
1. derives X, y, and existing source/class _weights;
2. passes logisticregression__sample_weight for ridge and sample_weight otherwise;
3. predicts the existing train-only calibration split;
4. calls _platt_calibrator;
5. returns raw and calibrated test probabilities, fitted model, and calibrator.

Extract the existing applicability math into fit_applicability_index and
score_applicability. The stored index contains the baseline-feature center,
scale, source centroids, p95, and p99. Keep _applicability as a compatibility
wrapper around those two functions so existing reports do not change.

Refactor _source_result to call fit_candidate("lgbm", ...) without changing its JSON schema.

- [ ] **Step 4: Run matrix regressions and commit**

~~~bash
~/Biostuff/.venv/bin/python -m unittest tests.test_developability_matrix -v
git add model/developability_matrix.py tests/test_developability_matrix.py
git commit -m "refactor: expose comparable developability learners"
~~~

Expected: all matrix tests pass.

---

### Task 3: Nested generalist evaluation

**Files:**
- Create: model/developability_multi_expert.py
- Create: tests/test_developability_multi_expert.py

**Interfaces:**
- CandidateSpec(name, feature_mode, training_arm, learner)
- nested_generalist(rows, held_sources, property_name, seed, max_per_source_class=5000, near_identity=0.90) -> dict
- summarize_generalists(runs) -> dict
- Each outer-source result contains baseline and trial metrics computed on the
  same record IDs, plus subgroup and overlap audits.

Use these fixed candidates:

~~~python
CANDIDATES = {
    "polyreactivity": (
        CandidateSpec("cdr_library_lgbm", "cdr_aware", "library_only", "lgbm"),
        CandidateSpec("bio_mixed_lgbm", "biophysical_charge", "mixed", "lgbm"),
        CandidateSpec("plm_mixed_lgbm", "frozen_plm", "mixed", "lgbm"),
    ),
    "aggregation": (
        CandidateSpec("bio_diverse_lgbm", "biophysical_charge", "diverse_only", "lgbm"),
        CandidateSpec("plm_diverse_lgbm", "frozen_plm", "diverse_only", "lgbm"),
        CandidateSpec("plm_diverse_histgb", "frozen_plm", "diverse_only", "histgb"),
        CandidateSpec("plm_diverse_ridge", "frozen_plm", "diverse_only", "ridge"),
    ),
    "hydrophobicity": (
        CandidateSpec("bio_mixed_lgbm", "biophysical_charge", "mixed", "lgbm"),
        CandidateSpec("plm_mixed_lgbm", "frozen_plm", "mixed", "lgbm"),
        CandidateSpec("bio_mixed_histgb", "biophysical_charge", "mixed", "histgb"),
        CandidateSpec("bio_mixed_ridge", "biophysical_charge", "mixed", "ridge"),
    ),
}
~~~

- [ ] **Step 1: Write failing leakage tests**

Build three synthetic diverse sources with 12 rows and balanced targets. Patch _predict_spec to return deterministic predictions. Assert:
- changing only the outer source labels cannot change that outer source's inner-selected members;
- an exact or >=0.90 VH match to the outer source never appears in any _predict_spec training argument;
- reconstructed holdout-ineligible rows never appear in test arguments.

- [ ] **Step 2: Confirm failure**

~~~bash
~/Biostuff/.venv/bin/python -m unittest tests.test_developability_multi_expert.TestNestedGeneralist -v
~~~

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement _predict_spec**

It must:
1. filter rows using existing _training_arm_rows;
2. call existing make_featurizer;
3. discard None vectors;
4. require >=0.80 real train and test coverage and >=40 training rows;
5. split training with _fit_calibration_split;
6. call fit_candidate;
7. return record_id -> (target, calibrated_score).

- [ ] **Step 4: Implement nested_generalist in this exact order**

For every outer source:
1. select only holdout_eligible outer rows;
2. select only training_eligible non-outer rows;
3. strip exact VH, exact VL, and >=0.90 equal-length VH overlap;
4. for every remaining inner development source, hold it out;
5. source/class cap inner training rows;
6. strip inner overlaps;
7. collect every candidate's inner predictions;
8. select the best single candidate and best equal-weight combination with select_blend;
9. refit both the selected single baseline and selected blend components on capped outer training rows;
10. intersect their outer prediction record IDs so baseline and trial use identical antibodies;
11. compute classification_metrics for baseline and trial per assay, then unweighted source metrics;
12. compute scorable VH-length and VL-present subgroup metrics without changing the primary macro;
13. save feature coverage, overlap, candidate-selection, calibration, applicability, and subgroup audits.

The outer source labels may be used only in step 11.

- [ ] **Step 5: Implement summarize_generalists**

For each property, preserve:
- seed_macro_source_auc in seed order;
- mean_macro_source_auc;
- seed_std_auc using statistics.pstdev;
- source_mean_auc;
- mean enrichment_at_15pct;
- every seed's selected components and overlap audits;
- source bootstrap CI using the existing source-bootstrap helper.
- baseline and trial summaries with identical source/record coverage.
- a matrix-shaped trial projection accepted by tools.promotion_gate.evaluate.

Do not round JSON values.

- [ ] **Step 6: Run tests and commit**

~~~bash
~/Biostuff/.venv/bin/python -m unittest tests.test_developability_multi_expert.TestNestedGeneralist -v
git add model/developability_multi_expert.py tests/test_developability_multi_expert.py
git commit -m "feat: add nested developability generalists"
~~~

Expected: all nested and leakage tests pass.

---

### Task 4: Specialists, artifact, and inference

**Files:**
- Modify: model/developability_multi_expert.py
- Modify: tests/test_developability_multi_expert.py

**Interfaces:**
- evaluate_specialists(rows, property_name, seed) -> dict
- train_experimental_artifact(rows_by_property, report, path) -> dict
- predict_artifact(path, vh, vl="") -> dict

- [ ] **Step 1: Write failing artifact tests**

Create a pickled schema_version 1 fixture with:
- threshold 0.90;
- one constant-risk generalist;
- one specialist with an all-A reference VH;
- three seed models.

Assert:
- an all-A query selects the specialist without a source-name input;
- an all-C query selects the generalist;
- risk_score is the seed mean;
- ensemble_disagreement is the seed population standard deviation;
- invalid or short VH raises a standard-20-AA error;
- a selected frozen_plm component with a cache miss raises an exact features.plm precompute command.

- [ ] **Step 2: Implement specialist evaluation**

For each LIBRARY source with enough classes:
1. deduplicate exact VH;
2. run StratifiedKFold with shuffled seeds 42/43/44;
3. train the biophysical_charge lgbm candidate;
4. report ROC AUC, PR AUC, enrichment@15, n, positives;
5. report router coverage: validation rows having >=0.90 identity to a training VH.

Do not combine these values with the unseen-source generalist macro.

- [ ] **Step 3: Implement artifact training**

Artifact schema:

~~~python
{
    "schema_version": "1",
    "head": "developability_multi_expert",
    "router_threshold": 0.90,
    "seeds": [42, 43, 44],
    "feature_schema": "candidate-specific; no zero fill",
    "training_manifest": report["training_manifest"],
    "properties": {
        property_name: {
            "generalist": seeded_component_models,
            "applicability_index": baseline_feature_index,
            "specialists": {
                source: {
                    "references": sorted_unique_vh,
                    "models": seeded_models,
                }
            },
        }
    },
}
~~~

Select final generalist members by majority across outer folds, breaking ties by fewer members and then lexical order. Store unique specialist references grouped by source. Write only model/saved/platform_experimental.pkl.

Store every fitted component's pickleable calibrator next to its model. Fit the
applicability index only on biophysical_charge training vectors so it is
available even when a PLM component is absent for a query.

- [ ] **Step 4: Implement inference**

Validate VH and optional VL against ACDEFGHIKLMNPQRSTVWY and length >=20. Use choose_specialist without a source label. Build each selected component's exact training feature mode. On frozen PLM cache miss, raise:

~~~text
frozen PLM cache miss; run: ~/Biostuff/.venv/bin/python -m features.plm --precompute results/query_sequences.txt --model both
~~~

Return per property:
- risk_score;
- expert_type and expert_name;
- selected components;
- nearest_family_identity;
- applicability and nearest training source;
- ensemble_disagreement;
- abstain and abstention_reason;
- artifact schema and manifest identifier.

Apply each stored calibrator before averaging component/seed scores. Compute
applicability with score_applicability against the stored baseline-feature
index; out_of_domain adds an abstention reason.

- [ ] **Step 5: Add one CLI**

~~~bash
~/Biostuff/.venv/bin/python -m model.developability_multi_expert \
  --database results/developability_registry.sqlite \
  --policy tools/benchmark_policy.json \
  --seeds 42,43,44 \
  --max-per-source-class 5000 \
  --near-identity 0.90 \
  --out results/developability_multi_expert.json \
  --predictions results/developability_multi_expert_predictions.csv \
  --artifact model/saved/platform_experimental.pkl
~~~

The CLI loads development sources only from policy, raises if any loaded source is gdpa3, prints separate generalist and specialist tables, and saves all row counts, coverage, overlaps, selections, uncertainty, and metrics.

- [ ] **Step 6: Run tests and commit**

~~~bash
~/Biostuff/.venv/bin/python -m unittest tests.test_developability_multi_expert -v
git add model/developability_multi_expert.py tests/test_developability_multi_expert.py
git commit -m "feat: build opt-in multi-expert artifact"
~~~

Expected: all nested, specialist, validation, PLM, routing, and artifact tests pass.

---

### Task 5: Experimental and serving gates

**Files:**
- Create: tools/multi_expert_gate.py
- Create: tests/test_multi_expert_gate.py

**Interfaces:**
- evaluate(baseline: dict, trial: dict, policy: dict) -> dict
- Experimental thresholds: macro delta >=0.010; each source delta >=-0.030; enrichment delta >=0; seed std <=0.030; required audits present.
- Existing tools.promotion_gate.evaluate remains the sole serving decision.

- [ ] **Step 1: Write failing gate tests**

Test one admitted 0.700 -> 0.720 trial with no source losing >0.030 and stable seeds. Assert experimental_admit true but serving_eligible false without a locked source. Test a source drop from 0.700 to 0.660 and assert rejection.

Also patch tools.promotion_gate.evaluate and assert it was called with the
report's matrix-shaped trial projection; serving_eligible must exactly equal
that returned decision's promote value.

- [ ] **Step 2: Implement the gate**

For each property return:
- baseline_macro_auc;
- trial_macro_auc;
- macro_delta_auc;
- source_delta_auc;
- enrichment delta;
- every boolean check;
- experimental_admit;
- serving_eligible;
- serving_reason.

Load tools/benchmark_policy.json and require GDPa3 prior_use false. Call the
existing promotion gate for absolute macro/source/enrichment/seed/locked-source
checks; do not duplicate or weaken them. Evaluate each specialist separately:
admit it only when row/class minimums pass and its three-seed mean AUC is no
more than 0.005 below the best comparable saved specialist. Specialist admission
must not change the generalist macro or serving decision. No gate code may train
or write a model.

- [ ] **Step 3: Add the CLI and run tests**

~~~bash
~/Biostuff/.venv/bin/python -m unittest \
  tests.test_multi_expert_gate \
  tests.test_promotion_gate \
  tests.test_benchmark_governance -v
git add tools/multi_expert_gate.py tests/test_multi_expert_gate.py
git commit -m "feat: gate multi-expert development gains"
~~~

Expected: all gate and governance tests pass.

---

### Task 6: Real run, independent verification, and iteration

**Files:**
- Modify: PLATFORM_SCORECARD.md
- Generate: results/developability_multi_expert.json
- Generate: results/developability_multi_expert_predictions.csv
- Generate: results/developability_multi_expert_gate.json
- Generate: model/saved/platform_experimental.pkl

- [ ] **Step 1: Run focused and full tests**

~~~bash
cd ~/Biostuff
~/Biostuff/.venv/bin/python -m unittest \
  tests.test_developability_experts \
  tests.test_developability_matrix \
  tests.test_developability_multi_expert \
  tests.test_multi_expert_gate \
  tests.test_developability_uncertainty \
  tests.test_promotion_gate \
  tests.test_train_serve_parity -v
~/Biostuff/.venv/bin/python -m unittest discover -s tests -v
~~~

Expected: all tests pass with only previously documented skips. Stop the expensive run on any failure and fix the root cause first.

- [ ] **Step 2: Run the three-seed experiment and gate**

Run the Task 4 CLI, then:

~~~bash
~/Biostuff/.venv/bin/python -m tools.multi_expert_gate \
  --report results/developability_multi_expert.json \
  --policy tools/benchmark_policy.json \
  --out results/developability_multi_expert_gate.json
~~~

Expected: exact baseline, trial, source deltas, specialist metrics, and decisions; no prescribed winning number.

- [ ] **Step 3: Verify through a second parser**

Independently load JSON, CSV, and pickle. Assert:
- schema_version is 1;
- router threshold is 0.90;
- seeds are [42, 43, 44];
- prediction CSV is nonempty;
- no report source equals gdpa3;
- every serving_eligible is false while locked_validation is empty;
- source means recomputed from per-seed values match saved values.

Print the real prediction count only after all assertions pass.

- [ ] **Step 4: Iterate autonomously without converting the holdout into training data**

If the first combination fails, inspect manifests, coverage, overlap audits,
feature parity, calibration, and inner-fold selections. Fix implementation or
data-contract defects at their root and rerun all three seeds. Do not replace a
component merely because its outer-source score is disappointing: that would
train on the holdout.

Any additional CDR-H3, structure, paired AbLang, assay-context, threshold, or
candidate experiment requires a committed A/B amendment before it runs. Because
the current development sources have already been observed, such follow-ups are
labelled exploratory and cannot pass the serving gate without genuinely new
locked data.

- [ ] **Step 5: Update the scorecard with artifact-backed values**

Append a dated section containing:
- each property's baseline -> trial macro AUC and delta;
- every held-source AUC and delta;
- enrichment@15 and seed std;
- specialist AUC and router coverage in a separate table;
- admitted or quarantined reason;
- exact test totals;
- the sentence: Serving unchanged: no genuinely locked validation source is assigned; GDPa3 remains untouched.

Copy values only from saved JSON and round Markdown display to three decimals.

- [ ] **Step 6: Commit evidence**

~~~bash
git add PLATFORM_SCORECARD.md \
  results/developability_multi_expert.json \
  results/developability_multi_expert_gate.json
git commit -m "docs: report multi-expert development evidence"
~~~

Do not commit the pickle or row-level CSV unless repository policy already tracks those types.

- [ ] **Step 7: Final report**

Report exact test totals, real per-property baseline-to-trial AUC, all source deltas, specialist metrics, gate decisions, artifact path, iteration attempts, and PLM cache requirements. State plainly that specialist and unseen-source numbers measure different operating regimes.

## Execution Mode

The user explicitly authorized autonomous execution. Use executing-plans inline, preserve checkpoints in commits, fix root causes, and continue through the real gate without asking for routine confirmations.
