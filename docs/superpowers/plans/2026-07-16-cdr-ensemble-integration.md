# CDR Ensemble Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify CDR-H3 on the strict AC-SINS ensemble, test raw-VH H1/H2/H3 on the diverse polyreactivity holdouts, and integrate only passing features.

**Architecture:** Add an `add_cdr_h3` switch to the existing ensemble A/B, extend the dependency-free CDR utility with bounded raw-VH H1/H2 motif fallbacks, then append only passed CDR columns in the matching platform head.

**Tech Stack:** Python stdlib, existing LightGBM/scikit-learn ensemble, current project feature helpers.

## Global Constraints

- Keep strict held-out-study/VH removal and the `trial >= baseline - 0.005` gate.
- Do not touch GDPa3, unrelated heads, or unrelated untracked files.
- Modify `model/head_platform.py` only for a CDR feature that passes this run.

---

### Task 1: Ensemble H3 A/B

**Files:**
- Modify: `model/head_b_acsins_ensemble.py`
- Modify: `tests/test_cdr_features.py`

**Interfaces:** `run_ensemble(..., add_cdr_h3: bool = False) -> dict` appends `cdr_feature_vector(row, "h3")` when true. CLI `--ab-cdr-h3` runs baseline and CDR arms in one process.

- [ ] Add the descriptor-shape assertion:

```python
assert len(cdr_feature_vector({"vh_sequence": "AAAACARDRWGQG"}, "h3")) == 5
```

- [ ] Add `add_cdr_h3` after the existing optional feature switches, import the utility only when requested, and append its five values within `feat(i)`.
- [ ] Add the CLI A/B branch beside `--ab-charge`, with JSON keys `baseline`, `with_features`, `features`, and `solid_delta`.
- [ ] Run `/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_cdr_features` then `/Users/saptarshighosh/Biostuff/.venv/bin/python -m model.head_b_acsins_ensemble --ab-cdr-h3 --out results/head_b_acsins_ensemble_cdr_h3.json`.

### Task 2: Raw H1/H2/H3 polyreactivity A/B

**Files:**
- Modify: `model/cdr_features.py`
- Modify: `tests/test_cdr_features.py`
- Modify: `tools/gate_cdr_polyreactivity.py`

**Interfaces:** `heavy_cdrs(row) -> dict[str, str]` returns raw-VH H1/H2 fallbacks when supplied fields are absent. `cdr_feature_vector(row, "heavy_all") -> list[float]` remains 15 values.

- [ ] Add failing raw-sequence tests:

```python
row = {"vh_sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVAIYPGDSDTRYSPSFQGQVTISADKSISTAYLQW"}
assert heavy_cdrs(row)["h1"]
assert heavy_cdrs(row)["h2"]
```

- [ ] Implement bounded standard-AA regex fallbacks: H1 between the FR1 Cys and first downstream `W[IV]`; H2 between the downstream FR2 Trp and first `R[FT]`. Keep supplied values first and return empty strings on failure.
- [ ] Change `tools/gate_cdr_polyreactivity.py` to always run direct baseline versus `heavy_all`, report individual H1/H2/H3 parser rates, and remove its claim that the all-heavy arm is zero-filled.
- [ ] Run `/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_cdr_features` then `/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.gate_cdr_polyreactivity`.

### Task 3: Conditional platform integration

**Files:**
- Modify: `model/head_platform.py`
- Modify: `tools/MORNING_DIGEST.md`

**Interfaces:** For a passing aggregation gate, append `cdr_feature_vector(row, "h3")` only in its aggregation buckets. For a passing polyreactivity gate, append `cdr_feature_vector(row, "heavy_all")` only in its polyreactivity buckets.

- [ ] Encode only the factual gate outcome from Tasks 1–2; leave nonpassing heads on their existing feature vector. Update artifact metadata per head so model width is inspectable.
- [ ] Run `/Users/saptarshighosh/Biostuff/.venv/bin/python -m model.head_platform --out results/platform_scorecard.json --artifact model/saved/platform.pkl` only if at least one gate passes; inspect its pickle metadata with Python.
- [ ] Append exact outcome, parser rates, integration/no-integration decision to the digest; run unit tests and `git diff --check`; commit only the CDR files and digest.
