# CDR feature gates — implementation plan

> **For Codex:** Execute inline. The active session may not delegate this task.

**Goal:** Test dependency-free heavy-chain CDR descriptors against the fixed diverse-polyreactivity and AC-SINS holdouts without changing any existing `model/head_*.py` file.

**Architecture:** Add one small CDR utility and two standalone gate scripts. Each gate builds the existing biophysical-plus-charge feature vector, then appends either no CDR values, H3-only values, or (only when H3 meets the threshold) H1+H2+H3 values. The scripts write reports, not serving artifacts: `head_platform.py` currently owns the runtime feature construction and is explicitly out of scope.

**Tech stack:** Python stdlib, installed NumPy/scikit-learn/LightGBM, existing project feature/model helpers.

---

### Task 1: Add a minimal, dependency-free CDR utility

**Files:**
- Create: `model/cdr_features.py`
- Create: `tests/test_cdr_features.py`

**Steps:**
1. Prefer nonempty row fields for H1/H2/H3, accepting the common `vh_cdr1`, `vh_cdr2`, `vh_cdr3` and `cdrh1`, `cdrh2`, `cdrh3` spellings before parsing sequence.
2. For missing supplied fields, parse H3 only with the documented FR3-Cys through FR4 `WGxG` motif. Do not claim H1/H2 parsing when those fields are absent; their descriptor values remain zeros and their validity flags remain zero.
3. Return five values per requested heavy CDR: length, net charge at pH 7.4, mean Kyte-Doolittle hydropathy, aromatic fraction, valid flag. Modes `h3` and `heavy_all` therefore add 5 and 15 values respectively.
4. Add a small `unittest` check for supplied-field precedence, a successful H3 parse, and an unparseable sequence producing zeros plus flag zero.
5. Run: `/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_cdr_features`.

### Task 2: Add the fixed diverse-polyreactivity A/B gate

**Files:**
- Create: `tools/gate_cdr_polyreactivity.py`

**Steps:**
1. Stream the polyreactivity CSV and retain the existing five library sources for training plus the fixed holdouts `aims_mouse`, `aims_flu`, and `shehata2019`.
2. Preserve the existing deterministic per-class reservoir cap, binary-label conversion, model hyperparameters, and strict held-out-VH removal used by `tools/gate_harvey_vhh.py`.
3. Run baseline and H3-only models. Report per-source row/class counts, parse rates, each held-out AUC, and mean AUC. State that small holdouts are unstable.
4. Run H1+H2+H3 only if H3 mean AUC is at least baseline minus 0.005. Since no H1/H2 fields exist in the active CSV, its parse-rate report must make that limitation visible.
5. Write JSON to `results/gate_cdr_polyreactivity.json` and print the gate decision.
6. Run: `/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.gate_cdr_polyreactivity`.

### Task 3: Add the AC-SINS external-study A/B gate

**Files:**
- Create: `tools/gate_cdr_acsins.py`

**Steps:**
1. Reuse `model.head_b_acsins_unified._acsins_rows`, its strict shared-sequence removal, and the existing charge-model regressor/metrics helpers.
2. Evaluate baseline versus H3 on the same leave-one-study-out AC-SINS task; summarize solid held-out studies only (`n >= 100`) as the gate metric and retain per-study values in JSON.
3. Run H1+H2+H3 only if the H3 polyreactivity trial satisfies the threshold. Print parse rates for every arm.
4. Require each reported trial metric to be no lower than its baseline by more than 0.005. Save `results/gate_cdr_acsins.json`.
5. Run: `/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.gate_cdr_acsins`.

### Task 4: Record the result and decide on serving integration

**Files:**
- Modify: `tools/MORNING_DIGEST.md`

**Steps:**
1. Record exact baselines, trials, parse rates, and acceptance/quarantine decision.
2. If both fixed gates pass, state that integration needs a separately approved change to `model/head_platform.py`; do not refresh a platform artifact whose runtime cannot generate these feature columns.
3. If either gate fails, record the CDR feature set as quarantined with no serving change.
4. Run `git diff --check`, re-run the unit test, inspect only relevant diff, and commit only the new CDR files and digest.

