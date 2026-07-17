# Evidence-Tier Data Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit independently verified parent-plus-mutation reconstructions as training-only evidence while keeping direct holdouts uncontaminated.

**Architecture:** Add a dependency-free substitution validator, store evidence quality and eligibility beside the existing supervision tier, and enforce those eligibility flags in the source-held-out matrix. Existing direct inputs keep their current behavior; optional `reconstructed_<property>_dataset.csv` files are discovered only when present.

**Tech Stack:** Python standard library, SQLite, existing `data.contract` helpers, and `unittest`.

## Global Constraints

- Preserve all unrelated dirty work.
- Do not infer labels, sequences, or chain boundaries.
- Reconstructed rows require two distinct provenance references plus an independently supplied full sequence or SHA-256 digest.
- Reconstructed rows may train but may never be held out for evaluation.
- Do not touch GDPa3 or automatically promote a serving model.
- Add no dependency.

---

### Task 1: Deterministic substitution reconstruction

**Files:**
- Create: `data/sequence_reconstruction.py`
- Create: `tests/test_sequence_reconstruction.py`

**Interfaces:**
- Consumes: a row containing `parent_vh`, optional `parent_vl`, chain mutation strings, independent expected sequence/digest fields, and two provenance references.
- Produces: `reconstruct_verified_pair(row: Mapping[str, object]) -> tuple[str, str | None, dict]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_reconstructs_substitution_and_records_all_checks():
    vh, vl, audit = reconstruct_verified_pair(valid_row())
    assert vh[22] == "G"
    assert vl is not None
    assert audit["chains"]["vh"]["round_trip_diff_check"] is True

def test_rejects_wrong_parent_residue_and_nonindependent_reference():
    with self.assertRaisesRegex(ValueError, "does not match parent"):
        reconstruct_verified_pair({**valid_row(), "vh_mutations": "C23G"})
    with self.assertRaisesRegex(ValueError, "distinct provenance"):
        reconstruct_verified_pair({**valid_row(), "verification_reference": "mutation-table"})
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_sequence_reconstruction -v`

Expected: import failure because `data.sequence_reconstruction` does not exist.

- [ ] **Step 3: Implement the minimum validator**

```python
MUTATION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])")

def apply_substitutions(parent: str, text: object) -> tuple[str, dict]:
    sequence = canonical_sequence(parent)
    mutations = parse_substitutions(text)
    child = list(sequence)
    for position, wild_type, replacement in mutations:
        if position > len(sequence) or sequence[position - 1] != wild_type:
            raise ValueError(f"mutation {wild_type}{position}{replacement} does not match parent")
        child[position - 1] = replacement
    rebuilt = "".join(child)
    if sequence_diff(sequence, rebuilt) != mutations:
        raise ValueError("round-trip mutation check failed")
    return rebuilt, {"parent_residue_check": True, "round_trip_diff_check": True}
```

`reconstruct_verified_pair` must also validate the independent expected sequence or SHA-256 digest and reject identical provenance references.

- [ ] **Step 4: Run the focused tests**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_sequence_reconstruction -v`

Expected: all reconstruction tests pass.

- [ ] **Step 5: Commit**

```bash
git add data/sequence_reconstruction.py tests/test_sequence_reconstruction.py
git commit -m "feat: verify deterministic antibody reconstructions"
```

### Task 2: Registry evidence and eligibility contract

**Files:**
- Modify: `data/developability_registry.py`
- Modify: `tests/test_developability_registry.py`

**Interfaces:**
- Consumes: current table formats plus `reconstructed_sequence_table`.
- Produces: record columns `evidence_tier`, `training_eligible`, `holdout_eligible`, and `verification_json`; report keys `evidence_counts`, `training_eligible_rows`, and `holdout_eligible_rows`.

- [ ] **Step 1: Add failing registry assertions**

```python
direct = connection.execute(
    "SELECT evidence_tier,training_eligible,holdout_eligible FROM records WHERE source='shehata2019'"
).fetchone()
self.assertEqual(direct, ("direct", 1, 1))

reconstructed = connection.execute(
    "SELECT evidence_tier,training_eligible,holdout_eligible FROM records WHERE source='reconstructed'"
).fetchone()
self.assertEqual(reconstructed, ("reconstructed_verified", 1, 0))
```

- [ ] **Step 2: Run the registry test and confirm failure**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_developability_registry -v`

Expected: SQLite reports missing evidence columns or the reconstructed format is unsupported.

- [ ] **Step 3: Extend the schema and row routing**

Direct supervised rows receive `direct,1,1`; unlabeled rows receive
`unlabeled,0,0`; reconstructed rows call `reconstruct_verified_pair` and receive
`reconstructed_verified,1,0`. Add only existing reconstructed files in
`default_inputs()`:

```python
for property_name in PROPERTY_BY_FAMILY.values():
    path = root / f"reconstructed_{property_name}_dataset.csv"
    if path.exists():
        inputs.append(RegistryInput(path, property_name, "reconstructed_sequence_table"))
```

- [ ] **Step 4: Run the registry tests**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_developability_registry -v`

Expected: all registry tests pass, including deterministic rebuilds and training-only reconstruction.

- [ ] **Step 5: Commit**

```bash
git add data/developability_registry.py tests/test_developability_registry.py
git commit -m "feat: track evidence eligibility in registry"
```

### Task 3: Enforce training and holdout eligibility

**Files:**
- Modify: `model/developability_matrix.py`
- Modify: `tests/test_developability_matrix.py`

**Interfaces:**
- Consumes: registry eligibility columns.
- Produces: training sets containing only `training_eligible=1` and held-out sets containing only `holdout_eligible=1`.

- [ ] **Step 1: Add a failing pure filter test**

```python
rows = [
    {"source": "held", "training_eligible": 1, "holdout_eligible": 1},
    {"source": "held", "training_eligible": 1, "holdout_eligible": 0},
    {"source": "train", "training_eligible": 0, "holdout_eligible": 0},
]
train, test = eligible_split(rows, "held")
self.assertEqual(len(train), 0)
self.assertEqual(len(test), 1)
```

- [ ] **Step 2: Run and confirm failure**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_developability_matrix -v`

Expected: `eligible_split` is not defined.

- [ ] **Step 3: Add one shared eligibility helper and use it in the matrix**

```python
def eligible_split(rows: list[dict], held_source: str) -> tuple[list[dict], list[dict]]:
    train = [row for row in rows if row["source"] != held_source and row["training_eligible"]]
    test = [row for row in rows if row["source"] == held_source and row["holdout_eligible"]]
    return train, test
```

The existing supervision-tier and training-arm filters remain additional
constraints. All held-source formats remain in overlap stripping so a
training-only reconstruction cannot leak through a related direct holdout.

- [ ] **Step 4: Run matrix tests**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_developability_matrix -v`

Expected: all matrix tests pass.

- [ ] **Step 5: Commit**

```bash
git add model/developability_matrix.py tests/test_developability_matrix.py
git commit -m "feat: keep reconstructed rows out of holdouts"
```

### Task 4: Report and verify the live registry

**Files:**
- Modify: `tools/build_developability_registry.py`

**Interfaces:**
- Consumes: registry report evidence and eligibility counts.
- Produces: beginner-readable counts in the existing CLI output.

- [ ] **Step 1: Print the new audit counts**

```python
print(f"  evidence tiers: {report['evidence_counts']}")
print(f"  supervised training eligible: {report['training_eligible_rows']}")
print(f"  holdout eligible: {report['holdout_eligible_rows']}")
```

- [ ] **Step 2: Run focused and full verification**

Run:

```bash
~/Biostuff/.venv/bin/python -m unittest tests.test_sequence_reconstruction tests.test_developability_registry tests.test_developability_matrix -v
~/Biostuff/.venv/bin/python -m tools.build_developability_registry
~/Biostuff/.venv/bin/python -m unittest discover -s tests
```

Expected: focused tests and full suite pass; the live direct dataset reports
zero reconstructed rows until a verified reconstruction CSV is present.

- [ ] **Step 3: Commit**

```bash
git add tools/build_developability_registry.py
git commit -m "feat: report recoverable evidence tiers"
```
