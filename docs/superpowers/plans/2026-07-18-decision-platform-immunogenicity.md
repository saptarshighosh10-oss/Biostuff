# Decision Platform and Immunogenicity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-serving decision-platform contract and an honestly evaluated sequence-humanness immunogenicity development head without changing the legacy developability platform.

**Architecture:** A small stdlib contract reports model maturity separately from prediction availability. Existing OAS parsing is extended only enough to support unit-balanced paired human repertoires; a local k-mer likelihood model is trained without ADA labels and evaluated once on the known BioPhi development cohort. IDC remains sealed and no new production artifact is promoted.

**Tech Stack:** Python 3.14, stdlib, existing scikit-learn, existing `data.contract`, `data.run_manifest`, `data.oas`, and `model.metrics`.

## Global Constraints

- Do not modify `model/head_platform.py` or `model/saved/platform.pkl`.
- Do not inspect, count, load, train on, tune on, or download GDPa3.
- Do not read IDC record-level ADA labels before the development model, filters, and hashes are frozen.
- Do not add a production dependency.
- Keep downloaded source files under ignored `data/external/public_sources/`.
- Use only standard 20-amino-acid VH/VL sequences of length at least 20.
- Never convert a missing input or score into zero.
- Report the all-molecule and humanized-only BioPhi results separately.
- Keep `intrinsic_immunogenicity` separate from contextual clinical ADA.
- Do not promote or serve the head without independent confirmation.

---

### Task 1: Shared decision-platform contract

**Files:**
- Create: `model/decision_platform.py`
- Test: `tests/test_decision_platform.py`

**Interfaces:**
- Consumes: `data.contract.canonical_sequence(value: Any) -> str`
- Produces: `DecisionRequest`, `head_result(name: str, model_status: str, required: tuple[str, ...], request: DecisionRequest, *, limitations: list[str]) -> dict[str, Any]`, `evaluate_request(request: DecisionRequest) -> dict[str, Any]`

- [ ] **Step 1: Write the failing contract tests**

```python
import unittest

from model.decision_platform import DecisionRequest, evaluate_request


VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS"


class TestDecisionPlatform(unittest.TestCase):
    def test_missing_head_inputs_abstain_without_fake_score(self):
        report = evaluate_request(DecisionRequest(vh=VH))
        self.assertEqual(set(report["heads"]), {
            "developability", "intrinsic_immunogenicity", "contextual_ada",
            "affinity", "off_target_panel", "off_target_pairwise", "pk", "clinical_evidence",
        })
        affinity = report["heads"]["affinity"]
        self.assertEqual(affinity["prediction_status"], "abstain")
        self.assertIn("antigen_sequence", affinity["missing_inputs"])
        self.assertIsNone(affinity["score"])

    def test_invalid_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "standard 20-amino-acid"):
            DecisionRequest(vh="NOT_A_SEQUENCE")
```

- [ ] **Step 2: Run the focused test and confirm the missing module failure**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_decision_platform -v`

Expected: `ModuleNotFoundError: No module named 'model.decision_platform'`.

- [ ] **Step 3: Implement the minimum explicit contract**

```python
from dataclasses import asdict, dataclass
from typing import Any

from data.contract import ContractError, canonical_sequence


@dataclass(frozen=True)
class DecisionRequest:
    vh: str
    vl: str = ""
    antigen_sequence: str = ""
    antigen_id: str = ""
    fc_sequence: str = ""
    isotype: str = ""
    molecule_format: str = ""
    target: str = ""
    indication: str = ""
    dose: str = ""
    route: str = ""
    population: str = ""

    def __post_init__(self) -> None:
        for field in ("vh", "vl", "antigen_sequence", "fc_sequence"):
            raw = getattr(self, field)
            if not raw and field != "vh":
                continue
            try:
                value = canonical_sequence(raw)
            except ContractError as exc:
                raise ValueError(
                    f"{field} must be a standard 20-amino-acid sequence of length >= 20"
                ) from exc
            if len(value) < 20:
                raise ValueError(
                    f"{field} must be a standard 20-amino-acid sequence of length >= 20"
                )
            object.__setattr__(self, field, value)


def head_result(name: str, model_status: str, required: tuple[str, ...], request: DecisionRequest, *, limitations: list[str]) -> dict[str, Any]:
    missing = [field for field in required if not getattr(request, field)]
    return {
        "name": name,
        "model_status": model_status,
        "prediction_status": "abstain" if missing or model_status != "served" else "predicted",
        "score": None,
        "score_direction": None,
        "required_inputs": list(required),
        "missing_inputs": missing,
        "inputs_used": [],
        "uncertainty": None,
        "applicability": "not_evaluated",
        "benchmark": None,
        "model_version": None,
        "data_cutoff": None,
        "limitations": limitations,
    }


def evaluate_request(request: DecisionRequest) -> dict[str, Any]:
    specs = {
        "developability": ("experimental", ("vh",), ["legacy platform is intentionally not wrapped here"]),
        "intrinsic_immunogenicity": ("experimental", ("vh", "vl"), ["humanness is not clinical ADA probability"]),
        "contextual_ada": ("unavailable", ("vh", "vl", "indication", "dose", "route", "population"), ["requires locked clinical confirmation"]),
        "affinity": ("unavailable", ("vh", "vl", "antigen_sequence"), ["antigen-aware benchmark not yet passed"]),
        "off_target_panel": ("experimental", ("vh", "vl"), ["single REAP source; not clinical toxicity"]),
        "off_target_pairwise": ("unavailable", ("vh", "vl", "antigen_sequence"), ["complete pairwise labels unavailable"]),
        "pk": ("unavailable", ("vh", "vl", "fc_sequence", "molecule_format", "target", "dose", "route", "indication"), ["VH/VL alone cannot determine clinical half-life"]),
        "clinical_evidence": ("unavailable", ("target", "indication"), ["evidence panel, not a success prediction"]),
    }
    return {
        "schema_version": "1",
        "request": asdict(request),
        "heads": {
            name: head_result(name, status, required, request, limitations=limits)
            for name, (status, required, limits) in specs.items()
        },
    }
```

- [ ] **Step 4: Run the contract test**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_decision_platform -v`

Expected: two tests pass.

- [ ] **Step 5: Commit only the contract slice**

```bash
git add model/decision_platform.py tests/test_decision_platform.py
git commit -m "feat: add non-serving antibody decision contract"
```

---

### Task 2: BioPhi fetch, parser, and immutable audit

**Files:**
- Create: `data/immunogenicity.py`
- Create: `tools/fetch_biophi_immunogenicity.py`
- Test: `tests/test_immunogenicity.py`

**Interfaces:**
- Consumes: `data.contract.canonical_sequence`, `data.run_manifest.file_sha256`
- Produces: `load_biophi(root: Path) -> tuple[list[dict], dict]`
- Produces: local ignored files `immunogenicity.tsv`, `mabs.fa`, and `manifest.json`

- [ ] **Step 1: Write parser tests for real schemas and molecule conflicts**

```python
import tempfile
import unittest
from pathlib import Path

from data.immunogenicity import load_biophi


VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQSISNYLNWYQQKPGKAPKLLIY"


class TestBioPhi(unittest.TestCase):
    def test_loads_pairs_and_retains_duplicate_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "immunogenicity.tsv").write_text(
                "Name\tImmunogenicity\tSpecies\tReference\nA\t12.0\tHumanized\tr1\nB\t2.0\tHumanized\tr2\n",
                encoding="utf-8",
            )
            (root / "mabs.fa").write_text(
                f">A VH\n{VH}\n>A VL\n{VL}\n>B VH\n{VH}\n>B VL\n{VL}\n",
                encoding="utf-8",
            )
            rows, audit = load_biophi(root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(audit["unique_molecules"], 1)
        self.assertEqual(audit["duplicate_molecule_groups"], 1)
        self.assertEqual({row["ada_percent"] for row in rows}, {2.0, 12.0})

    def test_missing_chain_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "immunogenicity.tsv").write_text(
                "Name\tImmunogenicity\tSpecies\tReference\nA\t12.0\tHumanized\tr1\n",
                encoding="utf-8",
            )
            (root / "mabs.fa").write_text(f">A VH\n{VH}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete VH/VL"):
                load_biophi(root)
```

- [ ] **Step 2: Run the parser tests and confirm the missing module failure**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_immunogenicity -v`

Expected: `ModuleNotFoundError: No module named 'data.immunogenicity'`.

- [ ] **Step 3: Implement strict TSV/FASTA parsing and audit counts**

```python
import csv
import math
from collections import Counter
from pathlib import Path

from data.contract import canonical_sequence, hash_payload


def _read_labels(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {"Name", "Immunogenicity", "Species", "Reference"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"unexpected BioPhi label columns: {reader.fieldnames}")
        rows = []
        for line, record in enumerate(reader, 2):
            try:
                value = float(record["Immunogenicity"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric ADA value at line {line}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite ADA value at line {line}")
            rows.append({
                "name": record["Name"].strip(),
                "ada_percent": value,
                "species": record["Species"].strip(),
                "reference": record["Reference"].strip(),
            })
    return rows


def _read_fasta(path: Path) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    current: tuple[str, str] | None = None
    chunks: list[str] = []

    def finish() -> None:
        if current is None:
            return
        name, chain = current
        bucket = pairs.setdefault(name, {})
        if chain in bucket:
            raise ValueError(f"duplicate FASTA header for {name} {chain}")
        bucket[chain] = "".join(chunks)

    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(">"):
            finish()
            header = raw[1:].rsplit(" ", 1)
            if len(header) != 2 or header[1] not in {"VH", "VL"}:
                raise ValueError(f"unexpected BioPhi FASTA header: {raw}")
            current = (header[0], header[1])
            chunks = []
        elif raw.strip():
            if current is None:
                raise ValueError("FASTA sequence appeared before a header")
            chunks.append(raw.strip())
    finish()
    return pairs


def load_biophi(root: Path) -> tuple[list[dict], dict]:
    labels = _read_labels(root / "immunogenicity.tsv")
    sequences = _read_fasta(root / "mabs.fa")
    rows = []
    for item in labels:
        pair = sequences.get(item["name"], {})
        if set(pair) != {"VH", "VL"}:
            raise ValueError(f"{item['name']} does not have a complete VH/VL pair")
        vh = canonical_sequence(pair["VH"])
        vl = canonical_sequence(pair["VL"])
        if min(len(vh), len(vl)) < 20:
            raise ValueError(f"{item['name']} has a chain shorter than 20 residues")
        rows.append({**item, "vh": vh, "vl": vl, "molecule_id": hash_payload({"vh": vh, "vl": vl})})
    groups = Counter(row["molecule_id"] for row in rows)
    return rows, {
        "raw_label_rows": len(labels),
        "paired_rows": len(rows),
        "unique_molecules": len(groups),
        "duplicate_molecule_groups": sum(count > 1 for count in groups.values()),
        "high_ada_rows": sum(row["ada_percent"] >= 10.0 for row in rows),
        "low_ada_rows": sum(row["ada_percent"] < 10.0 for row in rows),
    }
```

- [ ] **Step 4: Implement the pinned stdlib downloader**

Use commit `d6323cee83ebd544026c03da87d9805dbaf0ba7d` and URLs rooted at:

```text
https://raw.githubusercontent.com/Merck/BioPhi-2021-publication/d6323cee83ebd544026c03da87d9805dbaf0ba7d/data/tasks/immunogenicity/
```

Download only `immunogenicity.tsv` and `mabs.fa` with `urllib.request.urlopen`, write through `tempfile.NamedTemporaryFile` followed by `os.replace`, then write `manifest.json` containing retrieval time, commit, source DOI, MIT license URL, byte counts, and `file_sha256` values. Existing files whose hashes match the manifest are reused; a mismatch fails closed.

- [ ] **Step 5: Run the parser tests and fetch the real files**

Run:

```bash
~/Biostuff/.venv/bin/python -m unittest tests.test_immunogenicity -v
~/Biostuff/.venv/bin/python -m tools.fetch_biophi_immunogenicity
```

Expected real audit after parsing: 217 paired rows, 212 unique VH/VL molecules, five duplicate molecule groups, 53 rows at or above 10% ADA, and 164 below.

- [ ] **Step 6: Commit source code and tests, never downloaded files**

```bash
git add data/immunogenicity.py tools/fetch_biophi_immunogenicity.py tests/test_immunogenicity.py
git commit -m "feat: audit BioPhi immunogenicity benchmark"
```

---

### Task 3: Reproducible paired-OAS acquisition and balanced loading

**Files:**
- Modify: `data/oas.py`
- Create: `tools/fetch_oas_paired.py`
- Modify: `tests/test_oas.py`

**Interfaces:**
- Consumes: official POST endpoint `https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/`
- Produces: `load_oas(..., max_sequences_per_file: int | None = None) -> list[dict]`
- Produces: ignored OAS files and `manifest.json`

- [ ] **Step 1: Add a failing balance-and-pair test**

```python
import gzip
import json


VL1 = "DIQMTQSPSSLSASVGDRVTITCRASQSISNYLNWYQQKPGKAPKLLIY"
VL2 = "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIY"
VL3 = "QSVLTQPPSASGTPGQRVTISCSGSSSNIGNNYVSWYQQLPGTAPKLLIY"
VL4 = "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQ"
VH4 = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMG"


def _write_paired_oas_gz(path: Path, pairs: list[tuple[str, str]]) -> None:
    metadata = json.dumps({"Author": path.stem, "Species": "human", "Subject": path.stem})
    lines = [metadata, "sequence_alignment_aa_heavy,sequence_alignment_aa_light"]
    lines.extend(f"{vh},{vl}" for vh, vl in pairs)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def test_paired_units_can_be_capped_per_file(self) -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_paired_oas_gz(root / "a.csv.gz", [(VH1, VL1), (VH2, VL2)])
        _write_paired_oas_gz(root / "b.csv.gz", [(VH3, VL3), (VH4, VL4)])
        rows = load_oas(root, max_sequences=10, max_sequences_per_file=1)
    self.assertEqual(len(rows), 2)
    self.assertTrue(all(row["vh_sequence"] and row["vl_sequence"] for row in rows))
    self.assertEqual(len({row["unit_id"] for row in rows}), 2)
```

- [ ] **Step 2: Run the focused test and verify the signature failure**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_oas.TestOas.test_paired_units_can_be_capped_per_file -v`

Expected: failure because `max_sequences_per_file` is not accepted.

- [ ] **Step 3: Extend the existing loader, without another parser**

Add optional `max_sequences_per_file: int | None = None`; track `file_rows` inside each file loop, stop that file at the cap, and emit:

```python
"unit_id": path.name,
"subject_id": str(metadata.get("Subject") or metadata.get("subject") or "").strip() or None,
```

Keep every existing output field and default behavior unchanged.

- [ ] **Step 4: Implement manifest-first OAS downloading**

`tools.fetch_oas_paired` must:

1. POST `Species=human`, `Disease=None`, `Vaccine=None`, and `*` for the remaining form fields.
2. Extract only quoted `wget https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/...csv.gz` URLs.
3. Reject a URL with any other scheme, host, suffix, or path prefix.
4. Write `manifest.json` before data downloads with query fields, retrieval time, sorted URLs, and status `planned`.
5. Download atomically, record byte size and SHA-256 per file, and finish with status `complete`.
6. Support `--manifest-only` and `--max-units N`; default downloads all returned units.

- [ ] **Step 5: Run tests, preview the manifest, then download**

Run:

```bash
~/Biostuff/.venv/bin/python -m unittest tests.test_oas -v
~/Biostuff/.venv/bin/python -m tools.fetch_oas_paired --manifest-only
~/Biostuff/.venv/bin/python -m tools.fetch_oas_paired
```

Expected manifest preview: 91 URLs for the 2026-07-18 no-disease/no-vaccine query. The final sequence count is measured from downloaded files rather than assumed from the web summary.

- [ ] **Step 6: Commit the acquisition slice**

```bash
git add data/oas.py tools/fetch_oas_paired.py tests/test_oas.py
git commit -m "feat: fetch balanced paired OAS units"
```

---

### Task 4: One-class k-mer humanness model and honest BioPhi evaluation

**Files:**
- Create: `model/immunogenicity_humanness.py`
- Create: `tools/eval_immunogenicity.py`
- Test: `tests/test_immunogenicity_humanness.py`

**Interfaces:**
- Consumes: paired OAS rows and `load_biophi(...)`
- Produces: `KmerHumannessModel.fit(rows)`, `.humanness(vh, vl)`, and an evaluation JSON

- [ ] **Step 1: Write failing likelihood and subgroup tests**

```python
import unittest

from model.immunogenicity_humanness import KmerHumannessModel, evaluate_biophi


class TestHumanness(unittest.TestCase):
    def test_seen_kmers_score_more_human(self):
        model = KmerHumannessModel(k=3).fit([
            {"vh_sequence": "A" * 25, "vl_sequence": "C" * 25},
            {"vh_sequence": "A" * 24 + "C", "vl_sequence": "C" * 24 + "A"},
        ])
        self.assertGreater(model.humanness("A" * 25, "C" * 25), model.humanness("W" * 25, "Y" * 25))

    def test_evaluation_reports_humanized_separately(self):
        rows = [
            {"name": "a", "molecule_id": "1", "vh": "A" * 25, "vl": "C" * 25, "ada_percent": 12.0, "species": "Humanized", "reference": "r1"},
            {"name": "b", "molecule_id": "2", "vh": "W" * 25, "vl": "Y" * 25, "ada_percent": 2.0, "species": "Humanized", "reference": "r2"},
        ]
        oas = [
            {"vh_sequence": "A" * 25, "vl_sequence": "C" * 25},
            {"vh_sequence": "A" * 24 + "C", "vl_sequence": "C" * 24 + "A"},
        ]
        report = evaluate_biophi(KmerHumannessModel(k=2).fit(oas), rows, bootstrap_draws=20)
        self.assertEqual(report["subgroups"]["humanized"]["n"], 2)
        self.assertEqual(report["threshold_percent"], 10.0)
```

- [ ] **Step 2: Run tests and confirm the missing module failure**

Run: `~/Biostuff/.venv/bin/python -m unittest tests.test_immunogenicity_humanness -v`

Expected: `ModuleNotFoundError: No module named 'model.immunogenicity_humanness'`.

- [ ] **Step 3: Implement the minimum one-class model**

`KmerHumannessModel(k=4, alpha=1.0)` stores separate VH and VL `Counter[str]` objects. `fit` counts overlapping k-mers from OAS only. `humanness` returns the mean smoothed log-probability across both chains:

```python
log((count[kmer] + alpha) / (total + alpha * 20**k))
```

`risk(vh, vl)` returns `-humanness(vh, vl)`. Reject unpaired inputs and fewer than 40 paired training rows in the CLI. ADA labels never enter `fit`.

- [ ] **Step 4: Implement the fixed development evaluation**

For all unique molecule predictions and the all, human+humanized, humanized, and human subgroups, report:

- `n`, positive count at ADA >=10%;
- ROC AUC when both classes exist;
- Spearman risk versus ADA percent;
- reference-cluster bootstrap 95% intervals;
- explicit warning when either class has fewer than five molecules.

Also report an origin-only baseline. Duplicate clinical observations share one sequence score and one total molecule weight; add a median-collapsed sensitivity view.

- [ ] **Step 5: Add the beginner-facing CLI and freeze evidence**

Run:

```bash
~/Biostuff/.venv/bin/python -m tools.eval_immunogenicity \
  --oas data/external/public_sources/oas_paired \
  --biophi data/external/public_sources/biophi_immunogenicity \
  --max-per-unit 1000 \
  --out results/immunogenicity_biophi_development.json
```

The JSON must include source hashes, exact raw/rejected/deduplicated counts, k, alpha, OAS unit count, OAS paired training count, subgroup metrics, and limitations. Print the exact overall and humanized-only AUCs. If the OAS download or either BioPhi file is missing, print the exact error and stop.

- [ ] **Step 6: Run tests and the real development evaluation**

Run:

```bash
~/Biostuff/.venv/bin/python -m unittest \
  tests.test_decision_platform \
  tests.test_immunogenicity \
  tests.test_oas \
  tests.test_immunogenicity_humanness -v
~/Biostuff/.venv/bin/python -m tools.eval_immunogenicity
```

Expected: tests pass; the real AUC is reported without a target value assumption. A null or confounded result remains experimental.

- [ ] **Step 7: Commit code and the ignored evidence explicitly**

```bash
git add model/immunogenicity_humanness.py tools/eval_immunogenicity.py tests/test_immunogenicity_humanness.py
git add -f results/immunogenicity_biophi_development.json
git commit -m "results: evaluate intrinsic immunogenicity baseline"
```

---

### Task 5: Freeze the IDC confirmation protocol without opening labels

**Files:**
- Create: `tools/immunogenicity_confirmation_policy.json`
- Create: `tools/freeze_immunogenicity_confirmation.py`
- Test: `tests/test_immunogenicity_confirmation.py`

**Interfaces:**
- Consumes: the development report and model configuration from Task 4
- Produces: a checksum-locked protocol; does not download IDC

- [ ] **Step 1: Write the failing freeze test**

```python
import json
import tempfile
from pathlib import Path

from tools.freeze_immunogenicity_confirmation import freeze_protocol


def test_freeze_records_development_hash_without_idc_labels(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        development = root / "development.json"
        development.write_text('{"model": {"k": 4, "alpha": 1.0}}\n')
        output = root / "freeze.json"
        freeze_protocol(development, output)
        payload = json.loads(output.read_text())
        self.assertEqual(payload["status"], "frozen_before_idc_labels")
        self.assertEqual(len(payload["development_sha256"]), 64)
        self.assertNotIn("ada", json.dumps(payload).lower())
```

- [ ] **Step 2: Implement a checksum-only freeze command**

The frozen JSON contains schema version, UTC freeze time, development report path and SHA-256, model config, exact BioPhi exclusion rule, IDC source URL and expected CC-BY-4.0 license, minimum 30 unique non-overlapping molecules, minimum five per class, one-shot ledger path, and the statement that no IDC file was downloaded by this command.

- [ ] **Step 3: Run the test and freeze the protocol**

Run:

```bash
~/Biostuff/.venv/bin/python -m unittest tests.test_immunogenicity_confirmation -v
~/Biostuff/.venv/bin/python -m tools.freeze_immunogenicity_confirmation
```

Expected: no file appears under an IDC external-data directory.

- [ ] **Step 4: Commit the sealed protocol**

```bash
git add tools/immunogenicity_confirmation_policy.json tools/freeze_immunogenicity_confirmation.py tests/test_immunogenicity_confirmation.py
git commit -m "chore: freeze IDC immunogenicity confirmation protocol"
```

---

### Task 6: Final verification and handoff

**Files:**
- No file changes.

**Interfaces:**
- Consumes: all prior task outputs
- Produces: no production artifact

- [ ] **Step 1: Run the focused suite**

```bash
~/Biostuff/.venv/bin/python -m unittest \
  tests.test_decision_platform \
  tests.test_immunogenicity \
  tests.test_oas \
  tests.test_immunogenicity_humanness \
  tests.test_immunogenicity_confirmation -v
```

Expected: all tests pass.

- [ ] **Step 2: Prove production files and GDPa3 were untouched**

```bash
git diff --exit-code HEAD~5 -- model/head_platform.py model/saved/platform.pkl
git status --short
```

Expected: no diff for either production path; only pre-existing unrelated dirty files remain.

- [ ] **Step 3: Report exact data and benchmark facts**

Report the downloaded OAS unit and paired-row counts, BioPhi 217/212 counts, overall and humanized-only AUC with uncertainty, whether the result remains experimental, and that IDC labels were not opened. Do not claim clinical ADA prediction.
