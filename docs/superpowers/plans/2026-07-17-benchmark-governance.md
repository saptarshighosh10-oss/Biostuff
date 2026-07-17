# Benchmark Governance Phase 0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task with review checkpoints.

**Goal:** Freeze the current antibody-model evidence and add one honest status command that reports benchmark roles, traceable baseline values, independent row counts, metric definitions, and every future holdout use without retraining a model.

**Architecture:** A versioned JSON policy is the human-reviewable source of truth for each property's development, locked-validation, and final sources. A dependency-free Python module validates that policy, resolves baseline values from existing JSON evidence, hashes frozen files, counts rows without loading the 118 MB polyreactivity CSV into memory, and appends future holdout uses to a JSONL ledger. A thin beginner-friendly CLI prints the report and writes the machine-readable snapshot. Nothing in this phase imports a training head or reads GDPa3 data.

**Tech stack:** Python 3 standard library, existing `data.run_manifest.file_sha256`, `unittest`, JSON/JSONL, CSV streaming.

**Global constraints:** Preserve unrelated dirty work; do not add dependencies; do not retrain models; do not modify a served artifact; do not inspect, download, or count GDPa3; never infer a missing metric; report a missing file or JSON field as missing; keep all existing historical AUCs labeled as historical rather than silently replacing them.

---

## Contract locked by this plan

The policy must make these distinctions explicit:

| Property | Previously used development evidence | Locked validation | Unspent final source |
|---|---|---|---|
| Polyreactivity | `aims_mouse`, `aims_flu`, `shehata2019` | none assigned yet | `gdpa3` |
| Aggregation | `gdpa1`, `gdpa2`, `gdpa4.1`, `jain2017biophysical`, `jain2024assessment`, `dotad_jain_et_al` | none assigned yet | `gdpa3` |
| Hydrophobicity | `shehata2019`, plus the training sources named in `results/hic_platform_refresh.json` | none assigned yet | `gdpa3` |

The command reports two polyreactivity numbers side by side, never as interchangeable scores:

- historical platform scorecard: `results/platform_scorecard.json#/served_heads/polyreactivity/honest_auc` (`0.70` in the frozen evidence);
- hard library-to-diverse transfer: `results/gate_cdr_polyreactivity.json#/baseline_mean_auc` (`0.6269261074768057` in the frozen evidence).

Likewise, aggregation's `0.72` historical scorecard and the strict solid-study ensemble estimate at `results/head_b_acsins_ensemble_cdr_h3_plm.json#/baseline/mean_auc_solid_studies` are separately named. Hydrophobicity has only its traceable historical `0.70` baseline in current result JSON; the status command must say that no newer hard-transfer JSON evidence is registered instead of inventing one.

The lab-ready metric contract is source-macro ROC AUC plus per-source ROC AUC, PR AUC, top-10% and top-15% catch rate/enrichment, three seeds, source/sequence-cluster bootstrap confidence intervals, calibration, subgroup checks, and OOD/abstention behavior. Phase 0 documents these requirements; it does not compute absent metrics retroactively.

---

### Task 1: Add and validate the benchmark policy

**Files:**

- Create: `tools/benchmark_policy.json`
- Create: `model/benchmark_governance.py`
- Create: `tests/test_benchmark_governance.py`

**Step 1: Write the failing policy-validation tests**

Start `tests/test_benchmark_governance.py` with policy-focused tests. Use temporary files so tests cannot mutate the real policy or ledger.

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model.benchmark_governance import PolicyError, load_policy, validate_policy


def minimal_policy() -> dict:
    return {
        "schema_version": "1",
        "metric_definitions": {
            "primary": "macro_source_roc_auc",
            "required": ["source_roc_auc", "source_pr_auc"],
        },
        "properties": {
            name: {
                "source_roles": {
                    "development": [{"name": f"{name}_dev", "prior_use": True}],
                    "locked_validation": [],
                    "final": [{"name": "gdpa3", "prior_use": False}],
                },
                "baselines": [],
                "count_evidence": [],
            }
            for name in ("polyreactivity", "aggregation", "hydrophobicity")
        },
        "frozen_artifacts": [],
    }


class TestPolicyValidation(unittest.TestCase):
    def test_project_policy_is_valid_and_gdpa3_is_unspent_final_only(self) -> None:
        policy = load_policy(Path("tools/benchmark_policy.json"))
        validate_policy(policy)
        for property_policy in policy["properties"].values():
            final = property_policy["source_roles"]["final"]
            self.assertEqual(final, [{"name": "gdpa3", "prior_use": False}])
            self.assertNotIn(
                "gdpa3",
                {
                    source["name"]
                    for role in ("development", "locked_validation")
                    for source in property_policy["source_roles"][role]
                },
            )

    def test_rejects_a_source_in_multiple_roles(self) -> None:
        policy = minimal_policy()
        policy["properties"]["polyreactivity"]["source_roles"]["locked_validation"] = [
            {"name": "polyreactivity_dev", "prior_use": False}
        ]
        with self.assertRaisesRegex(PolicyError, "multiple roles"):
            validate_policy(policy)

    def test_rejects_a_final_source_marked_as_previously_used(self) -> None:
        policy = minimal_policy()
        policy["properties"]["polyreactivity"]["source_roles"]["final"][0]["prior_use"] = True
        with self.assertRaisesRegex(PolicyError, "final.*unspent"):
            validate_policy(policy)

    def test_load_policy_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(PolicyError, "valid JSON"):
                load_policy(path)
```

**Step 2: Run the tests and verify RED**

Run:

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance.TestPolicyValidation
```

Expected: `ImportError` or `ModuleNotFoundError` because `model.benchmark_governance` does not exist yet.

**Step 3: Add the smallest validation implementation**

Start `model/benchmark_governance.py` with these public interfaces:

```python
"""Traceable benchmark policy, evidence snapshots, and holdout-use records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1"
PROPERTIES = ("polyreactivity", "aggregation", "hydrophobicity")
ROLES = ("development", "locked_validation", "final")


class PolicyError(ValueError):
    """Raised when benchmark governance would be ambiguous or unsafe."""


def load_policy(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyError(f"policy file not found: {candidate}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"policy is not valid JSON: {candidate}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PolicyError("policy root must be a JSON object")
    return payload


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise PolicyError(f"unsupported schema_version: {policy.get('schema_version')!r}")
    if set(policy.get("properties", {})) != set(PROPERTIES):
        raise PolicyError(f"properties must be exactly: {', '.join(PROPERTIES)}")
    if not isinstance(policy.get("metric_definitions"), dict):
        raise PolicyError("metric_definitions must be an object")
    if not isinstance(policy.get("frozen_artifacts"), list):
        raise PolicyError("frozen_artifacts must be a list")

    for property_name, property_policy in policy["properties"].items():
        roles = property_policy.get("source_roles", {})
        if set(roles) != set(ROLES):
            raise PolicyError(f"{property_name}: source_roles must be exactly {ROLES}")
        seen: set[str] = set()
        for role in ROLES:
            entries = roles[role]
            if not isinstance(entries, list):
                raise PolicyError(f"{property_name}.{role} must be a list")
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    raise PolicyError(f"{property_name}.{role} has an invalid source entry")
                if not isinstance(entry.get("prior_use"), bool):
                    raise PolicyError(f"{property_name}.{role}.{entry.get('name')} needs boolean prior_use")
                source = entry["name"]
                if source in seen:
                    raise PolicyError(f"{property_name}: {source} appears in multiple roles")
                seen.add(source)
                if role == "final" and entry["prior_use"]:
                    raise PolicyError(f"{property_name}: final source {source} must remain unspent")
        if not isinstance(property_policy.get("baselines"), list):
            raise PolicyError(f"{property_name}.baselines must be a list")
        if not isinstance(property_policy.get("count_evidence"), list):
            raise PolicyError(f"{property_name}.count_evidence must be a list")
```

**Step 4: Add the real policy as data, not Python constants**

Create `tools/benchmark_policy.json` with schema version `1`. Use the source roles in the contract table above and these exact evidence registrations:

```json
{
  "schema_version": "1",
  "metric_definitions": {
    "primary": "macro_source_roc_auc",
    "primary_definition": "Arithmetic mean of ROC AUC computed separately for each assigned held-out source; never a row-weighted pooled mean.",
    "required": [
      "source_roc_auc",
      "source_pr_auc",
      "catch_rate_at_10pct",
      "catch_rate_at_15pct",
      "enrichment_at_10pct",
      "enrichment_at_15pct",
      "three_seed_summary",
      "source_cluster_bootstrap_95pct_ci",
      "calibration",
      "subgroup_checks",
      "ood_abstention"
    ],
    "positive_direction": "Higher endpoint value is worse for polyreactivity, aggregation/self-association, and HIC retention.",
    "locked_validation_rule": "Assign a genuinely new diverse source before labels are used and freeze predictions before reading labels.",
    "final_rule": "GDPa3 is single-shot final evidence and may not be inspected, downloaded, counted, trained on, or tuned against during development."
  },
  "properties": {
    "polyreactivity": {
      "source_roles": {
        "development": [
          {"name": "aims_mouse", "prior_use": true},
          {"name": "aims_flu", "prior_use": true},
          {"name": "shehata2019", "prior_use": true}
        ],
        "locked_validation": [],
        "final": [{"name": "gdpa3", "prior_use": false}]
      },
      "baselines": [
        {
          "name": "historical_platform_scorecard",
          "metric": "historical_honest_auc",
          "path": "results/platform_scorecard.json",
          "json_pointer": "/served_heads/polyreactivity/honest_auc",
          "scope": "Historical platform estimate; not the hard library-to-diverse transfer benchmark."
        },
        {
          "name": "hard_library_to_diverse_transfer",
          "metric": "macro_source_roc_auc",
          "path": "results/gate_cdr_polyreactivity.json",
          "json_pointer": "/baseline_mean_auc",
          "scope": "Library-trained model tested separately on aims_mouse, aims_flu, and shehata2019 after held-out VH removal."
        }
      ],
      "count_evidence": [
        {
          "name": "dataset_rows_by_source",
          "kind": "csv_group_count",
          "path": "~/Downloads/antibody_sources/polyreactivity_dataset.csv",
          "column": "source"
        },
        {
          "name": "hard_transfer_test_rows",
          "kind": "object_field_map",
          "path": "results/gate_cdr_polyreactivity.json",
          "json_pointer": "/baseline",
          "field": "test_n"
        }
      ]
    },
    "aggregation": {
      "source_roles": {
        "development": [
          {"name": "gdpa1", "prior_use": true},
          {"name": "gdpa2", "prior_use": true},
          {"name": "gdpa4.1", "prior_use": true},
          {"name": "jain2017biophysical", "prior_use": true},
          {"name": "jain2024assessment", "prior_use": true},
          {"name": "dotad_jain_et_al", "prior_use": true}
        ],
        "locked_validation": [],
        "final": [{"name": "gdpa3", "prior_use": false}]
      },
      "baselines": [
        {
          "name": "historical_platform_scorecard",
          "metric": "historical_honest_auc",
          "path": "results/platform_scorecard.json",
          "json_pointer": "/served_heads/aggregation/honest_auc",
          "scope": "Historical platform estimate."
        },
        {
          "name": "strict_solid_study_ensemble",
          "metric": "mean_roc_auc_solid_studies",
          "path": "results/head_b_acsins_ensemble_cdr_h3_plm.json",
          "json_pointer": "/baseline/mean_auc_solid_studies",
          "scope": "Strict leakage-free baseline over solid AC-SINS development studies; still development evidence."
        }
      ],
      "count_evidence": [
        {
          "name": "evaluation_rows_by_assay",
          "kind": "json_records",
          "path": "results/head_b_acsins_ensemble_cdr_h3_plm.json",
          "json_pointer": "/baseline/per_assay",
          "fields": ["study", "assay", "n", "n_bad"]
        }
      ]
    },
    "hydrophobicity": {
      "source_roles": {
        "development": [
          {"name": "gdpa1", "prior_use": true},
          {"name": "gdpa2", "prior_use": true},
          {"name": "gdpa4.1", "prior_use": true},
          {"name": "jain2017_hic", "prior_use": true},
          {"name": "jetha2019_hic", "prior_use": true},
          {"name": "roche34_2022", "prior_use": true},
          {"name": "shanehsazzadeh2023_hic", "prior_use": true},
          {"name": "shehata2019", "prior_use": true}
        ],
        "locked_validation": [],
        "final": [{"name": "gdpa3", "prior_use": false}]
      },
      "baselines": [
        {
          "name": "historical_platform_scorecard",
          "metric": "historical_honest_auc",
          "path": "results/platform_scorecard.json",
          "json_pointer": "/served_heads/hydrophobicity/honest_auc",
          "scope": "Only registered traceable AUC JSON at Phase 0; no newer hard-transfer AUC is claimed."
        }
      ],
      "count_evidence": [
        {
          "name": "platform_training_rows_by_source",
          "kind": "json_value",
          "path": "results/hic_platform_refresh.json",
          "json_pointer": "/source_counts"
        },
        {
          "name": "standalone_hic_dataset_rows_by_source",
          "kind": "csv_group_count",
          "path": "~/Downloads/antibody_sources/hydrophobicity_dataset.csv",
          "column": "source"
        }
      ]
    }
  },
  "frozen_artifacts": [
    {"kind": "dataset", "path": "~/Downloads/antibody_sources/polyreactivity_dataset.csv"},
    {"kind": "dataset", "path": "~/Downloads/antibody_sources/hydrophobicity_dataset.csv"},
    {"kind": "dataset", "path": "data/external/gdpa/GDPa1_v1.2_20250814.csv"},
    {"kind": "dataset", "path": "data/external/gdpa/GDPa2.1.xlsx"},
    {"kind": "dataset", "path": "data/external/gdpa/GDPa4.1_20260616.xlsx"},
    {"kind": "serving_artifact", "path": "model/saved/platform.pkl"},
    {"kind": "feature_code", "path": "model/head_b_gbm.py"},
    {"kind": "feature_code", "path": "model/head_b_acsins_charge.py"},
    {"kind": "feature_code", "path": "model/cdr_features.py"},
    {"kind": "feature_code", "path": "model/head_platform.py"},
    {"kind": "benchmark_report", "path": "results/platform_scorecard.json"},
    {"kind": "benchmark_report", "path": "results/gate_cdr_polyreactivity.json"},
    {"kind": "benchmark_report", "path": "results/head_b_acsins_ensemble_cdr_h3_plm.json"},
    {"kind": "training_manifest", "path": "results/hic_platform_refresh.json"}
  ]
}
```

Do not add a GDPa3 file path to `frozen_artifacts`: the whole point is that the final source remains uninspected and uncounted.

**Step 5: Run the policy tests and verify GREEN**

Run the same focused command. Expected: four tests pass.

**Step 6: Commit the policy contract**

```bash
git add tools/benchmark_policy.json model/benchmark_governance.py tests/test_benchmark_governance.py
git commit -m "feat: lock benchmark source roles"
```

Checkpoint: inspect `git show --stat --oneline HEAD` and confirm no pre-existing dirty file entered the commit.

---

### Task 2: Resolve evidence, hashes, and independent counts without training

**Files:**

- Modify: `model/benchmark_governance.py`
- Modify: `tests/test_benchmark_governance.py`

**Step 1: Add failing evidence tests**

Extend the test module. Test actual behavior against small temporary CSV/JSON files, including missing evidence.

```python
from model.benchmark_governance import build_status, read_json_pointer


class TestEvidenceStatus(unittest.TestCase):
    def test_json_pointer_reads_nested_values_and_rejects_missing_fields(self) -> None:
        payload = {"a": {"b": [10, {"c": 0.625}]}}
        self.assertEqual(read_json_pointer(payload, "/a/b/1/c"), 0.625)
        with self.assertRaisesRegex(PolicyError, "JSON pointer"):
            read_json_pointer(payload, "/a/missing")

    def test_status_resolves_values_hashes_and_streamed_csv_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.json"
            rows = root / "rows.csv"
            artifact = root / "artifact.bin"
            evidence.write_text('{"score": 0.625, "counts": {"heldout": {"test_n": 97}}}\n')
            rows.write_text("source,label\na,0\na,1\nb,1\n", encoding="utf-8")
            artifact.write_bytes(b"frozen")
            policy = minimal_policy()
            prop = policy["properties"]["polyreactivity"]
            prop["baselines"] = [{
                "name": "hard_transfer",
                "metric": "macro_source_roc_auc",
                "path": "evidence.json",
                "json_pointer": "/score",
                "scope": "test",
            }]
            prop["count_evidence"] = [{
                "name": "rows",
                "kind": "csv_group_count",
                "path": "rows.csv",
                "column": "source",
            }]
            policy["frozen_artifacts"] = [{"kind": "test", "path": "artifact.bin"}]

            report = build_status(policy, repo_root=root, ledger_path=root / "uses.jsonl")

            baseline = report["properties"]["polyreactivity"]["baselines"][0]
            self.assertEqual(baseline["value"], 0.625)
            self.assertEqual(baseline["status"], "ok")
            self.assertEqual(
                report["properties"]["polyreactivity"]["counts"][0]["value"],
                {"a": 2, "b": 1},
            )
            self.assertEqual(report["frozen_artifacts"][0]["size_bytes"], 6)
            self.assertEqual(len(report["frozen_artifacts"][0]["sha256"]), 64)

    def test_missing_evidence_is_reported_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = minimal_policy()
            policy["properties"]["hydrophobicity"]["baselines"] = [{
                "name": "missing",
                "metric": "roc_auc",
                "path": "absent.json",
                "json_pointer": "/auc",
                "scope": "test",
            }]
            report = build_status(policy, repo_root=root, ledger_path=root / "uses.jsonl")
            item = report["properties"]["hydrophobicity"]["baselines"][0]
            self.assertEqual(item["status"], "missing")
            self.assertIsNone(item["value"])
            self.assertIn("absent.json", item["error"])
```

**Step 2: Run the evidence tests and verify RED**

Run:

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance.TestEvidenceStatus
```

Expected: import errors for the new functions.

**Step 3: Implement evidence resolution and counting**

Add these interfaces to `model/benchmark_governance.py`:

```python
import csv
import datetime as dt

from data.run_manifest import file_sha256


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(repo_root: str | Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    return candidate if candidate.is_absolute() else Path(repo_root) / candidate


def read_json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise PolicyError(f"invalid JSON pointer: {pointer!r}")
    current = payload
    try:
        for raw_part in pointer[1:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            current = current[int(part)] if isinstance(current, list) else current[part]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PolicyError(f"JSON pointer not found: {pointer}") from exc
    return current


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyError(f"evidence file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"evidence is not valid JSON: {path}: {exc}") from exc


def _csv_group_count(path: Path, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise PolicyError(f"CSV {path} is missing column {column!r}")
            for row in reader:
                value = (row.get(column) or "").strip()
                if value:
                    counts[value] = counts.get(value, 0) + 1
    except FileNotFoundError as exc:
        raise PolicyError(f"count evidence file not found: {path}") from exc
    return dict(sorted(counts.items()))
```

Implement an internal `_inspect_count_spec` supporting only the three policy kinds, with no implicit fallback:

```python
def _inspect_count_spec(spec: dict[str, Any], repo_root: Path) -> Any:
    path = resolve_path(repo_root, spec["path"])
    kind = spec["kind"]
    if kind == "csv_group_count":
        return _csv_group_count(path, spec["column"])
    payload = _load_json(path)
    value = read_json_pointer(payload, spec.get("json_pointer", ""))
    if kind == "json_value":
        return value
    if kind == "object_field_map":
        if not isinstance(value, dict):
            raise PolicyError(f"{spec['name']}: pointer must resolve to an object")
        field = spec["field"]
        return {name: record[field] for name, record in value.items()}
    if kind == "json_records":
        if not isinstance(value, list):
            raise PolicyError(f"{spec['name']}: pointer must resolve to a list")
        fields = spec["fields"]
        return [{field: record[field] for field in fields} for record in value]
    raise PolicyError(f"unsupported count evidence kind: {kind!r}")
```

Then implement:

```python
def build_status(
    policy: dict[str, Any],
    *,
    repo_root: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    validate_policy(policy)
    root = Path(repo_root).resolve()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "metric_definitions": policy["metric_definitions"],
        "properties": {},
        "frozen_artifacts": [],
        "holdout_uses": read_ledger(ledger_path),
    }
    for property_name, property_policy in policy["properties"].items():
        property_report = {
            "source_roles": property_policy["source_roles"],
            "baselines": [],
            "counts": [],
        }
        for baseline in property_policy["baselines"]:
            item = dict(baseline)
            path = resolve_path(root, baseline["path"])
            item["evidence_sha256"] = file_sha256(path)
            try:
                item["value"] = read_json_pointer(_load_json(path), baseline["json_pointer"])
                item["status"] = "ok"
            except PolicyError as exc:
                item.update({"value": None, "status": "missing", "error": str(exc)})
            property_report["baselines"].append(item)
        for count_spec in property_policy["count_evidence"]:
            item = {"name": count_spec["name"], "kind": count_spec["kind"]}
            try:
                item.update({"value": _inspect_count_spec(count_spec, root), "status": "ok"})
            except (KeyError, PolicyError) as exc:
                item.update({"value": None, "status": "missing", "error": str(exc)})
            property_report["counts"].append(item)
        report["properties"][property_name] = property_report
    for artifact in policy["frozen_artifacts"]:
        path = resolve_path(root, artifact["path"])
        report["frozen_artifacts"].append({
            **artifact,
            "resolved_path": str(path),
            "present": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": file_sha256(path),
        })
    return report
```

For the moment add a stub `read_ledger` that returns `[]` for a nonexistent file and parses each nonblank JSONL line for an existing file. Task 3 hardens and tests it.

**Step 4: Run tests and inspect the real report in-memory**

Run:

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance
/Users/saptarshighosh/Biostuff/.venv/bin/python - <<'PY'
from pathlib import Path
from model.benchmark_governance import build_status, load_policy

root = Path.cwd()
report = build_status(
    load_policy(root / "tools/benchmark_policy.json"),
    repo_root=root,
    ledger_path=root / "tools/holdout_use_ledger.jsonl",
)
for name, prop in report["properties"].items():
    print(name, [(item["name"], item["status"], item["value"]) for item in prop["baselines"]])
PY
```

Expected baseline values from the current evidence:

- polyreactivity: `0.7` historical and `0.6269261074768057` hard transfer;
- aggregation: `0.72` historical and `0.739` strict solid-study development estimate;
- hydrophobicity: `0.7` historical only.

If any value differs, preserve and report the real value. Do not edit the evidence to match this plan.

**Step 5: Commit the evidence reader**

```bash
git add model/benchmark_governance.py tests/test_benchmark_governance.py
git commit -m "feat: snapshot benchmark evidence"
```

---

### Task 3: Add the append-only holdout-use ledger with final-source guards

**Files:**

- Modify: `model/benchmark_governance.py`
- Modify: `tests/test_benchmark_governance.py`
- Create: `tools/holdout_use_ledger.jsonl`

`tools/holdout_use_ledger.jsonl` starts as an empty file. Historical use is represented honestly by `prior_use: true` in policy because exact first-use timestamps are not known. Do not fabricate backdated ledger entries.

**Step 1: Add failing ledger tests**

```python
from model.benchmark_governance import read_ledger, record_holdout_use


class TestHoldoutLedger(unittest.TestCase):
    def test_development_use_appends_a_traceable_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "uses.jsonl"
            evidence = root / "trial.json"
            evidence.write_text('{"auc": 0.63}\n', encoding="utf-8")
            record = record_holdout_use(
                minimal_policy(),
                ledger_path=ledger,
                property_name="polyreactivity",
                source="polyreactivity_dev",
                experiment_id="exp-001",
                purpose="fixed development comparison",
                evidence_path=evidence,
            )
            self.assertEqual(record["role"], "development")
            self.assertEqual(len(record["evidence_sha256"]), 64)
            self.assertEqual(read_ledger(ledger), [record])

    def test_locked_use_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = minimal_policy()
            policy["properties"]["polyreactivity"]["source_roles"]["locked_validation"] = [
                {"name": "new_locked", "prior_use": False}
            ]
            evidence = root / "trial.json"
            evidence.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PolicyError, "confirm_locked_use"):
                record_holdout_use(
                    policy,
                    ledger_path=root / "uses.jsonl",
                    property_name="polyreactivity",
                    source="new_locked",
                    experiment_id="exp-locked",
                    purpose="one locked evaluation",
                    evidence_path=evidence,
                )

            record_holdout_use(
                policy,
                ledger_path=root / "uses.jsonl",
                property_name="polyreactivity",
                source="new_locked",
                experiment_id="exp-locked",
                purpose="one locked evaluation",
                evidence_path=evidence,
                confirm_locked_use=True,
            )
            with self.assertRaisesRegex(PolicyError, "already recorded"):
                record_holdout_use(
                    policy,
                    ledger_path=root / "uses.jsonl",
                    property_name="polyreactivity",
                    source="new_locked",
                    experiment_id="exp-locked-repeat",
                    purpose="repeat locked evaluation",
                    evidence_path=evidence,
                    confirm_locked_use=True,
                )

    def test_final_use_is_confirmed_and_single_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "uses.jsonl"
            evidence = root / "final.json"
            evidence.write_text("{}\n", encoding="utf-8")
            kwargs = dict(
                policy=minimal_policy(),
                ledger_path=ledger,
                property_name="polyreactivity",
                source="gdpa3",
                experiment_id="final-001",
                purpose="single-shot final evaluation",
                evidence_path=evidence,
                confirm_final_use=True,
            )
            record_holdout_use(**kwargs)
            with self.assertRaisesRegex(PolicyError, "already recorded"):
                record_holdout_use(**kwargs)

    def test_unknown_source_and_missing_evidence_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(PolicyError, "not assigned"):
                record_holdout_use(
                    minimal_policy(),
                    ledger_path=root / "uses.jsonl",
                    property_name="polyreactivity",
                    source="unknown",
                    experiment_id="exp",
                    purpose="test",
                    evidence_path=root / "missing.json",
                )
```

**Step 2: Run ledger tests and verify RED**

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance.TestHoldoutLedger
```

Expected: the append/guard interfaces do not exist yet.

**Step 3: Implement append-only records**

Use a single `os.write` on a file opened with `O_APPEND`, followed by `fsync`. Never read-modify-rewrite the ledger.

```python
import os


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise PolicyError(f"ledger line {line_number} is not an object")
            records.append(record)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"ledger contains invalid JSON at line {exc.lineno}: {candidate}") from exc
    return records


def _source_role(policy: dict[str, Any], property_name: str, source: str) -> str:
    try:
        roles = policy["properties"][property_name]["source_roles"]
    except KeyError as exc:
        raise PolicyError(f"unknown property: {property_name}") from exc
    for role, entries in roles.items():
        if source in {entry["name"] for entry in entries}:
            return role
    raise PolicyError(f"{source} is not assigned to {property_name}")


def record_holdout_use(
    policy: dict[str, Any],
    *,
    ledger_path: str | Path,
    property_name: str,
    source: str,
    experiment_id: str,
    purpose: str,
    evidence_path: str | Path,
    confirm_locked_use: bool = False,
    confirm_final_use: bool = False,
) -> dict[str, Any]:
    validate_policy(policy)
    role = _source_role(policy, property_name, source)
    if role == "locked_validation" and not confirm_locked_use:
        raise PolicyError("locked-validation use requires confirm_locked_use=True")
    if role == "final" and not confirm_final_use:
        raise PolicyError("final use requires confirm_final_use=True")
    evidence = Path(evidence_path).expanduser().resolve()
    evidence_hash = file_sha256(evidence)
    if evidence_hash is None:
        raise PolicyError(f"evidence file not found: {evidence}")
    existing = read_ledger(ledger_path)
    if role in {"locked_validation", "final"} and any(
        item.get("property") == property_name and item.get("source") == source
        for item in existing
    ):
        raise PolicyError(f"{role} source {source} already recorded for {property_name}")
    if role == "final" and any(
        item.get("source") == source and item.get("experiment_id") != experiment_id.strip()
        for item in existing
    ):
        raise PolicyError(
            f"final source {source} was already spent by another experiment; "
            "all property records must share the original experiment_id"
        )
    if not experiment_id.strip() or not purpose.strip():
        raise PolicyError("experiment_id and purpose must be nonempty")
    record = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _utc_now(),
        "property": property_name,
        "source": source,
        "role": role,
        "experiment_id": experiment_id.strip(),
        "purpose": purpose.strip(),
        "evidence_path": str(evidence),
        "evidence_sha256": evidence_hash,
    }
    target = Path(ledger_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    return record
```

Create the empty ledger without a shell redirection, using `apply_patch`, so the file is visible in review.

**Step 4: Run tests and verify GREEN**

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance
```

Expected: all policy, evidence, and ledger tests pass.

**Step 5: Commit the ledger guard**

```bash
git add model/benchmark_governance.py tests/test_benchmark_governance.py tools/holdout_use_ledger.jsonl
git commit -m "feat: guard holdout use with append-only ledger"
```

---

### Task 4: Add the beginner CLI and real status snapshot

**Files:**

- Create: `tools/benchmark_status.py`
- Modify: `tests/test_benchmark_governance.py`
- Generate by command, then review (ignored runtime artifact): `results/benchmark_status.json`

**Step 1: Add failing rendering and CLI tests**

The unit test should call `render_status` directly rather than spawning Python. One lightweight subprocess smoke test comes after implementation.

```python
from tools.benchmark_status import render_status


class TestStatusRendering(unittest.TestCase):
    def test_render_keeps_historical_and_hard_transfer_separate(self) -> None:
        report = {
            "generated_at": "2026-07-17T00:00:00+00:00",
            "metric_definitions": {"primary": "macro_source_roc_auc", "required": []},
            "properties": {
                "polyreactivity": {
                    "source_roles": {
                        "development": [{"name": "aims_flu", "prior_use": True}],
                        "locked_validation": [],
                        "final": [{"name": "gdpa3", "prior_use": False}],
                    },
                    "baselines": [
                        {"name": "historical_platform_scorecard", "status": "ok", "value": 0.7},
                        {"name": "hard_library_to_diverse_transfer", "status": "ok", "value": 0.6269},
                    ],
                    "counts": [],
                }
            },
            "frozen_artifacts": [],
            "holdout_uses": [],
        }
        text = render_status(report)
        self.assertIn("historical_platform_scorecard: 0.700", text)
        self.assertIn("hard_library_to_diverse_transfer: 0.627", text)
        self.assertIn("final (UNSPENT): gdpa3", text)
        self.assertIn("locked validation: NONE ASSIGNED", text)
```

**Step 2: Run the rendering test and verify RED**

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance.TestStatusRendering
```

Expected: `tools.benchmark_status` is missing.

**Step 3: Implement the CLI**

Create `tools/benchmark_status.py` with two modes:

```python
"""Print benchmark status or append a traceable holdout-use record."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from model.benchmark_governance import (
    PolicyError,
    build_status,
    load_policy,
    record_holdout_use,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "tools/benchmark_policy.json"
DEFAULT_LEDGER = ROOT / "tools/holdout_use_ledger.jsonl"
DEFAULT_OUT = ROOT / "results/benchmark_status.json"


def _format_value(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_status(report: dict[str, Any]) -> str:
    lines = [
        "Antibody benchmark governance status",
        f"generated: {report['generated_at']}",
        f"primary metric: {report['metric_definitions']['primary']}",
        "required metrics: " + ", ".join(report["metric_definitions"]["required"]),
    ]
    for property_name, prop in report["properties"].items():
        lines.extend(["", property_name.upper()])
        roles = prop["source_roles"]
        for role in ("development", "locked_validation", "final"):
            entries = roles[role]
            label = role.replace("_", " ")
            if not entries:
                lines.append(f"  {label}: NONE ASSIGNED")
                continue
            states = {
                "previously used" if entry["prior_use"] else "UNSPENT"
                for entry in entries
            }
            if len(states) == 1:
                names = ", ".join(entry["name"] for entry in entries)
                lines.append(f"  {label} ({states.pop()}): {names}")
            else:
                rendered = ", ".join(
                    f"{entry['name']} ({'previously used' if entry['prior_use'] else 'UNSPENT'})"
                    for entry in entries
                )
                lines.append(f"  {label}: {rendered}")
        lines.append("  traceable baselines:")
        for baseline in prop["baselines"]:
            if baseline["status"] == "ok":
                lines.append(f"    {baseline['name']}: {_format_value(baseline['value'])}")
            else:
                lines.append(f"    {baseline['name']}: MISSING ({baseline['error']})")
        lines.append("  independent counts:")
        for count in prop["counts"]:
            value = count["value"] if count["status"] == "ok" else f"MISSING ({count['error']})"
            lines.append(f"    {count['name']}: {value}")
    missing = [item["path"] for item in report["frozen_artifacts"] if not item["present"]]
    lines.extend([
        "",
        f"frozen artifacts: {len(report['frozen_artifacts']) - len(missing)}/{len(report['frozen_artifacts'])} present",
        f"future holdout-use records: {len(report['holdout_uses'])}",
    ])
    if missing:
        lines.append("missing frozen files: " + ", ".join(missing))
    return "\n".join(lines)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
```

Build an `argparse` parser with a default `status` action and an explicit `record-use` subcommand:

```text
python -m tools.benchmark_status
python -m tools.benchmark_status status --out results/benchmark_status.json
python -m tools.benchmark_status record-use \
  --property polyreactivity \
  --source aims_flu \
  --experiment-id cdr-ablation-seed-44 \
  --purpose "fixed development comparison" \
  --evidence results/my_trial.json
```

Required record-use flags are `--property`, `--source`, `--experiment-id`, `--purpose`, and `--evidence`. Add boolean `--confirm-locked-use` and `--confirm-final-use`, pass them directly to `record_holdout_use`, and print the appended record. Catch `PolicyError`, print `ERROR: <exact message>` to stderr, and return exit code `2`. Never catch unexpected exceptions.

To make the no-argument form work, treat no subcommand as `status`. For status, load and validate the policy, call `build_status`, atomically write `--out`, print `render_status(report)`, print the output path, and return `0` even when an evidence file is missing—the report itself must expose the missing evidence. Invalid policy must return `2`.

**Step 4: Run tests and the real command**

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance
/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.benchmark_status
```

Expected user-facing highlights:

```text
POLYREACTIVITY
  development (previously used): aims_mouse, aims_flu, shehata2019
  locked validation: NONE ASSIGNED
  final (UNSPENT): gdpa3
    historical_platform_scorecard: 0.700
    hard_library_to_diverse_transfer: 0.627
```

The output must also state the real independent counts from the current files, the aggregation and hydrophobicity baselines, any missing frozen file, and the path `results/benchmark_status.json`.

Do not run `record-use` against the real ledger merely as a smoke test. Its tests already use temporary ledgers.

**Step 5: Commit the CLI, not the generated snapshot**

First inspect `results/benchmark_status.json`; verify it contains hashes but no sequence data. The repository already ignores `results/*.json`, so leave this reproducible snapshot untracked rather than forcing a timestamped file into Git. Then:

```bash
git add tools/benchmark_status.py tests/test_benchmark_governance.py
git commit -m "feat: report frozen benchmark status"
```

---

### Task 5: Reconcile the human scorecard and verify the Phase 0 slice

**Files:**

- Modify: `PLATFORM_SCORECARD.md`

**Step 1: Add a precise interpretation section**

Do not rewrite or erase the historical scorecard. Add a section immediately after its current summary table:

```markdown
## Benchmark interpretation (Phase 0 lock)

The table above is the historical platform scorecard. It is not a claim that a
library-trained polyreactivity model achieves 0.70 on unseen diverse antibodies.
The current fixed hard-transfer baseline is 0.626926 macro-source ROC AUC across
`aims_mouse`, `aims_flu`, and `shehata2019`; all three have already been used for
development. Aggregation's strict solid-study development estimate is 0.739.
Hydrophobicity currently has no newer hard-transfer AUC registered in machine-
readable evidence, so its traceable Phase 0 value remains the historical 0.70.

GDPa3 remains the unspent, single-shot final source for all three properties. It
must not be inspected, downloaded, counted, trained on, or tuned against during
development. No locked-validation source is assigned yet.

Run the traceable status report:

```bash
cd /Users/saptarshighosh/Biostuff
/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.benchmark_status
```
```

If the command in Task 4 reports a value different from the text above, update the prose to the real traceable value before committing.

**Step 2: Run the complete focused verification**

```bash
/Users/saptarshighosh/Biostuff/.venv/bin/python -m unittest tests.test_benchmark_governance tests.test_run_manifest
/Users/saptarshighosh/Biostuff/.venv/bin/python -m tools.benchmark_status
git diff --check
git status --short
```

Expected:

- all focused tests pass;
- the status command performs no model training and writes only `results/benchmark_status.json`;
- polyreactivity prints both `0.700` historical and `0.627` hard transfer with different names;
- `aims_flu` reports its real small hard-transfer test count (`97` in the frozen gate evidence), making its instability visible;
- GDPa3 appears only as `final (UNSPENT)` and is never accessed;
- `git diff --check` is clean;
- unrelated pre-existing modified/untracked files remain untouched.

**Step 3: Self-review against the approved design**

Before the final commit, answer these in the implementation notes:

- Does every printed AUC resolve from a named evidence file and JSON pointer?
- Are historical, hard-transfer, locked-validation, and final evidence clearly different?
- Are source counts labeled by what they actually count (dataset rows, test rows, assay rows, or training rows)?
- Does a missing file yield `MISSING` rather than a default number?
- Can a locked source be recorded without explicit confirmation? It must not.
- Can GDPa3 be recorded twice for one property? It must not.
- Did any command inspect or download GDPa3? It must not.
- Did any served artifact or training head change? It must not.

**Step 4: Commit only the documentation change**

```bash
git add PLATFORM_SCORECARD.md
git commit -m "docs: distinguish hard-transfer benchmarks"
```

Checkpoint: `git log -5 --oneline` should show small, independently reviewable Phase 0 commits. Do not bundle unrelated overnight, hunter, Colab, or model-serving files.

---

## Phase 0 completion criteria

Phase 0 is complete only when all of the following are true:

1. `python -m tools.benchmark_status` succeeds from the repo root with the project interpreter.
2. The generated JSON records policy schema, timestamp, evidence hashes, frozen file hashes/sizes, role assignments, exact source/assay counts, and ledger records.
3. Historical aggregation/polyreactivity/HIC scores remain visible but explicitly historical.
4. Hard diverse polyreactivity is the traceable `0.626926...` development baseline, not the misleading within-source mean.
5. Every currently used development source is marked `prior_use: true` without fabricated historical timestamps.
6. No locked-validation source is falsely claimed; the report says none is assigned.
7. GDPa3 remains policy-only, unspent, uncounted, and single-shot guarded.
8. The only runtime write in ordinary status mode is the atomic status JSON; ledger writes require the explicit `record-use` command.
9. Focused tests and `git diff --check` pass.
10. No model is retrained and no serving artifact changes.

This is deliberately a governance slice, not a modeling win. Its value is making every later data, PLM, multitask, uncertainty, and active-learning result comparable to the same honest baseline without spending the final benchmark.
