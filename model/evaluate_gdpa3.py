"""GDPa3 external evaluation — single-shot, fail-closed (design §11).

GDPa3 is a frozen holdout: it is never downloaded, inspected, or logged during
development (`data.gdpa.load_gdpa3` is a hard tripwire). This command is the ONE
sanctioned path that reads GDPa3 labels, and only after every gate passes:

  1. Append-only attempt ledger (`results/gdpa3_attempts.jsonl`). If it already
     records a COMPLETED attempt, refuse — no result-driven re-runs.
  2. The GDPa3 file must exist and its SHA-256 must equal `expected_data_hash`.
     Any mismatch/absence → ledger an `aborted` row and raise (fail-closed).
  3. Labels are loaded only after 1–2 pass and the model/predictions are frozen.

`predict_fn` and `load_labels_fn` are injectable so the gating logic is testable
without a real model or the real (embargoed) GDPa3 data.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Callable

LEDGER_PATH = Path("results/gdpa3_attempts.jsonl")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _read_ledger(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    return [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]


def _append_ledger(ledger_path: Path, entry: dict) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def has_completed_attempt(ledger_path: Path = LEDGER_PATH) -> bool:
    return any(e.get("state") == "completed" for e in _read_ledger(ledger_path))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run_once(
    *,
    gdpa3_file: str | Path,
    expected_data_hash: str,
    out_path: str | Path,
    predict_fn: Callable[[list[dict]], list[dict]],
    load_labels_fn: Callable[[Path], list[dict]] | None = None,
    ledger_path: Path = LEDGER_PATH,
    manifest: dict | None = None,
) -> dict:
    """Run the one-and-only external GDPa3 evaluation. Fail-closed at every gate.

    `predict_fn(rows) -> list[dict]` scores frozen inputs (each result should
    carry at least a risk score); `load_labels_fn(path) -> list[dict]` reads the
    GDPa3 label rows (defaults to a strict local CSV read). Returns the written
    report dict."""
    ledger_path = Path(ledger_path)
    gdpa3_file = Path(gdpa3_file)

    # Gate 1: no repeated completed attempt.
    if has_completed_attempt(ledger_path):
        raise RuntimeError(
            "GDPa3 already has a completed attempt in the ledger; single-shot "
            "policy forbids re-running (design §11, fail-closed)."
        )

    _append_ledger(ledger_path, {"state": "started", "ts": _now(),
                                 "expected_data_hash": expected_data_hash,
                                 "manifest": manifest or {}})

    # Gate 2: file present and hash matches the pre-registered expectation.
    if not gdpa3_file.exists():
        _append_ledger(ledger_path, {"state": "aborted", "ts": _now(),
                                     "reason": "gdpa3_file_missing", "path": str(gdpa3_file)})
        raise FileNotFoundError(
            f"GDPa3 file not found at {gdpa3_file}; nothing evaluated (fail-closed)."
        )
    actual_hash = _sha256_file(gdpa3_file)
    if actual_hash != expected_data_hash:
        _append_ledger(ledger_path, {"state": "aborted", "ts": _now(),
                                     "reason": "data_hash_mismatch",
                                     "expected": expected_data_hash, "actual": actual_hash})
        raise ValueError(
            f"GDPa3 data hash mismatch (expected={expected_data_hash}, actual={actual_hash}); "
            f"refusing to evaluate against unverified data (fail-closed)."
        )

    # Gate 3 passed: labels loaded only now, after inputs/model are frozen.
    loader = load_labels_fn or _default_label_loader
    rows = loader(gdpa3_file)
    predictions = predict_fn(rows)

    report = {
        "head": "antibody_aggregation",
        "external_set": "gdpa3",
        "data_hash": actual_hash,
        "n_rows": len(rows),
        "predictions": predictions,
        "manifest": manifest or {},
        "ts": _now(),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    _append_ledger(ledger_path, {"state": "completed", "ts": _now(),
                                 "data_hash": actual_hash, "out": str(out_path),
                                 "n_rows": len(rows)})
    return report


def _default_label_loader(path: Path) -> list[dict]:
    import csv
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GDPa3 single-shot external evaluation")
    parser.add_argument("--run-once", action="store_true", required=True)
    parser.add_argument("--gdpa3-file", required=True)
    parser.add_argument("--expected-data-hash", required=True)
    parser.add_argument("--out", default="results/gdpa3_report.json")
    args = parser.parse_args()

    def _predict(rows: list[dict]) -> list[dict]:
        raise SystemExit(
            "Wire a frozen, serialized model into predict_fn before running GDPa3. "
            "This command intentionally ships no default scorer to prevent an "
            "accidental un-frozen run."
        )

    run_once(gdpa3_file=args.gdpa3_file, expected_data_hash=args.expected_data_hash,
             out_path=args.out, predict_fn=_predict)
