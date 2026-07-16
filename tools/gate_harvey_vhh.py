"""Memory-safe diverse-holdout gate for the Harvey VHH PSR candidate."""

from __future__ import annotations

import json
import random
import csv
import argparse
from pathlib import Path

import numpy as np
import model.features as model_features

from model.head_c_polyreactivity import _binary_labels, _feat, _model


LIBRARIES = ("emi", "igg", "iso", "tessier_lib1", "tessier_lib2")
TESTS = ("aims_mouse", "aims_flu")
CANDIDATE = "harvey2022_vhh"
MAX_PER_SOURCE = 2_000
DATA = Path.home() / "Downloads/antibody_sources/polyreactivity_dataset.csv"


def _unique(rows):
    by_vh = {}
    for row in rows:
        by_vh.setdefault(row["variant_sequence"], row)
    return list(by_vh.values())


def _matrix(rows):
    width = len(_feat(rows[0]))
    X = np.empty((len(rows), width), dtype=np.float32)
    for index, row in enumerate(rows):
        X[index] = _feat(row)
        if index and index % 1_000 == 0:
            print(f"  built {index:,}/{len(rows):,} feature rows", flush=True)
    return X


def _load_gate_rows():
    """Stream only the reservoir and holdout rows, not the whole CSV."""
    selected = set(LIBRARIES + (CANDIDATE,) + TESTS)
    samples = {source: {0: [], 1: []} for source in LIBRARIES + (CANDIDATE,)}
    tests = {source: [] for source in TESTS}
    seen = {source: set() for source in selected}
    totals = {source: 0 for source in selected}
    label_totals = {source: {0: 0, 1: 0} for source in samples}
    rng = {source: random.Random(42) for source in samples}
    with DATA.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            source = record["source"]
            if source not in selected:
                continue
            vh = (record["vh"] or "").strip().upper()
            try:
                label = int(float(record["label"]))
            except ValueError:
                continue
            if len(vh) < 20 or label not in (0, 1) or vh in seen[source]:
                continue
            seen[source].add(vh)
            row = {"variant_sequence": vh, "vh_sequence": vh,
                   "vl_sequence": (record["vl"] or "").strip().upper() or None,
                   "label": label}
            totals[source] += 1
            if source in tests:
                tests[source].append(row)
                continue
            label_totals[source][label] += 1
            bucket = samples[source][label]
            limit = MAX_PER_SOURCE // 2
            if len(bucket) < limit:
                bucket.append(row)
            else:
                replacement = rng[source].randrange(label_totals[source][label])
                if replacement < limit:
                    bucket[replacement] = row
    training = {source: samples[source][0] + samples[source][1] for source in samples}
    return training, tests, totals


def _score(train_rows, test_rows):
    from sklearn.metrics import roc_auc_score

    held_vh = {row["variant_sequence"] for row in test_rows}
    train_rows = _unique(row for row in train_rows if row["variant_sequence"] not in held_vh)
    X_train = _matrix(train_rows)
    print(f"  fitting {len(train_rows):,} rows", flush=True)
    model = _model()
    model.set_params(n_jobs=1)
    model.fit(X_train, _binary_labels(train_rows))
    X_test = _matrix(test_rows)
    prediction = model.predict_proba(X_test)[:, 1]
    return {"auc": float(roc_auc_score(_binary_labels(test_rows), prediction)),
            "train_n": len(train_rows), "test_n": len(test_rows),
            "test_positives": sum(_binary_labels(test_rows))}


def run():
    # Head C is documented as biophysical+charge.  Avoid loading the optional,
    # unrelated ProteinGym scorer once per sequence during this gate.
    model_features._FITNESS_HEAD_TRIED = True
    model_features._FITNESS_HEAD = None
    training, tests, totals = _load_gate_rows()
    baseline = [row for source in LIBRARIES for row in training[source]]
    candidate = training[CANDIDATE]
    trial = baseline + candidate
    report = {"candidate": CANDIDATE, "candidate_n": totals[CANDIDATE],
              "candidate_used_n": len(candidate), "max_per_source": MAX_PER_SOURCE,
              "baseline": {}, "trial": {}}
    for source, test_rows in tests.items():
        print(f"baseline: {source}", flush=True)
        report["baseline"][source] = _score(baseline, test_rows)
        print(f"trial: {source}", flush=True)
        report["trial"][source] = _score(trial, test_rows)
    baseline_auc = np.mean([value["auc"] for value in report["baseline"].values()])
    trial_auc = np.mean([value["auc"] for value in report["trial"].values()])
    report.update({"baseline_mean_auc": float(baseline_auc), "trial_mean_auc": float(trial_auc),
                   "accepted": bool(trial_auc >= baseline_auc - 0.005)})
    return report


def run_one(arm, source):
    """Run one fit at a time so the local command runner can report it."""
    model_features._FITNESS_HEAD_TRIED = True
    model_features._FITNESS_HEAD = None
    training, tests, totals = _load_gate_rows()
    baseline = [row for name in LIBRARIES for row in training[name]]
    rows = baseline if arm == "baseline" else baseline + training[CANDIDATE]
    result = _score(rows, tests[source])
    result.update({"arm": arm, "source": source, "candidate_n": totals[CANDIDATE],
                   "candidate_used_n": len(training[CANDIDATE]),
                   "max_per_source": MAX_PER_SOURCE})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("baseline", "trial"))
    parser.add_argument("--test", choices=TESTS)
    args = parser.parse_args()
    report = run_one(args.arm, args.test) if args.arm else run()
    print(json.dumps(report, indent=2, sort_keys=True))
