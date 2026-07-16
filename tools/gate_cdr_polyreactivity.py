"""Leakage-free diverse-polyreactivity CDR feature ablation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import model.features as model_features
from model.cdr_features import cdr_feature_vector, parse_rate
from model.head_b_acsins_charge import charge_features
from model.head_b_gbm import build_feature_vector
from model.head_c_polyreactivity import _binary_labels, _model


LIBRARIES = ("emi", "igg", "iso", "tessier_lib1", "tessier_lib2")
TESTS = ("aims_mouse", "aims_flu", "shehata2019")
MAX_PER_SOURCE = 2_000
DATA = Path.home() / "Downloads/antibody_sources/polyreactivity_dataset.csv"


def _load():
    selected = set(LIBRARIES + TESTS)
    samples = {source: {0: [], 1: []} for source in LIBRARIES}
    tests = {source: [] for source in TESTS}
    seen = {source: set() for source in selected}
    totals = {source: 0 for source in selected}
    class_totals = {source: {0: 0, 1: 0} for source in selected}
    rng = {source: random.Random(42) for source in LIBRARIES}
    with DATA.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            source = record["source"]
            if source not in selected:
                continue
            vh = (record["vh"] or "").strip().upper()
            vl = (record["vl"] or "").strip().upper() or None
            if len(vh) < 20 or vh in seen[source]:
                continue
            try:
                label = float(record["label"])
            except (TypeError, ValueError):
                continue
            seen[source].add(vh)
            row = {"variant_sequence": vh, "vh_sequence": vh, "vl_sequence": vl, "label": label}
            totals[source] += 1
            if source in tests:
                tests[source].append(row)
                continue
            binary = int(label)
            if binary not in (0, 1):
                continue
            class_totals[source][binary] += 1
            bucket = samples[source][binary]
            limit = MAX_PER_SOURCE // 2
            if len(bucket) < limit:
                bucket.append(row)
            else:
                replacement = rng[source].randrange(class_totals[source][binary])
                if replacement < limit:
                    bucket[replacement] = row
    return {source: samples[source][0] + samples[source][1] for source in LIBRARIES}, tests, totals, class_totals


def _features(rows, mode: str | None, cache: dict[tuple[str, str | None], list[float]]):
    out = []
    for row in rows:
        key = (row["variant_sequence"], mode)
        if key not in cache:
            value = build_feature_vector(row, use_plm=False) + charge_features(row)
            cache[key] = value + (cdr_feature_vector(row, mode) if mode else [])
        out.append(cache[key])
    return out


def _score(train, test, mode, cache):
    held = {row["variant_sequence"] for row in test}
    unique = {}
    for row in train:
        if row["variant_sequence"] not in held:
            unique.setdefault(row["variant_sequence"], row)
    train = list(unique.values())
    model = _model()
    model.set_params(n_jobs=1)
    model.fit(_features(train, mode, cache), _binary_labels(train))
    prediction = model.predict_proba(_features(test, mode, cache))[:, 1]
    return {"auc": float(roc_auc_score(_binary_labels(test), prediction)),
            "train_n": len(train), "test_n": len(test), "test_positives": sum(_binary_labels(test)),
            "parse_rate": parse_rate(test, mode) if mode else {}}


def _mean(report):
    return float(np.mean([value["auc"] for value in report.values()]))


def run():
    model_features._FITNESS_HEAD_TRIED = True
    model_features._FITNESS_HEAD = None
    training, tests, totals, class_totals = _load()
    baseline_rows = [row for source in LIBRARIES for row in training[source]]
    cache = {}
    report = {"libraries": list(LIBRARIES), "holdouts": list(TESTS), "max_per_source": MAX_PER_SOURCE,
              "source_rows": totals, "library_class_rows": class_totals,
              "baseline": {}, "h3": {}, "heavy_all": None}
    for source, rows in tests.items():
        report["baseline"][source] = _score(baseline_rows, rows, None, cache)
        report["h3"][source] = _score(baseline_rows, rows, "h3", cache)
        if report["heavy_all"] is None:
            report["heavy_all"] = {}
        report["heavy_all"][source] = _score(baseline_rows, rows, "heavy_all", cache)
    report["baseline_mean_auc"] = _mean(report["baseline"])
    report["h3_mean_auc"] = _mean(report["h3"])
    report["h3_passed"] = report["h3_mean_auc"] >= report["baseline_mean_auc"] - 0.005
    report["heavy_all_mean_auc"] = _mean(report["heavy_all"])
    report["heavy_all_passed"] = report["heavy_all_mean_auc"] >= report["baseline_mean_auc"] - 0.005
    return report


def _print(report):
    print("CDR polyreactivity gate — fixed diverse holdouts, strict shared-VH removal")
    print(f"  baseline mean AUC: {report['baseline_mean_auc']:.3f}")
    print(f"  H3 mean AUC      : {report['h3_mean_auc']:.3f} ({'PASS' if report['h3_passed'] else 'QUARANTINE'})")
    for source in TESTS:
        b, h = report["baseline"][source], report["h3"][source]
        print(f"  {source:<15} n={b['test_n']:<4} baseline={b['auc']:.3f} h3={h['auc']:.3f} "
              f"H3-parse={h['parse_rate']['h3']:.1%}")
    print(f"  H1+H2+H3 mean AUC: {report['heavy_all_mean_auc']:.3f} "
          f"({'PASS' if report['heavy_all_passed'] else 'QUARANTINE'})")
    for source in TESTS:
        rates = report["heavy_all"][source]["parse_rate"]
        print(f"  {source:<15} H1/H2/H3-parse={rates['h1']:.1%}/{rates['h2']:.1%}/{rates['h3']:.1%}")
    print("  Note: aims_flu has n=97; its AUC is intrinsically noisy.")


if __name__ == "__main__":
    result = run()
    Path("results").mkdir(exist_ok=True)
    Path("results/gate_cdr_polyreactivity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print(result)
