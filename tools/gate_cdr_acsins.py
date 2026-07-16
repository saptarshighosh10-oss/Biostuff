"""External-study AC-SINS CDR feature ablation, with strict shared-sequence removal."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from model.cdr_features import cdr_feature_vector, parse_rate
from model.head_b_acsins_charge import charge_features, _tuned_lgbm
from model.head_b_acsins_unified import _acsins_rows, _seqkey
from model.head_b_catch_rate import BAD_FRACTION, _bad_labels, _metrics, _study
from model.head_b_gbm import _assay_key, build_feature_vector
from model.head_b_pooled import _percentile_targets


def _targets(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(_study(row), _assay_key(row))].append(i)
    target = [0.0] * len(rows)
    for indices in groups.values():
        for i, value in zip(indices, _percentile_targets([float(rows[i]["endpoint_value"]) for i in indices])):
            target[i] = value
    return target


def _evaluate(rows, target, mode, use_plm, cache):
    reports = []
    for held in sorted({_study(row) for row in rows}):
        test = [i for i, row in enumerate(rows) if _study(row) == held]
        held_sequences = {_seqkey(rows[i]) for i in test}
        train = [i for i, row in enumerate(rows)
                 if _study(row) != held and _seqkey(row) not in held_sequences]
        if len(train) < 50 or len(test) < 20:
            continue

        def features(indices):
            out = []
            for i in indices:
                key = (i, mode)
                if key not in cache:
                    row = rows[i]
                    cache[key] = (build_feature_vector(row, use_plm=use_plm) + charge_features(row)
                                  + (cdr_feature_vector(row, mode) if mode else []))
                out.append(cache[key])
            return out

        model = _tuned_lgbm()
        model.set_params(n_jobs=1)
        model.fit(features(train), [target[i] for i in train])
        predicted = dict(zip(test, model.predict(features(test))))
        assays = defaultdict(list)
        for i in test:
            assays[_assay_key(rows[i])].append(i)
        for key, indices in assays.items():
            if len(indices) < 20:
                continue
            metrics = _metrics(_bad_labels([float(rows[i]["endpoint_value"]) for i in indices], "higher_bad", BAD_FRACTION),
                               [predicted[i] for i in indices], BAD_FRACTION)
            if metrics:
                metrics.update({"study": held, "assay": key[2], "parse_rate": parse_rate([rows[i] for i in indices], mode) if mode else {}})
                reports.append(metrics)
    return reports


def _solid_auc(values):
    solid = [value["auc"] for value in values if value["n"] >= 100]
    return float(mean(solid)) if solid else None


def run(use_plm=True, include_heavy_all=False):
    rows = _acsins_rows(include_gdpa=True)
    target = _targets(rows)
    cache = {}
    baseline = _evaluate(rows, target, None, use_plm, cache)
    h3 = _evaluate(rows, target, "h3", use_plm, cache)
    report = {"task": "worst-15%-AC-SINS on unseen studies; strict shared-sequence removal",
              "features": "plm+biophysical+charge" if use_plm else "biophysical+charge",
              "n_rows": len(rows), "baseline": baseline, "h3": h3,
              "baseline_solid_auc": _solid_auc(baseline), "h3_solid_auc": _solid_auc(h3)}
    report["h3_passed"] = report["h3_solid_auc"] >= report["baseline_solid_auc"] - 0.005
    if include_heavy_all and report["h3_passed"]:
        all_heavy = _evaluate(rows, target, "heavy_all", use_plm, cache)
        report["heavy_all"] = all_heavy
        report["heavy_all_solid_auc"] = _solid_auc(all_heavy)
        report["heavy_all_passed"] = report["heavy_all_solid_auc"] >= report["baseline_solid_auc"] - 0.005
    return report


def _print(report):
    print("CDR AC-SINS gate — unseen studies, strict shared-sequence removal")
    print(f"  baseline solid AUC: {report['baseline_solid_auc']:.3f}")
    print(f"  H3 solid AUC      : {report['h3_solid_auc']:.3f} ({'PASS' if report['h3_passed'] else 'QUARANTINE'})")
    for value in report["h3"]:
        print(f"  {value['study'][:20]:<20} n={value['n']:<4} AUC={value['auc']:.3f} H3-parse={value['parse_rate']['h3']:.1%}")
    if "heavy_all" in report:
        print(f"  H1+H2+H3 solid AUC: {report['heavy_all_solid_auc']:.3f} "
              f"({'PASS' if report['heavy_all_passed'] else 'QUARANTINE'})")
        print("  H1/H2 without supplied columns are zero-filled and flagged by their 0% parse rate.")


if __name__ == "__main__":
    # This isolates the hand-crafted CDR ablation from optional PLM embeddings.
    result = run(use_plm=False, include_heavy_all=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/gate_cdr_acsins.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print(result)
