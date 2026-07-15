"""Overnight hyperparameter search for the AC-SINS smoke-detector.

Objective: maximise mean ROC-AUC on the STRICT leakage-free leave-one-study-out
AC-SINS test (the honest 'catch the sticky ones on unseen antibodies' number,
solid studies n>=100). Features are precomputed ONCE; each config only refits the
models, so dozens of configs run comfortably overnight on CPU.

Resumable: every evaluated config is appended to results/overnight_tune.jsonl and
skipped on restart. Best-so-far is printed and saved to results/overnight_tune_best.json.

Run (background):
  python -m tools.overnight_tune
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path
from statistics import mean

import numpy as np

from model.head_b_gbm import _assay_key, build_feature_vector
from model.head_b_pooled import _percentile_targets
from model.head_b_catch_rate import _study, _is_acsins, _bad_labels, _metrics, BAD_FRACTION
from model.head_b_acsins_unified import _acsins_rows, _seqkey

LOG = Path("results/overnight_tune.jsonl")
BEST = Path("results/overnight_tune_best.json")


def _build_predict(cfg: dict):
    """Return predict_fn(Xtr, ytr, Xte) -> list[float] for a config dict."""
    def rank01(v):
        order = np.argsort(np.argsort(v))
        return order / max(len(v) - 1, 1)

    def predict(Xtr, ytr, Xte):
        preds = []
        if cfg.get("lgbm"):
            from lightgbm import LGBMRegressor
            p = cfg["lgbm"]
            m = LGBMRegressor(n_estimators=p["n_estimators"], num_leaves=p["num_leaves"],
                              learning_rate=p["learning_rate"], min_child_samples=p["min_child_samples"],
                              reg_lambda=p["reg_lambda"], subsample=p.get("subsample", 0.8),
                              subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.8),
                              random_state=42, verbose=-1)
            m.fit(Xtr, ytr)
            preds.append(rank01(np.asarray(m.predict(Xte))))
        if cfg.get("hgb"):
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_iter=cfg["hgb"]["max_iter"],
                                              learning_rate=cfg["hgb"]["learning_rate"],
                                              l2_regularization=cfg["hgb"]["l2"], random_state=42)
            m.fit(Xtr, ytr)
            preds.append(rank01(np.asarray(m.predict(Xte))))
        if cfg.get("ridge"):
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            m = make_pipeline(StandardScaler(), Ridge(alpha=cfg["ridge"]["alpha"]))
            m.fit(Xtr, ytr)
            preds.append(rank01(np.asarray(m.predict(Xte))))
        return list(np.mean(preds, axis=0))
    return predict


def _search_space() -> list[dict]:
    configs = []
    # LightGBM-centric grid (the workhorse)
    for ne, nl, lr, mcs, rl in itertools.product(
        [400, 800, 1200], [31, 63], [0.02, 0.03, 0.05], [20, 30, 50], [0.0, 1.0, 5.0]
    ):
        lgbm = {"n_estimators": ne, "num_leaves": nl, "learning_rate": lr,
                "min_child_samples": mcs, "reg_lambda": rl, "subsample": 0.8, "colsample_bytree": 0.8}
        configs.append({"name": f"lgbm_ne{ne}_nl{nl}_lr{lr}_mcs{mcs}_rl{rl}", "lgbm": lgbm})
    # a few strong ensembles on top of the best-guess lgbm
    best_lgbm = {"n_estimators": 800, "num_leaves": 63, "learning_rate": 0.03,
                 "min_child_samples": 30, "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8}
    hgb = {"max_iter": 500, "learning_rate": 0.05, "l2": 1.0}
    for name, cfg in [
        ("ens_lgbm_hgb", {"lgbm": best_lgbm, "hgb": hgb}),
        ("ens_lgbm_ridge", {"lgbm": best_lgbm, "ridge": {"alpha": 10.0}}),
        ("ens_lgbm_hgb_ridge", {"lgbm": best_lgbm, "hgb": hgb, "ridge": {"alpha": 10.0}}),
        ("ens_lgbm_hgb_ridge_a30", {"lgbm": best_lgbm, "hgb": hgb, "ridge": {"alpha": 30.0}}),
    ]:
        configs.append({"name": name, **cfg})
    return configs


def main() -> None:
    rows = _acsins_rows(True)
    # per-(study,assay) percentile targets
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(_study(r), _assay_key(r))].append(i)
    y = np.zeros(len(rows))
    for key, idxs in buckets.items():
        for i, p in zip(idxs, _percentile_targets([float(rows[i]["endpoint_value"]) for i in idxs])):
            y[i] = p

    print(f"precomputing features for {len(rows)} rows ...", flush=True)
    X = np.asarray([build_feature_vector(r, True) for r in rows], dtype=float)
    study = [_study(r) for r in rows]
    seqk = [_seqkey(r) for r in rows]
    akey = [_assay_key(r) for r in rows]
    studies = sorted(set(study))
    print(f"features ready: X={X.shape}", flush=True)

    def objective(predict) -> dict:
        solid, allv = [], []
        for held in studies:
            test_idx = [i for i in range(len(rows)) if study[i] == held]
            held_seqs = {seqk[i] for i in test_idx}
            train_idx = [i for i in range(len(rows)) if study[i] != held and seqk[i] not in held_seqs]
            if len(train_idx) < 50 or len(test_idx) < 20:
                continue
            pr = predict(X[train_idx], y[train_idx], X[test_idx])
            pred = dict(zip(test_idx, pr))
            by = defaultdict(list)
            for i in test_idx:
                by[akey[i]].append(i)
            for key, idxs in by.items():
                if len(idxs) < 20:
                    continue
                vals = [float(rows[i]["endpoint_value"]) for i in idxs]
                bad = _bad_labels(vals, "higher_bad", BAD_FRACTION)
                m = _metrics(bad, [pred[i] for i in idxs], BAD_FRACTION)
                if m:
                    allv.append(m["auc"])
                    if len(idxs) >= 100:
                        solid.append(m["auc"])
        return {"solid_auc": round(mean(solid), 4) if solid else None,
                "all_auc": round(mean(allv), 4) if allv else None,
                "n_solid": len(solid)}

    done = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass

    configs = [c for c in _search_space() if c["name"] not in done]
    print(f"{len(configs)} configs to evaluate ({len(done)} already done)", flush=True)
    best = json.loads(BEST.read_text()) if BEST.exists() else {"solid_auc": -1}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for n, cfg in enumerate(configs, 1):
        res = objective(_build_predict(cfg))
        rec = {"name": cfg["name"], "config": cfg, **res}
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        flag = ""
        if res["solid_auc"] is not None and res["solid_auc"] > (best.get("solid_auc") or -1):
            best = rec
            BEST.write_text(json.dumps(best, indent=2))
            flag = "  <-- NEW BEST"
        print(f"[{n}/{len(configs)}] {cfg['name']:<40} solid_auc={res['solid_auc']} all={res['all_auc']}{flag}",
              flush=True)

    print("\n=== DONE ===")
    print(f"best: {best['name']}  solid_auc={best.get('solid_auc')}  (baseline ensemble was 0.717)")
    print(f"saved best -> {BEST}")


if __name__ == "__main__":
    main()
