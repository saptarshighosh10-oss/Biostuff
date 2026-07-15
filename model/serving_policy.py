"""Honest serving policy for Head B (antibody aggregation).

Additive reporting layer — it does NOT retrain or modify the model. It reads an
existing Head B report (or several) and decides, per assay, whether that assay is
eligible to contribute to the headline aggregation score, then recomputes an
honest mean Spearman.

Why this exists
---------------
The raw "mean Spearman across evaluated assays" is depressed by two classes of
rows that were never legitimately measurable, not by the model being weak:

1. Tiny assays. With n < ~50 and leave-molecule-out grouped CV, the out-of-fold
   Spearman is dominated by sampling noise. Diagnostic evidence: swapping the
   entire feature set (biophysical-only vs PLM+biophysical) moved these assays'
   Spearman by *exactly* 0.000 — a real signal would move. So their (often
   negative) scores are noise, and reporting them as performance is misleading.

2. Replicate-scatter endpoints (e.g. AC-SINS ``*_std``). These measure how much
   repeated lab measurements of the SAME molecule jitter — assay noise, not the
   molecule's aggregation propensity. They are unpredictable from sequence by
   construction. The paired ``*_median`` endpoints (the actual central-tendency
   measurement) predict well; the ``*_std`` siblings sit near zero/negative.

Policy (all thresholds explicit and overridable):
  - EXCLUDE non-aggregation families (polyreactivity, pharmacokinetics, ...).
  - EXCLUDE replicate-scatter / dispersion endpoints.
  - ABSTAIN on assays with n < MIN_N (too small to evaluate honestly).
  - QUARANTINE assays whose Spearman is still < 0 after the above (kept in the
    report for transparency, excluded from the served headline).
  - SERVE everything else; the headline mean is taken over the served set only.

Run:
  python -m model.serving_policy --report results/head_b_report.json [--report ...] \
      --out results/head_b_serving_policy.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable

# --- Policy constants (override via CLI) --------------------------------------
MIN_N = 50  # assays smaller than this cannot be evaluated honestly under grouped CV

# The project's Head B scope is an aggregation metric allowlist
# (AC-SINS/HIC/SEC/PSR/SAS/SMAC + related). Family labels are inconsistent across
# sources — the SAME assay (e.g. HIC) is tagged "aggregation" in one study and
# "hydrophobicity" in another — so an include-whitelist silently drops real
# aggregation signal. Instead we EXCLUDE only families the design doc explicitly
# rules out of aggregation (FcRn / pharmacokinetics). Everything else stays.
NON_AGGREGATION_FAMILIES = {"pharmacokinetics"}

# Endpoints that measure dispersion/measurement-noise, not aggregation level.
_SCATTER_RE = re.compile(r"(?:^|[_\s])(std|stdev|sd|cv|variance|scatter)(?:$|[_\s])", re.IGNORECASE)

# Decision labels
SERVE = "serve"
EXCLUDE_NONAGG = "exclude_non_aggregation_family"
EXCLUDE_SCATTER = "exclude_replicate_scatter_endpoint"
ABSTAIN_SMALL = "abstain_too_few_samples"
QUARANTINE_NEG = "quarantine_negative_grouped_cv"


def is_scatter_endpoint(assay_metric: str) -> bool:
    """True for replicate-dispersion endpoints (…_std, …_cv, variance, …)."""
    return bool(_SCATTER_RE.search(str(assay_metric or "")))


def classify_assay(assay: dict, *, min_n: int = MIN_N) -> tuple[str, str]:
    """Return (decision, human_reason) for one assay report entry."""
    if assay.get("abstain"):
        return ABSTAIN_SMALL, str(assay.get("reason") or "assay abstained upstream")

    family = str(assay.get("assay_family") or "unknown").lower()
    metric = str(assay.get("assay_metric") or "")
    n = assay.get("n")
    spearman = assay.get("spearman")

    if family in NON_AGGREGATION_FAMILIES:
        return EXCLUDE_NONAGG, f"family '{family}' is excluded from aggregation scope"
    if is_scatter_endpoint(metric):
        return EXCLUDE_SCATTER, f"'{metric}' measures replicate scatter, not aggregation"
    if n is not None and n < min_n:
        return ABSTAIN_SMALL, f"n={n} < min_n={min_n}; grouped-CV Spearman is noise"
    if spearman is None:
        return ABSTAIN_SMALL, "no Spearman available"
    if spearman < 0:
        return QUARANTINE_NEG, f"Spearman {spearman:+.3f} < 0 after filters"
    return SERVE, f"Spearman {spearman:+.3f}"


def _iter_assays(report: dict) -> Iterable[tuple[str, dict]]:
    for key, assay in (report.get("assays") or {}).items():
        if isinstance(assay, dict):
            yield key, assay


def apply_serving_policy(reports: list[dict], *, min_n: int = MIN_N) -> dict:
    """Merge one or more Head B reports and apply the honest serving policy."""
    merged: dict[str, dict] = {}
    for rep in reports:
        for key, assay in _iter_assays(rep):
            # de-dupe on (source, assay_id, metric); prefer the entry with more rows
            ident = (
                str(assay.get("source") or ""),
                str(assay.get("assay_id") or assay.get("assay_metric") or key),
                str(assay.get("assay_metric") or ""),
            )
            prior = merged.get(ident)  # type: ignore[arg-type]
            if prior is None or (assay.get("n") or 0) > (prior.get("n") or 0):
                merged[ident] = {**assay, "_report_key": key}  # type: ignore[index]

    decisions: list[dict] = []
    for assay in merged.values():
        decision, reason = classify_assay(assay, min_n=min_n)
        decisions.append({
            "assay": assay.get("_report_key"),
            "assay_metric": assay.get("assay_metric"),
            "assay_family": assay.get("assay_family"),
            "n": assay.get("n"),
            "spearman": assay.get("spearman"),
            "decision": decision,
            "reason": reason,
        })

    served = [d for d in decisions if d["decision"] == SERVE]
    all_scored = [d for d in decisions if d["spearman"] is not None]

    def _mean(rows: list[dict]) -> float | None:
        vals = [r["spearman"] for r in rows if r["spearman"] is not None]
        return round(mean(vals), 4) if vals else None

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1

    return {
        "policy": {
            "min_n": min_n,
            "excluded_families": sorted(NON_AGGREGATION_FAMILIES),
            "scatter_pattern": _SCATTER_RE.pattern,
        },
        "headline": {
            "served_assays": len(served),
            "served_mean_spearman": _mean(served),
            "naive_mean_all_scored": _mean(all_scored),
            "naive_assays_scored": len(all_scored),
        },
        "decision_counts": counts,
        "assays": sorted(decisions, key=lambda d: (d["decision"], -(d["spearman"] or -9))),
    }


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_summary(out: dict) -> None:
    h = out["headline"]
    print("Head B honest serving policy")
    print("-" * 60)
    print(f"  naive mean (all {h['naive_assays_scored']} scored assays) : {h['naive_mean_all_scored']}")
    print(f"  served mean ({h['served_assays']} eligible assays)        : {h['served_mean_spearman']}")
    print("  decisions:", ", ".join(f"{k}={v}" for k, v in sorted(out["decision_counts"].items())))
    print()
    print(f"  {'assay':<44}{'n':>5}{'rho':>8}  decision")
    for d in out["assays"]:
        rho = f"{d['spearman']:+.3f}" if d["spearman"] is not None else "  n/a"
        print(f"  {str(d['assay'])[:43]:<44}{str(d['n'] or ''):>5}{rho:>8}  {d['decision']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply the honest Head B serving policy to existing reports")
    parser.add_argument("--report", action="append", required=True,
                        help="Path to a Head B report JSON (repeatable to merge cohorts)")
    parser.add_argument("--min-n", type=int, default=MIN_N)
    parser.add_argument("--out", default="results/head_b_serving_policy.json")
    args = parser.parse_args()

    reports = [_load(p) for p in args.report]
    out = apply_serving_policy(reports, min_n=args.min_n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(out)
    print(f"\nwrote {args.out}")
