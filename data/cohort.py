"""Antibody cohort assembler — the ingest glue.

Combines every antibody developability source into one list of training rows
ready for `model/head_b_gbm.py`:

    FLAb (all assay subdirs)  +  GDPa1/GDPa2 (when local CSVs present)
      → attach VH/VL pair_id                     (data.pairs)
      → (optional) conflict/exclusion ledgers    (model.preprocess, design §18)
      → rows ready for per-assay grouped-CV eval

Rows are plain dicts (the shape `head_b_gbm` consumes) — stdlib only, no
numpy/network. GDPa loaders raise `FileNotFoundError` when their CSV is absent;
that source is skipped rather than fatal, so a FLAb-only cohort runs today.
"""

from __future__ import annotations

from data.contract import sha256_hex


_HEAD_B_METRIC_TOKENS = (
    "ac-sins", "ac_sins", "hic", "sec", "psr", "polyreactivity",
    "aggregation", "self_association", "hydrophobicity",
)


def is_head_b_aggregation_row(row: dict) -> bool:
    """Return whether a supervised row measures the aggregation target family."""
    family = str(row.get("assay_family", "")).strip().lower()
    metric = str(row.get("assay_metric", "")).strip().lower()
    return family in {"aggregation", "self_association", "hydrophobicity", "polyreactivity"} or any(
        token in metric for token in _HEAD_B_METRIC_TOKENS
    )


def _attach_pairing(rows: list[dict]) -> list[dict]:
    """Set a deterministic `pair_id` when a row carries both VH and VL; otherwise
    flag it single-chain. GDPa rows already hold both chains; FLAb rows are
    single-sequence, so they route as single_chain_only."""
    for r in rows:
        vh = (r.get("vh_sequence") or "").strip()
        vl = (r.get("vl_sequence") or "").strip()
        if vh and vl:
            r["pair_id"] = sha256_hex(f"{vh.upper()}|{vl.upper()}")[:16]
            r.setdefault("molecule_group_id", f"pair:{r['pair_id']}")
            r.setdefault("feature_flags", [])
        else:
            r["pair_id"] = None
            seq = (r.get("variant_sequence") or r.get("sequence") or "").strip().upper()
            r.setdefault("molecule_group_id", f"sequence:{sha256_hex(seq)[:24]}" if seq else None)
            flags = r.setdefault("feature_flags", [])
            if "single_chain_only" not in flags:
                flags.append("single_chain_only")
    return rows


def build_antibody_cohort(include_gdpa: bool = True, gdpa_dir: str = "data/external/gdpa") -> list[dict]:
    """Assemble the full antibody developability cohort as head_b_gbm-ready rows.

    FLAb is always loaded (local snapshot). GDPa1/GDPa2 are added when their CSVs
    exist under `gdpa_dir`; a missing GDPa file is skipped (logged), never fatal —
    so this runs before benchmark access is granted."""
    from data.flab import load_all_developability_data

    rows: list[dict] = list(load_all_developability_data())

    if include_gdpa:
        from data.gdpa import load_gdpa
        for dataset in ("gdpa1", "gdpa2"):
            try:
                gdpa_rows = load_gdpa(dataset, local_dir=gdpa_dir)
                rows.extend(gdpa_rows)
                print(f"  cohort: +{len(gdpa_rows)} rows from {dataset}")
            except FileNotFoundError as e:
                print(f"  cohort: {dataset} skipped ({e})")

    _attach_pairing(rows)
    rows = [r for r in rows if is_head_b_aggregation_row(r)]
    print(f"  cohort: {len(rows)} total rows "
          f"({sum(1 for r in rows if r.get('pair_id'))} paired, "
          f"{sum(1 for r in rows if not r.get('pair_id'))} single-chain)")
    return rows


def build_full_cohort(include: set[str] | None = None) -> dict[str, list[dict]]:
    """Every registered source, bucketed by role (design-correct scaling):

        {"supervised":     antibody rows with real assay endpoints (Head B target),
         "auxiliary":      aggregation-adjacent non-antibody signal (derived features),
         "background_ood": unlabeled antibody/protein reference (OOD + abstention)}

    VH/VL pairing is attached to the antibody buckets. Missing sources (loader not
    written, file absent, network down) are skipped, so this grows as data lands
    without changing callers. Head B still trains ONLY on the supervised bucket —
    background/auxiliary never contaminate the supervised target."""
    from data.sources_registry import assemble

    buckets = assemble(include=include)
    buckets["supervised"] = [r for r in buckets["supervised"] if is_head_b_aggregation_row(r)]
    _attach_pairing(buckets["supervised"])
    _attach_pairing(buckets["background_ood"])
    print(f"  cohort: supervised={len(buckets['supervised'])} "
          f"auxiliary={len(buckets['auxiliary'])} "
          f"background_ood={len(buckets['background_ood'])}")
    return buckets


def emit_ledgers(rows: list[dict], out_dir: str = "results") -> dict:
    """Optional Plan C audit artifacts (design §18): normalize rows and write
    conflicts.tsv / exclusions.tsv. Returns counts. Skips gracefully if a row
    can't be normalized (e.g. a source not in contract.SOURCE_ROLES) — the audit
    is best-effort and never blocks cohort assembly.

    Note: pair_id and a vh/vl chain_id can't coexist on a NormalizedRow, so
    chain_id is cleared here before normalization (contract invariant)."""
    from pathlib import Path
    from data.normalize import normalize_rows
    from model.preprocess import (
        build_conflict_and_exclusion_ledgers,
        write_conflicts_tsv,
        write_exclusions_tsv,
    )

    prepared = []
    for r in rows:
        r = dict(r)
        if r.get("pair_id") and r.get("chain_id"):
            r["chain_id"] = None  # contract forbids pair_id + chain_id together
        prepared.append(r)

    normalized: list = []
    skipped = 0
    by_source: dict[str, list[dict]] = {}
    for r in prepared:
        by_source.setdefault(r.get("source", "unknown"), []).append(r)
    for source, group in by_source.items():
        try:
            norm, n_skip = normalize_rows(group, source)
            normalized.extend(norm)
            skipped += n_skip
        except Exception as e:  # ponytail: audit is best-effort, never fatal
            print(f"  ledger: source '{source}' not normalizable ({e}); skipped")

    kept, conflicts, exclusions = build_conflict_and_exclusion_ledgers(normalized)
    out = Path(out_dir)
    write_conflicts_tsv(conflicts, out / "conflicts.tsv")
    write_exclusions_tsv(exclusions, out / "exclusions.tsv")
    counts = {"normalized": len(normalized), "skipped": skipped,
              "kept": len(kept), "conflicts": len(conflicts), "exclusions": len(exclusions)}
    print(f"  ledger: {counts}")
    return counts
