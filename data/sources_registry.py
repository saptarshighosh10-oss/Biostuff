"""Central catalog of every data source with its CORRECT role.

The baseline audit (docs/audits/2026-07-13) flagged pooling every source as
"working" as a High-severity error. This registry prevents that by pinning each
source to a role and routing it accordingly:

  - "supervised"     → antibody rows with a real assay endpoint. Head B trains on these.
  - "auxiliary"      → aggregation-adjacent non-antibody signal (peptide nucleation,
                       computed A3D). Feeds derived features, never the supervised target.
  - "background_ood" → unlabeled antibody/protein reference (natural repertoires,
                       clinical-stage mAbs, structural deposits). Used for OOD-distance
                       and abstention, NOT as supervised labels.

Adding a source = one `SourceSpec` entry. Loaders are imported lazily and every
one is guarded: a missing module (loader not written yet), a missing local file,
or a network failure skips that source rather than aborting the cohort. So this
runs today (FLAb only) and grows as data lands, with zero code change here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SourceSpec:
    name: str
    role: str                      # supervised | auxiliary | background_ood
    is_antibody: bool
    loader: Callable[[], list[dict]]  # returns row dicts (empty on any failure)


# --- per-source loader adapters (all return list[dict], never raise) ----------

def _guard(fn: Callable[[], list[dict]]) -> list[dict]:
    """Run a loader, swallow the expected failure modes (module not written yet,
    local file absent, network down) and return []."""
    try:
        return list(fn() or [])
    except (ImportError, FileNotFoundError) as e:
        print(f"  registry: skipped ({type(e).__name__}: {e})")
        return []
    except Exception as e:  # ponytail: network/parse failures are non-fatal for a source
        print(f"  registry: skipped ({type(e).__name__}: {e})")
        return []


def _load_flab() -> list[dict]:
    from data.flab import load_all_developability_data
    return load_all_developability_data()


def _load_gdpa() -> list[dict]:
    from data.gdpa import load_gdpa
    rows: list[dict] = []
    for ds in ("gdpa1", "gdpa2"):
        try:
            rows.extend(load_gdpa(ds))
        except FileNotFoundError:
            pass
    return rows


def _load_oas() -> list[dict]:
    from data.oas import load_oas
    return load_oas()


def _load_thera() -> list[dict]:
    from data.thera_sabdab import load_thera_sabdab
    return load_thera_sabdab()


def _load_abdev() -> list[dict]:
    from data.abdev import load_abdev_negatives
    return load_abdev_negatives()


def _load_antiref() -> list[dict]:
    from data.antiref import load_antiref_negatives
    return load_antiref_negatives()


def _load_sabdab() -> list[dict]:
    from data.sabdab import load_sabdab_negatives
    return load_sabdab_negatives()


def _load_canya() -> list[dict]:
    from data.canya import load_canya_data
    failures, working = load_canya_data()
    return list(failures) + list(working)


def _load_figshare() -> list[dict]:
    from data.figshare_agg import load_figshare_agg_data
    failures, working = load_figshare_agg_data()
    return list(failures) + list(working)


REGISTRY: list[SourceSpec] = [
    # supervised antibody developability (the Head B training target)
    SourceSpec("flab", "supervised", True, _load_flab),
    SourceSpec("gdpa", "supervised", True, _load_gdpa),
    # background / OOD reference (natural + clinical antibodies, structural)
    SourceSpec("oas", "background_ood", True, _load_oas),
    SourceSpec("thera_sabdab", "background_ood", True, _load_thera),
    SourceSpec("abdev", "background_ood", True, _load_abdev),
    SourceSpec("antiref", "background_ood", True, _load_antiref),
    SourceSpec("sabdab", "background_ood", True, _load_sabdab),
    # aggregation-adjacent auxiliary signal (non-antibody)
    SourceSpec("canya", "auxiliary", False, _load_canya),
    SourceSpec("figshare_agg", "auxiliary", False, _load_figshare),
]


def assemble(include: set[str] | None = None) -> dict[str, list[dict]]:
    """Load every registered source (or the `include` subset) and bucket rows by
    role. Returns {"supervised": [...], "auxiliary": [...], "background_ood": [...]}.
    Each row is stamped with its `source` and `supervision_status`."""
    buckets: dict[str, list[dict]] = {"supervised": [], "auxiliary": [], "background_ood": []}
    for spec in REGISTRY:
        if include is not None and spec.name not in include:
            continue
        rows = _guard(spec.loader)
        for r in rows:
            r.setdefault("source", spec.name)
            r.setdefault("supervision_status", spec.role)
        if rows:
            print(f"  registry: {spec.name} → {len(rows)} rows ({spec.role})")
        buckets[spec.role].extend(rows)
    return buckets
