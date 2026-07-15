"""Export separate heavy/light chains for proper paired antibody folding (v2).

Unlike export_acsins_sequences (which dumped the combined sequence), this writes each
antibody's VH and VL chains separately so IgFold can fold them as a real paired Fv —
giving a correct 3D interface, which the crude single-chain fold (v1) did not.

Output columns: seq_id, mode, H, L
  mode = "paired"  (VH + VL)      -> IgFold folds {"H": H, "L": L}
       = "single"  (VH/VHH only)  -> IgFold folds {"H": H}
seq_id = same hash of variant_sequence used elsewhere, so features map back to model rows.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from model.head_b_acsins_unified import _acsins_rows


def _sid(variant: str) -> str:
    return hashlib.sha256(variant.strip().upper().encode()).hexdigest()[:16]


def main(out: str = "results/paired_chains_to_fold.csv") -> None:
    rows = _acsins_rows(True)
    seen: dict[str, dict] = {}
    for r in rows:
        var = (r.get("variant_sequence") or "").strip().upper()
        if not var or var in seen:
            continue
        vh = (r.get("vh_sequence") or "").strip().upper()
        vl = (r.get("vl_sequence") or "").strip().upper()
        if vh and vl:
            seen[var] = {"seq_id": _sid(var), "mode": "paired", "H": vh, "L": vl}
        elif vh:
            seen[var] = {"seq_id": _sid(var), "mode": "single", "H": vh, "L": ""}
        elif len(var) >= 20:
            # no split chains available — fall back to folding the combined seq as one chain
            seen[var] = {"seq_id": _sid(var), "mode": "single", "H": var, "L": ""}

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq_id", "mode", "H", "L"])
        w.writeheader()
        for rec in seen.values():
            w.writerow(rec)
    paired = sum(1 for v in seen.values() if v["mode"] == "paired")
    print(f"wrote {len(seen)} antibodies to {out}  ({paired} paired, {len(seen)-paired} single)")


if __name__ == "__main__":
    main()
