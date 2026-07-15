"""Export the unique AC-SINS antibody sequences that need folding.

Writes a CSV (seq_id, sequence) of every unique sequence in the AC-SINS cohort.
Upload this to the Colab folding notebook (tools/fold_structures_colab.py) to
generate structures + structural aggregation features on a GPU.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from model.head_b_acsins_unified import _acsins_rows


def main(out: str = "results/acsins_sequences_to_fold.csv") -> None:
    rows = _acsins_rows(True)
    seen: dict[str, str] = {}
    for r in rows:
        seq = (r.get("variant_sequence") or r.get("sequence") or "").strip().upper()
        if len(seq) >= 20:
            seen.setdefault(hashlib.sha256(seq.encode()).hexdigest()[:16], seq)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "sequence"])
        for sid, seq in seen.items():
            w.writerow([sid, seq])
    print(f"wrote {len(seen)} unique sequences to {out}")


if __name__ == "__main__":
    main()
