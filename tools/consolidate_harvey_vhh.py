"""Append the public Harvey VHH PSR dataset without changing existing rows.

The source CSV has one full VHH sequence per row and binary PSR labels where
1 means high polyreactivity.  VHH antibodies have no separate light chain, so
``vl`` is intentionally empty.
"""

from __future__ import annotations

import csv
from pathlib import Path


AA = set("ACDEFGHIKLMNPQRSTVWY")
SOURCE = "harvey2022_vhh"
INPUT = Path("/tmp/harvey_vhh.csv")
OUTPUT = Path.home() / "Downloads/antibody_sources/polyreactivity_dataset.csv"


def run() -> dict[str, int]:
    if not INPUT.exists():
        raise FileNotFoundError(f"Download missing: {INPUT}")
    with OUTPUT.open(newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    seen = {(row["source"], row["vh"].strip().upper(), row["vl"].strip().upper())
            for row in existing}
    added: list[dict[str, str]] = []
    invalid = duplicates = 0
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            vh = (row.get("sequence") or "").strip().upper()
            try:
                label = int(row.get("label") or "")
            except ValueError:
                label = -1
            if len(vh) < 20 or set(vh) - AA or label not in (0, 1):
                invalid += 1
                continue
            key = (SOURCE, vh, "")
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            added.append({"source": SOURCE, "vh": vh, "vl": "", "label": str(label),
                          "label_type": "psr_binary_vhh"})
    if added:
        with OUTPUT.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=("source", "vh", "vl", "label", "label_type")).writerows(added)
    return {"added": len(added), "duplicates": duplicates, "invalid": invalid}


if __name__ == "__main__":
    print(run())
