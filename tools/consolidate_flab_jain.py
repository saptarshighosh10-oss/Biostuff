"""Append four measured Jain clinical-antibody tables from the open FLAb repo.

The source tables contain full paired VH/VL amino-acid sequences and continuous
measurements.  HIC and PSR are kept in their respective local datasets; no
unrelated FLAb endpoints are extracted.  Existing rows are never altered.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = "https://raw.githubusercontent.com/Graylab/FLAb/main/"
SOURCES = (
    {
        "path": "data/aggregation/jain2017biophyscial_HICRT.csv",
        "destination": "hydrophobicity_dataset.csv",
        "source": "jain2017_hic",
        "label_column": "HIC Retention Time (Min)a",
        "label_type": "hic_retention_min",
    },
    {
        "path": "data/aggregation/jain2024assessment_HIC.csv",
        "destination": "hydrophobicity_dataset.csv",
        "source": "jain2024_hic",
        "label_column": "HIC RT (min)",
        "label_type": "hic_retention_min",
    },
    {
        "path": "data/polyreactivity/jain2017biophysical_PSR.csv",
        "destination": "polyreactivity_dataset.csv",
        "source": "jain2017_psr",
        "label_column": "Poly-Specificity Reagent (PSR) SMP Score (0-1)",
        "label_type": "psr_smp_continuous",
    },
    {
        "path": "data/polyreactivity/jain2024assessment_PSR.csv",
        "destination": "polyreactivity_dataset.csv",
        "source": "jain2024_psr",
        "label_column": "PSR",
        "label_type": "psr_continuous",
    },
)
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
DATA_DIR = Path.home() / "Downloads" / "antibody_sources"
FIELDS = ("source", "vh", "vl", "label", "label_type")


def clean_sequence(value: str | None) -> str | None:
    """Return a standard-amino-acid variable domain, otherwise ``None``."""
    sequence = (value or "").strip().upper()
    return sequence if len(sequence) >= 20 and set(sequence) <= AA else None


def existing_vhs(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["vh"].strip().upper() for row in csv.DictReader(handle)}


def download_rows(relative_path: str) -> list[dict[str, str]]:
    """Download one immutable public CSV table from FLAb."""
    with urlopen(ROOT + relative_path, timeout=30) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def run(dry_run: bool = False) -> list[dict[str, object]]:
    """Validate and append FLAb HIC/PSR observations, reporting real counts."""
    known_by_destination = {
        name: existing_vhs(DATA_DIR / name)
        for name in {spec["destination"] for spec in SOURCES}
    }
    appended_by_destination: dict[str, list[dict[str, object]]] = {
        name: [] for name in known_by_destination
    }
    report: list[dict[str, object]] = []

    for spec in SOURCES:
        raw = download_rows(str(spec["path"]))
        accepted: list[dict[str, object]] = []
        invalid = duplicate = 0
        known = known_by_destination[str(spec["destination"])]
        for row in raw:
            vh, vl = clean_sequence(row.get("heavy")), clean_sequence(row.get("light"))
            try:
                label = float(row[str(spec["label_column"])])
            except (KeyError, TypeError, ValueError):
                label = None
            if vh is None or vl is None or label is None:
                invalid += 1
                continue
            if vh in known:
                duplicate += 1
                continue
            known.add(vh)
            accepted.append({
                "source": spec["source"], "vh": vh, "vl": vl,
                "label": label, "label_type": spec["label_type"],
            })
        appended_by_destination[str(spec["destination"])].extend(accepted)
        values = [float(row["label"]) for row in accepted]
        report.append({
            "source": spec["source"], "raw_rows": len(raw), "added": len(accepted),
            "invalid": invalid, "duplicate_vh": duplicate,
            "range": [min(values), max(values)] if values else None,
        })

    if not dry_run:
        for name, rows in appended_by_destination.items():
            if rows:
                with (DATA_DIR / name).open("a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=FIELDS).writerows(rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate without appending")
    args = parser.parse_args()
    for item in run(args.dry_run):
        print(json.dumps(item, sort_keys=True))
