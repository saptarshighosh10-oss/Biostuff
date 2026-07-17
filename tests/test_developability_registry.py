from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from data.developability_registry import RegistryInput, build_registry


VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQY"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestDevelopabilityRegistry(unittest.TestCase):
    def test_routes_tiers_and_quarantines_conflicts_without_fabricating_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poly = root / "poly.csv"
            cdr = root / "cdr.csv"
            database = root / "registry.sqlite"
            write_csv(
                poly,
                ["source", "vh", "vl", "label", "label_type"],
                [
                    {"source": "tessier_lib1", "vh": VH, "vl": VL, "label": "0", "label_type": "polyreactivity_highlow"},
                    {"source": "tessier_lib1", "vh": VH, "vl": VL, "label": "1", "label_type": "polyreactivity_highlow"},
                    {"source": "harvey2022_vhh", "vh": VH + "A", "vl": "", "label": "1", "label_type": "psr_binary_vhh"},
                    {"source": "bad", "vh": "NOT_A_SEQUENCE", "vl": VL, "label": "1", "label_type": "psr_continuous"},
                ],
            )
            write_csv(
                cdr,
                ["source", "cdr_h1", "cdr_h2", "cdr_h3", "cdr_l1", "cdr_l2", "cdr_l3", "label", "label_type"],
                [{
                    "source": "aims_nat_hiv_cdr",
                    "cdr_h1": "GFTFSSYA",
                    "cdr_h2": "ISGSGGST",
                    "cdr_h3": "ARDRGGYFDY",
                    "cdr_l1": "RASQDVST",
                    "cdr_l2": "SAS",
                    "cdr_l3": "QQYNSYPYT",
                    "label": "3",
                    "label_type": "num_react",
                }],
            )

            report = build_registry(
                database,
                [
                    RegistryInput(poly, "polyreactivity", "sequence_table"),
                    RegistryInput(cdr, "polyreactivity", "cdr_table"),
                ],
            )

            self.assertEqual(report["input_rows"], 5)
            self.assertEqual(report["rejected_rows"], 1)
            self.assertEqual(report["quarantined_conflict_rows"], 2)
            self.assertEqual(report["accepted_rows"], 2)
            self.assertEqual(report["tier_counts"]["single_chain_supervised"], 1)
            self.assertEqual(report["tier_counts"]["cdr_supervised"], 1)

            with closing(sqlite3.connect(database)) as connection:
                harvey = connection.execute(
                    "SELECT supervision_tier, vl, split_group FROM records WHERE source = 'harvey2022_vhh'"
                ).fetchone()
                self.assertEqual(harvey[0], "single_chain_supervised")
                self.assertIsNone(harvey[1])
                self.assertEqual(harvey[2], "source:harvey2022_vhh")
                reasons = connection.execute("SELECT reason FROM rejections").fetchall()
                self.assertEqual(reasons, [("invalid amino-acid sequence",)])

    def test_build_is_deterministic_and_replaces_the_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "hic.csv"
            database = root / "registry.sqlite"
            write_csv(
                source,
                ["source", "vh", "vl", "label", "label_type"],
                [{"source": "shehata2019", "vh": VH, "vl": VL, "label": "8.5", "label_type": "hic_retention_min"}],
            )
            spec = [RegistryInput(source, "hydrophobicity", "sequence_table")]
            model_row = {
                "source": "gdpa",
                "study_id": "gdpa1",
                "vh_sequence": VH + "C",
                "vl_sequence": VL + "C",
                "variant_sequence": VH + "C",
                "endpoint_value": 12.5,
                "endpoint_unit": "nm",
                "endpoint_direction": "higher_bad",
                "assay_metric": "AC-SINS_pH6.0",
                "assay_id": "acsins_ph6",
                "assay_family": "self_association",
                "ph": 6.0,
            }
            second_construct = {
                **model_row,
                "molecule_id": "same_vh_different_construct",
                "endpoint_value": 14.0,
            }
            first_construct = {**model_row, "molecule_id": "first_construct"}
            first = build_registry(database, spec, model_rows=[first_construct, second_construct])
            with closing(sqlite3.connect(database)) as connection:
                first_ids = connection.execute("SELECT record_id FROM records").fetchall()
            second = build_registry(database, spec, model_rows=[first_construct, second_construct])
            with closing(sqlite3.connect(database)) as connection:
                second_ids = connection.execute("SELECT record_id FROM records").fetchall()
            self.assertEqual(first["registry_hash"], second["registry_hash"])
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(second["source_counts"], {"gdpa1": 2, "shehata2019": 1})
            with closing(sqlite3.connect(database)) as connection:
                aggregation = connection.execute(
                    "SELECT property, endpoint_unit, assay_context_json FROM records "
                    "WHERE source = 'gdpa1' ORDER BY endpoint_value LIMIT 1"
                ).fetchone()
            self.assertEqual(aggregation[0], "aggregation")
            self.assertEqual(aggregation[1], "nm")
            self.assertEqual(
                aggregation[2],
                '{"molecule_id":"first_construct","ph":6.0}',
            )


if __name__ == "__main__":
    unittest.main()
