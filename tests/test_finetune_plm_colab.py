from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.finetune_plm_colab import prepare_rows


VH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGYFDYWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVSTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPYTFGQGTKLEIK"


class TestFinetunePlmColab(unittest.TestCase):
    def test_preparation_honors_arm_pair_overlap_and_numreact_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "poly.csv"
            rows = [
                {"source": "tessier_lib1", "vh": VH, "vl": VL, "label": 0, "label_type": "polyreactivity_highlow"},
                {"source": "tessier_lib1", "vh": VH + "A", "vl": VL + "A", "label": 1, "label_type": "polyreactivity_highlow"},
                {"source": "jain2017_psr", "vh": VH + "C", "vl": VL + "C", "label": 0.2, "label_type": "psr_continuous"},
                {"source": "jain2017_psr", "vh": VH + "D", "vl": VL + "D", "label": 0.8, "label_type": "psr_continuous"},
                {"source": "aims_mouse", "vh": VH, "vl": VL + "E", "label": 1, "label_type": "yn_binary"},
                {"source": "aims_flu", "vh": VH + "F", "vl": VL + "F", "label": 1, "label_type": "num_react"},
                {"source": "aims_flu", "vh": VH + "G", "vl": VL + "G", "label": 2, "label_type": "num_react"},
                {"source": "shehata2019", "vh": VH + "H", "vl": VL + "H", "label": 0.1, "label_type": "psr_continuous"},
                {"source": "shehata2019", "vh": VH + "I", "vl": VL + "I", "label": 0.9, "label_type": "psr_continuous"},
            ]
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("source", "vh", "vl", "label", "label_type"))
                writer.writeheader()
                writer.writerows(rows)

            library, holdouts, report = prepare_rows(
                source, 10, 42, arm="library_only", max_per_cluster_class=10
            )
            self.assertEqual([row["vh"] for row in library], [VH + "A"])
            self.assertEqual([row["target"] for row in holdouts["aims_flu"]], [0, 1])
            self.assertEqual(report["shared_holdout_rows_removed"], 1)

            diverse, _, diverse_report = prepare_rows(
                source, 10, 42, arm="diverse_only", max_per_cluster_class=10
            )
            self.assertEqual({row["source"] for row in diverse}, {"jain2017_psr"})
            self.assertEqual(diverse_report["training_sources"], ["jain2017_psr"])


if __name__ == "__main__":
    unittest.main()
