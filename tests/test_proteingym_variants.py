from __future__ import annotations

import unittest

from data.proteingym import _parse_assay
from data.build_proteingym_partition import split_by_protein


class TestProteinGymVariants(unittest.TestCase):
    def test_single_and_multi_mutants_keep_parent_metadata(self) -> None:
        text = "mutant,DMS_score,DMS_score_bin\nA1V,-1,0\nA1V:D3C,1,1\n" \
               "A1V:F5Y,1,1\nA1V:G6H,1,1\nA1V:H7I,1,1\nA1V:K9R,1,1\n" \
               "A1V:L10M,1,1\nA1V:M11N,1,1\n"
        failures, working = _parse_assay(
            text, percentile=0.25, max_keep=20, group_id="protein_v1",
            target_sequence="ACDEFGHIKLMN", keep_all=True,
        )
        self.assertEqual(failures[0]["variant_type"], "single_mutant")
        self.assertEqual(working[0]["variant_type"], "multi_mutant")
        self.assertEqual(working[0]["mutation_count"], 2)
        self.assertEqual(working[0]["wild_type_sequence"], "ACDEFGHIKLMN")

    def test_split_is_protein_disjoint_not_assay_disjoint(self) -> None:
        rows = [
            {"protein_group_id": "uniprot:P1", "group_id": "uniprot:P1", "protein_group_fallback": False, "variant_sequence": "A" * 10, "label": "confirmed_failure"},
            {"protein_group_id": "uniprot:P1", "group_id": "uniprot:P1", "protein_group_fallback": False, "variant_sequence": "C" * 10, "label": "working"},
            {"protein_group_id": "uniprot:P2", "group_id": "uniprot:P2", "protein_group_fallback": False, "variant_sequence": "D" * 10, "label": "confirmed_failure"},
            {"protein_group_id": "uniprot:P2", "group_id": "uniprot:P2", "protein_group_fallback": False, "variant_sequence": "E" * 10, "label": "working"},
        ]
        train, test, train_groups, test_groups, fallback_count = split_by_protein(rows, train_min=2, test_min=2)
        self.assertFalse(set(train_groups) & set(test_groups))
        self.assertFalse({row["protein_group_id"] for row in train} & {row["protein_group_id"] for row in test})
        self.assertEqual(fallback_count, 0)


if __name__ == "__main__":
    unittest.main()
