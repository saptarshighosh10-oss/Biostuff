from __future__ import annotations

import unittest

from data.proteingym import _parse_assay


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


if __name__ == "__main__":
    unittest.main()
