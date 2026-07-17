from __future__ import annotations

import hashlib
import unittest

from data.sequence_reconstruction import reconstruct_verified_pair


PARENT_VH = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
PARENT_VL = "YWVTSRQPNMLKIHGFEDCAYWVTSRQPNMLKIHGFEDCA"
EXPECTED_VH = PARENT_VH[:22] + "W" + PARENT_VH[23:]


def valid_row() -> dict[str, str]:
    return {
        "parent_vh": PARENT_VH,
        "parent_vl": PARENT_VL,
        "vh_mutations": "D23W",
        "vl_mutations": "",
        "expected_vh": EXPECTED_VH,
        "expected_vl_sha256": hashlib.sha256(PARENT_VL.encode("ascii")).hexdigest(),
        "parent_mutation_reference": "supplementary-table-2",
        "verification_reference": "genbank:AB123456",
    }


class TestSequenceReconstruction(unittest.TestCase):
    def test_reconstructs_substitution_and_records_all_checks(self) -> None:
        vh, vl, audit = reconstruct_verified_pair(valid_row())
        self.assertEqual(vh, EXPECTED_VH)
        self.assertEqual(vl, PARENT_VL)
        self.assertTrue(audit["chains"]["vh"]["parent_residue_check"])
        self.assertTrue(audit["chains"]["vh"]["round_trip_diff_check"])
        self.assertEqual(audit["chains"]["vh"]["independent_sequence_check"], "full_sequence")
        self.assertEqual(audit["chains"]["vl"]["independent_sequence_check"], "sha256")

    def test_rejects_wrong_parent_residue(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match parent"):
            reconstruct_verified_pair({**valid_row(), "vh_mutations": "C23G"})

    def test_rejects_nonindependent_reference(self) -> None:
        row = valid_row()
        row["verification_reference"] = row["parent_mutation_reference"]
        with self.assertRaisesRegex(ValueError, "distinct provenance"):
            reconstruct_verified_pair(row)

    def test_rejects_missing_or_wrong_independent_sequence(self) -> None:
        row = valid_row()
        row.pop("expected_vh")
        with self.assertRaisesRegex(ValueError, "independent expected VH"):
            reconstruct_verified_pair(row)
        with self.assertRaisesRegex(ValueError, "independent VH sequence"):
            reconstruct_verified_pair({**valid_row(), "expected_vh": PARENT_VH})


if __name__ == "__main__":
    unittest.main()
