from __future__ import annotations

import unittest

from data.variant_quality import validate_rows


class TestVariantQuality(unittest.TestCase):
    def test_parent_mismatch_is_quarantined(self) -> None:
        valid, quarantine, quality = validate_rows([
            {
                "variant_sequence": "VCDEFGHIKLMN",
                "wild_type_sequence": "ACDEFGHIKLMN",
                "mutations": [(0, "A", "V")],
                "label": "working",
                "group_id": "assay_v1",
            },
            {
                "variant_sequence": "XCDEFGHIKLMN",
                "wild_type_sequence": "ACDEFGHIKLMN",
                "mutations": [(0, "A", "X")],
                "label": "working",
                "group_id": "assay_v1",
            },
        ])
        self.assertEqual(len(valid), 1)
        self.assertEqual(quarantine[0]["quarantine_reason"], "invalid_variant_sequence")
        self.assertEqual(quality["accepted"], 1)


if __name__ == "__main__":
    unittest.main()
