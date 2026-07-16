import unittest

from model.cdr_features import cdr_feature_vector, heavy_cdrs


class CdrFeaturesTest(unittest.TestCase):
    def test_supplied_h3_wins_over_heuristic(self):
        row = {"vh_sequence": "AAAACARDRWGQG", "vh_cdr3": "YYY"}
        self.assertEqual(heavy_cdrs(row)["h3"], "YYY")

    def test_h3_heuristic_and_failure_flag(self):
        self.assertEqual(heavy_cdrs({"vh_sequence": "AAAACARDRWGQG"})["h3"], "ARDR")
        self.assertEqual(cdr_feature_vector({"vh_sequence": "AAAA"}), [0.0] * 5)


if __name__ == "__main__":
    unittest.main()
