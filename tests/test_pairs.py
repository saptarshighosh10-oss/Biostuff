"""Tests for VH/VL chain typing and deterministic pairing (data/pairs.py)."""

from __future__ import annotations

import unittest

from data import pairs
from data.contract import sha256_hex

# Trastuzumab-like variable-domain stubs (real FR4 J motifs).
VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"
    "RFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSR"
    "SGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


class PairChainsTest(unittest.TestCase):
    def test_one_vh_one_vl_get_shared_deterministic_pair_id(self):
        rows = [
            {"molecule_id": "m1", "chain_id": "vh", "sequence": VH},
            {"molecule_id": "m1", "chain_id": "vl", "sequence": VL},
        ]
        out = pairs.pair_chains(rows)
        expected = sha256_hex(f"{VH}|{VL}")[:16]

        self.assertEqual(out[0]["pair_id"], expected)
        self.assertEqual(out[1]["pair_id"], out[0]["pair_id"])
        for r in out:
            self.assertEqual(r["vh_sequence"], VH)
            self.assertEqual(r["vl_sequence"], VL)
            self.assertNotIn("single_chain_only", r["feature_flags"])

    def test_single_chain_molecule_is_unpaired_and_flagged(self):
        rows = [{"molecule_id": "m2", "chain_id": "vh", "sequence": VH}]
        out = pairs.pair_chains(rows)
        self.assertIsNone(out[0]["pair_id"])
        self.assertIn("single_chain_only", out[0]["feature_flags"])

    def test_pairing_never_crosses_molecules(self):
        rows = [
            {"molecule_id": "a", "chain_id": "vh", "sequence": VH},
            {"molecule_id": "b", "chain_id": "vl", "sequence": VL},
        ]
        out = pairs.pair_chains(rows)
        self.assertIsNone(out[0]["pair_id"])
        self.assertIsNone(out[1]["pair_id"])


class InvariantTest(unittest.TestCase):
    def test_two_vh_same_molecule_raises(self):
        rows = [
            {"molecule_id": "bad", "chain_id": "vh", "sequence": VH},
            {"molecule_id": "bad", "chain_id": "vh", "sequence": VH},
        ]
        out = pairs.pair_chains(rows)  # stays unpaired
        with self.assertRaises(ValueError):
            pairs.assert_pairing_invariants(out)

    def test_pair_id_without_both_chains_raises(self):
        rows = [{"molecule_id": "x", "chain_id": "vh", "pair_id": "deadbeef", "vh_sequence": VH}]
        with self.assertRaises(ValueError):
            pairs.assert_pairing_invariants(rows)

    def test_paired_rows_pass_invariants(self):
        out = pairs.pair_chains(
            [
                {"molecule_id": "ok", "chain_id": "vh", "sequence": VH},
                {"molecule_id": "ok", "chain_id": "vl", "sequence": VL},
            ]
        )
        pairs.assert_pairing_invariants(out)  # must not raise


class ClassifyChainTest(unittest.TestCase):
    ALLOWED = {"vh", "vl", "unknown"}

    def test_returns_allowed_label_and_does_not_crash(self):
        self.assertIn(pairs.classify_chain(VH), self.ALLOWED)
        self.assertIn(pairs.classify_chain(VL), self.ALLOWED)
        self.assertIn(pairs.classify_chain(""), self.ALLOWED)
        self.assertIn(pairs.classify_chain("NOTASEQUENCE"), self.ALLOWED)

    def test_plausible_labels_on_stubs(self):
        # Tolerant: stubs carry canonical FR4 motifs, so expect the right call.
        self.assertEqual(pairs.classify_chain(VH), "vh")
        self.assertEqual(pairs.classify_chain(VL), "vl")


if __name__ == "__main__":
    unittest.main()
