"""Focused tests for Plan C identity graph and component derivation."""

from __future__ import annotations

import unittest

from data import identity_graph
from data.contract import NormalizedRow


def _normalized_row(
    *,
    record_id: str,
    sequence: str,
    pair_id: str | None,
    chain_id: str | None = None,
    vh_sequence: str | None = None,
    vl_sequence: str | None = None,
    molecule_id: str = "mol",
) -> NormalizedRow:
    return NormalizedRow(
        record_id=record_id,
        source="flab",
        study_id="study",
        campaign_id="campaign",
        molecule_id=molecule_id,
        pair_id=pair_id,
        chain_id=chain_id,
        sequence=sequence,
        vh_sequence=vh_sequence,
        vl_sequence=vl_sequence,
        assay_id="assay",
        assay_metric="HIC_RT",
        endpoint_direction="higher_bad",
        endpoint_value=1.0,
        endpoint_unit="ms",
        target_value=1.0,
        supervision_status="supervised",
        binary_label=1,
        label_threshold=0.5,
        group_study="study",
        group_molecule=molecule_id,
        group_campaign="campaign",
        sequence_hash="x" + record_id,
        exclusions=[],
        source_url="",
        source_row_hash="row-" + record_id,
        feature_flags=[],
    )


class TestIdentitySimilarity(unittest.TestCase):
    def test_is_similar_exact_sequences(self) -> None:
        self.assertTrue(identity_graph.is_similar("ACDEFG", "ACDEFG"))

    def test_is_similar_requires_length_ratio_guard(self) -> None:
        long = "ACDEFGHIKLMNPQRSTVWYAC"
        short = "ACDEFGHIKL"
        self.assertFalse(identity_graph.is_similar(long, short))

        near_same_length = "ACDEFGHIKLMNP"
        similar = "ACDEFGHIKLMPP"
        self.assertTrue(identity_graph.is_similar(near_same_length, similar))

    def test_is_similar_falls_back_to_alignment_ratio(self) -> None:
        lhs = "ACDEFGHIKLMNPQRSTVWYA"
        rhs = "ACDEFGHIKLMNPQRSTVWYC"
        self.assertTrue(identity_graph.is_similar(lhs, rhs))

    def test_invalid_sequence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            identity_graph.is_similar("ACD*EFG", "ACDEFG")


class TestIdentityComponents(unittest.TestCase):
    def test_connected_pairs_form_single_component_via_transitivity(self) -> None:
        rows = [
            _normalized_row(
                record_id="r1",
                pair_id="pair-1",
                sequence="ACDEFGHIKLMNPQRSTVWYA",
                vh_sequence="ACDEFGHIKL",
                vl_sequence="MNPQRSTVWY",
            ),
            _normalized_row(
                record_id="r2",
                pair_id="pair-2",
                sequence="ACDEFGHIKLMNPQRSTVWYB",
                vh_sequence="ACDEFGHIKL",
                vl_sequence="MNPQRSTVWY",
            ),
            _normalized_row(
                record_id="r3",
                pair_id="pair-3",
                sequence="ACDEFGHIKLMNPQRSTVWA",
                vh_sequence="TTTTT",
                vl_sequence="MNPQRSTVWY",
            ),
        ]
        components = identity_graph.build_identity_components(rows)
        self.assertEqual(len(set(components.values())), 1)
        self.assertEqual(components["r1"], components["r2"])
        self.assertEqual(components["r2"], components["r3"])

    def test_single_chain_nodes_use_chain_and_molecule_key(self) -> None:
        rows = [
            _normalized_row(
                record_id="vh-1",
                pair_id=None,
                chain_id="vh",
                molecule_id="mol",
                sequence="ACDEFGHIKLMNPQRSTVWY",
            ),
            _normalized_row(
                record_id="vh-2",
                pair_id=None,
                chain_id="vh",
                molecule_id="mol",
                sequence="ACDEFGHIKLMNPQRSTVVA",
            ),
            _normalized_row(
                record_id="vl-1",
                pair_id=None,
                chain_id="vl",
                molecule_id="mol",
                sequence="ACDEFGHIKLMNPQRSTVWY",
            ),
        ]
        components = identity_graph.build_identity_components(rows)
        self.assertNotEqual(components["vh-1"], components["vl-1"])
        self.assertEqual(components["vh-1"], components["vh-2"])

    def test_component_representative_is_stable(self) -> None:
        rows = [
            _normalized_row(
                record_id="a",
                pair_id="b",
                sequence="ACDEFGHIKL",
            ),
            _normalized_row(
                record_id="b",
                pair_id="a",
                sequence="ACDEFGHIKL",  # exact copy in different pair key
            ),
            _normalized_row(
                record_id="c",
                pair_id="c",
                sequence="TTTTTTTTTT",
            ),
        ]
        components = identity_graph.build_identity_components(rows)
        self.assertEqual(components["a"], "pair:a")
        self.assertEqual(components["b"], "pair:a")
        self.assertEqual(components["c"], "pair:c")

    def test_assign_identity_components_alias_is_stable(self) -> None:
        rows = [
            _normalized_row(record_id="x", pair_id="x", sequence="ACDEFGHIKL"),
            _normalized_row(record_id="y", pair_id="y", sequence="ACDEFGHIKA"),
        ]
        self.assertEqual(
            identity_graph.assign_identity_components(rows),
            identity_graph.build_identity_components(rows),
        )


if __name__ == "__main__":
    unittest.main()
