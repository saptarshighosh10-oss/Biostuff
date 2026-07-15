"""Phase 0 guard: the training and serving feature paths must produce byte-for-byte
identical vectors for the same (sequence, mutations). This is the check that would
have caught the proteingym_fitness_score train/serve skew."""
from __future__ import annotations

import unittest

from model.features import (
    FEATURE_NAMES,
    HEAD_A_FEATURE_NAMES,
    extract_features,
    extract_from_sequence,
    feature_schema_hash,
    features_to_vector,
    general_fitness_score,
)

# A fixed antibody-like VH stretch (canonical AAs only).
SEQ = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
    "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGRRGYYYYGMDVWGQGTTVTVSS"
)


class TestTrainServeParity(unittest.TestCase):
    def test_serve_path_matches_training_path(self) -> None:
        # Training path: build the candidate exactly as build_training_data does
        # (populate the derived feature with the shared scorer), then featurize.
        train_candidate = {
            "variant_sequence": SEQ,
            "mutations": [],
            "proteingym_fitness_score": general_fitness_score(SEQ),
        }
        train_vec = features_to_vector(extract_features(train_candidate))

        # Serve path: the single entry point every inference caller uses.
        serve_vec = features_to_vector(extract_from_sequence(SEQ, []))

        self.assertEqual(len(train_vec), len(FEATURE_NAMES))
        self.assertEqual(train_vec, serve_vec)

    def test_proteingym_feature_is_not_train_only(self) -> None:
        # Whatever the derived feature is (0.0 without a head, nonzero with one),
        # serve must produce the same value the training path would — never a
        # silent default that differs from training.
        idx = FEATURE_NAMES.index("proteingym_fitness_score")
        serve_vec = features_to_vector(extract_from_sequence(SEQ, []))
        self.assertEqual(serve_vec[idx], general_fitness_score(SEQ))

    def test_head_a_schema_excludes_leaky_features(self) -> None:
        # Head A must not depend on features that are absent/zero when scoring a
        # raw antibody sequence, nor on its own output (circular).
        self.assertNotIn("proteingym_fitness_score", HEAD_A_FEATURE_NAMES)
        for leaky in ("n_mutations", "mean_hydrophobicity_delta", "combined_risk", "plddt_mean"):
            self.assertNotIn(leaky, HEAD_A_FEATURE_NAMES)
        self.assertTrue(set(HEAD_A_FEATURE_NAMES) < set(FEATURE_NAMES))

    def test_schema_hash_is_stable_and_order_sensitive(self) -> None:
        self.assertEqual(feature_schema_hash(FEATURE_NAMES), feature_schema_hash(FEATURE_NAMES))
        self.assertNotEqual(
            feature_schema_hash(FEATURE_NAMES),
            feature_schema_hash(list(reversed(FEATURE_NAMES))),
        )


if __name__ == "__main__":
    unittest.main()
