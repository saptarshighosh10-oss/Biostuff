"""Focused tests for Plan C data-contract and provenance primitives."""

from __future__ import annotations

import os
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import cache as cache_module
from data.cache import (
    assert_frozen_cache_match,
    load_frozen_source_manifest,
    load_frozen_source_rows,
    manifest_content_hash,
    _metadata_hash,
    _row_hash,
    write_frozen_source_bundle,
)
from data.contract import (
    ContractError,
    HASH_EXCLUDE_KEYS,
    canonical_source_name,
    canonical_json_dumps,
    hash_payload,
    normalize_legacy_row,
    NormalizedRow,
)
from model.preprocess import build_conflict_and_exclusion_ledgers


def _mk_row(overrides: dict | None = None) -> NormalizedRow:
    base = {
        "record_id": "r1",
        "source": "flab",
        "study_id": "study",
        "campaign_id": "campaign",
        "molecule_id": "mol",
        "pair_id": None,
        "chain_id": None,
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "vh_sequence": None,
        "vl_sequence": None,
        "assay_id": "assay",
        "assay_metric": "HIC_RT",
        "endpoint_direction": "higher_bad",
        "endpoint_value": 1.0,
        "endpoint_unit": "ms",
        "target_value": 1.0,
        "supervision_status": "supervised",
        "binary_label": 1,
        "label_threshold": 0.5,
        "group_study": "study",
        "group_molecule": "mol",
        "group_campaign": "campaign",
        "sequence_hash": "x",
        "exclusions": [],
        "source_url": "",
        "source_row_hash": "sha",
        "feature_flags": [],
    }
    if overrides:
        base.update(overrides)
    return NormalizedRow(**base)


class TestCanonicalHashing(unittest.TestCase):
    """Canonical JSON and hash helpers must ignore key order and volatile fields."""

    def test_hash_stable_across_key_order(self) -> None:
        payload_a = {
            "sequence": "ACD",
            "meta": {"score": 1.0, "created_at": "2026-01-01T00:00:00Z"},
            "timestamp": "2026-01-01T12:00:00Z",
        }
        payload_b = {
            "timestamp": "2026-01-02T00:00:00Z",
            "meta": {"score": 1.0, "created_at": "2026-02-01T00:00:00Z"},
            "sequence": "ACD",
        }
        self.assertEqual(hash_payload(payload_a), hash_payload(payload_b))
        self.assertEqual(canonical_json_dumps(payload_a), canonical_json_dumps(payload_b))

    def test_hash_changes_on_nonvolatile_field(self) -> None:
        payload_a = {"sequence": "ACD", "timestamp": "x"}
        payload_b = {"sequence": "ACDE", "timestamp": "y"}
        self.assertNotEqual(hash_payload(payload_a), hash_payload(payload_b))


class TestContractRules(unittest.TestCase):
    """Contract normalization should enforce explicit provenance rules."""

    def test_contract_gate_errors_do_not_depend_on_train(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_unit": "ms",
            "label": "confirmed_failure",
        }
        with self.assertRaises(ContractError):
            normalize_legacy_row(row)

    def test_supervised_row_accepts_endpoint_value_zero(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_value": 0,
            "endpoint_unit": "ms",
        }
        normalized = normalize_legacy_row(row)
        self.assertEqual(normalized.endpoint_value, 0.0)
        self.assertEqual(normalized.target_value, 0.0)
        self.assertIsNone(normalized.binary_label)

    def test_supervised_non_finite_endpoint_is_rejected(self) -> None:
        for value in ["nan", "inf", "-inf", float("nan"), float("inf"), float("-inf")]:
            row = {
                "source": "flab",
                "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
                "assay_metric": "HIC_RT",
                "endpoint_direction": "higher_bad",
                "endpoint_value": value,
                "endpoint_unit": "ms",
                "label": "confirmed_failure",
                "label_threshold": 0.5,
            }
            with self.assertRaises(ContractError):
                normalize_legacy_row(row)

    def test_supervised_non_finite_label_threshold_is_rejected(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_value": 1.0,
            "endpoint_unit": "ms",
            "label": "confirmed_failure",
            "label_threshold": "nan",
        }
        with self.assertRaises(ContractError):
            normalize_legacy_row(row)

    def test_supervised_row_without_explicit_direction_is_rejected(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_value": 1.0,
            "endpoint_unit": "ms",
            "label": "confirmed_failure",
        }
        with self.assertRaises(ContractError):
            normalize_legacy_row(row)

    def test_supervised_row_with_invalid_direction_is_rejected(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "invalid",
            "endpoint_value": 1.0,
            "endpoint_unit": "ms",
            "label": "confirmed_failure",
        }
        with self.assertRaises(ContractError):
            normalize_legacy_row(row)

    def test_supervised_row_without_label_threshold_is_rejected(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_value": 2.0,
            "endpoint_unit": "ms",
            "label": "confirmed_failure",
        }
        with self.assertRaises(ContractError):
            normalize_legacy_row(row)

    def test_continuous_supervised_row_without_binary_label_is_accepted(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_value": 0.5,
            "endpoint_unit": "ms",
        }
        normalized = normalize_legacy_row(row)
        self.assertEqual(normalized.supervision_status, "supervised")
        self.assertIsNone(normalized.binary_label)
        self.assertIsNone(normalized.label_threshold)

    def test_supervised_non_string_endpoint_unit_is_rejected(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_value": 1.0,
            "endpoint_unit": 1,
            "label": "confirmed_failure",
            "label_threshold": 0.5,
        }
        with self.assertRaises(ContractError):
            normalize_legacy_row(row)

    def test_endpoint_unit_whitespace_is_stripped_for_supervised_rows(self) -> None:
        row = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_value": 1.0,
            "endpoint_unit": "  ms  ",
            "label": "confirmed_failure",
            "label_threshold": 0.5,
        }
        normalized = normalize_legacy_row(row)
        self.assertEqual(normalized.endpoint_unit, "ms")

    def test_auxiliary_and_background_rows_are_not_supervised(self) -> None:
        canya = {
            "source": "canya",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "label": "confirmed_failure",
            "a3d_score": 1.0,
        }
        aux = normalize_legacy_row(canya)
        self.assertEqual(aux.supervision_status, "auxiliary")
        self.assertIsNone(aux.binary_label)

        abdev = {
            "source": "abdev",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "label": "working",
        }
        background = normalize_legacy_row(abdev)
        self.assertEqual(background.supervision_status, "background_ood")
        self.assertIsNone(background.binary_label)
        self.assertEqual(background.endpoint_direction, "higher_bad")

    def test_auxiliary_row_with_explicit_direction_prefers_explicit_value(self) -> None:
        row = {
            "source": "canya",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "a3d_score": 1.0,
            "endpoint_direction": "lower_bad",
        }
        normalized = normalize_legacy_row(row)
        self.assertEqual(normalized.endpoint_direction, "lower_bad")
        self.assertEqual(normalized.supervision_status, "auxiliary")

    def test_missing_source_is_rejected(self) -> None:
        base_row = {
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
            "endpoint_unit": "ms",
            "endpoint_value": 1.0,
        }
        bad_sources = [
            ("missing", object()),
            ("none", None),
            ("empty", ""),
            ("whitespace", "   "),
            ("nonstr", 123),
        ]
        for label, source in bad_sources:
            row = dict(base_row)
            if label == "missing":
                row.pop("source", None)
            else:
                row["source"] = source
            with self.subTest(source=label):
                with self.assertRaises(ContractError):
                    normalize_legacy_row(row)

    def test_explicit_unknown_source_is_preserved_as_explicit_unlabeled(self) -> None:
        row = {
            "source": "unknown",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
        }
        normalized = normalize_legacy_row(row)
        self.assertEqual(normalized.source, "unknown")
        self.assertEqual(normalized.supervision_status, "unlabeled")

    def test_canonical_source_name_alias_and_whitespace_stable(self) -> None:
        self.assertEqual(canonical_source_name(" FlAb "), "flab")
        self.assertEqual(canonical_source_name("protein-gym"), "proteingym")

    def test_pdb_anchor_alias_and_row_hash_stability(self) -> None:
        row = {
            "source": "pdb-anchor",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "label": "working",
        }
        normalized = normalize_legacy_row(row)
        self.assertEqual(normalized.source, "pdb_anchor")
        self.assertEqual(
            normalized.source_row_hash, hash_payload(row, exclude_keys=HASH_EXCLUDE_KEYS)
        )

    def test_normalized_source_row_hash_ignores_embedded_hash(self) -> None:
        row_without_hash = {
            "source": "flab",
            "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "endpoint_value": 1.0,
            "endpoint_unit": "ms",
            "assay_metric": "HIC_RT",
            "endpoint_direction": "higher_bad",
        }
        row_with_hash = dict(row_without_hash)
        row_with_hash["source_row_hash"] = "tampered"

        normalized_without_hash = normalize_legacy_row(row_without_hash)
        normalized_with_hash = normalize_legacy_row(row_with_hash)

        self.assertEqual(
            normalized_without_hash.source_row_hash, normalized_with_hash.source_row_hash
        )

    def test_pair_id_only_set_when_both_chains_present(self) -> None:
        with self.assertRaises(ContractError):
            normalize_legacy_row(
                {
                    "source": "flab",
                    "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "assay_metric": "HIC_RT",
                    "endpoint_direction": "higher_bad",
                    "endpoint_value": 1.0,
                    "endpoint_unit": "ms",
                    "chain_id": "vh",
                    "pair_id": "pair-1",
                },
            )

        normalized = normalize_legacy_row(
            {
                "source": "flab",
                "variant_sequence": "ACDEFGHIKLMNPQRSTVWY",
                "assay_metric": "HIC_RT",
                "endpoint_direction": "higher_bad",
                "endpoint_value": 1.0,
                "endpoint_unit": "ms",
                "vh_sequence": "ACDEFGHIKL",
                "vl_sequence": "MNPQRSTVWY",
                "pair_id": "pair-1",
            },
        )
        self.assertEqual(normalized.pair_id, "pair-1")


class TestProvenanceLedgers(unittest.TestCase):
    """Conflict and exclusion ledgers must not silently drop conflicts."""

    def test_label_flip_is_preserved(self) -> None:
        r1 = _mk_row({"record_id": "r1", "sequence_hash": "s1", "binary_label": 0})
        r2 = _mk_row({"record_id": "r2", "sequence_hash": "s1", "binary_label": 1})

        kept, conflicts, exclusions = build_conflict_and_exclusion_ledgers([r1, r2])
        self.assertEqual(kept, [])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["reason"], "label_flip")
        self.assertEqual({e["record_id"] for e in exclusions}, {"r1", "r2"})

    def test_identity_overlap_and_chain_mismatch(self) -> None:
        overlap_1 = _mk_row(
            {"record_id": "r1", "sequence_hash": "s2", "pair_id": "p1", "chain_id": "vh"}
        )
        overlap_2 = _mk_row(
            {"record_id": "r2", "sequence_hash": "s2", "pair_id": "p2", "chain_id": "vh"}
        )
        mismatch_1 = _mk_row(
            {
                "record_id": "r3",
                "sequence_hash": "s3",
                "pair_id": "pair",
                "chain_id": "vh",
            }
        )
        mismatch_2 = _mk_row(
            {
                "record_id": "r4",
                "sequence_hash": "s3",
                "pair_id": "pair",
                "chain_id": "ab",
            }
        )
        kept, conflicts, exclusions = build_conflict_and_exclusion_ledgers(
            [overlap_1, overlap_2, mismatch_1, mismatch_2]
        )
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(conflicts), 0)
        reason_map = {row["record_id"]: row["reason"] for row in exclusions}
        self.assertIn("identity_overlap", reason_map["r1"])
        self.assertIn("identity_overlap", reason_map["r2"])
        self.assertIn("chain_mismatch", reason_map["r3"])
        self.assertIn("chain_mismatch", reason_map["r4"])

    def test_single_chain_row_with_pair_id_is_excluded(self) -> None:
        row = _mk_row(
            {
                "record_id": "single",
                "chain_id": "vh",
                "pair_id": "pair-1",
                "binary_label": 1,
            }
        )
        _, _, exclusions = build_conflict_and_exclusion_ledgers([row])
        reasons = [row["reason"] for row in exclusions if row["record_id"] == "single"]
        self.assertTrue(any("single_chain_pair_id" in reason for reason in reasons))

    def test_invalid_chain_id_marks_row_excluded(self) -> None:
        row = _mk_row(
            {
                "record_id": "bad_chain",
                "chain_id": "x",
                "binary_label": 1,
            }
        )
        _, _, exclusions = build_conflict_and_exclusion_ledgers([row])
        reasons = [row["reason"] for row in exclusions if row["record_id"] == "bad_chain"]
        self.assertTrue(any("invalid_chain_id" in reason for reason in reasons))

    def test_duplicate_record_ids_create_exclusion_ledger_entries(self) -> None:
        first = _mk_row({"record_id": "dup", "binary_label": 1})
        second = _mk_row({"record_id": "dup", "binary_label": 1})
        kept, _, exclusions = build_conflict_and_exclusion_ledgers([first, second])
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(exclusions), 2)
        self.assertEqual(len([row for row in exclusions if row["record_id"] == "dup"]), 2)
        self.assertTrue(
            all("duplicate_record_id" in row["reason"] for row in exclusions)
        )

    def test_repeated_pair_level_rows_without_chain_ids_are_not_chain_mismatch(self) -> None:
        row_1 = _mk_row(
            {
                "record_id": "pair-level-1",
                "sequence_hash": "s100",
                "pair_id": "pair-xyz",
                "chain_id": None,
                "vh_sequence": "ACDEFGHIKL",
                "vl_sequence": "MNPQRSTVWY",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
            }
        )
        row_2 = _mk_row(
            {
                "record_id": "pair-level-2",
                "sequence_hash": "s101",
                "pair_id": "pair-xyz",
                "chain_id": None,
                "vh_sequence": "FGHIKLMNPQ",
                "vl_sequence": "RSTVWYACDE",
                "sequence": "FGHIKLMNPQRSTVWYACDE",
            }
        )
        kept, _, exclusions = build_conflict_and_exclusion_ledgers([row_1, row_2])
        self.assertEqual(
            {row["record_id"] for row in exclusions},
            set(),
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(
            {row.record_id for row in kept},
            {"pair-level-1", "pair-level-2"},
        )

    def test_mixed_pair_level_and_chain_member_rows_fail_chain_consistency(self) -> None:
        pair_level = _mk_row(
            {
                "record_id": "pair-level-mixed",
                "sequence_hash": "s200",
                "pair_id": "pair-mix",
                "chain_id": None,
                "vh_sequence": "ACDEFGHIKL",
                "vl_sequence": "MNPQRSTVWY",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
            }
        )
        chain_level = _mk_row(
            {
                "record_id": "chain-level-mixed",
                "sequence_hash": "s201",
                "pair_id": "pair-mix",
                "chain_id": "vh",
                "vh_sequence": "ACDEFGHIKL",
                "vl_sequence": "MNPQRSTVWY",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
            }
        )
        _, _, exclusions = build_conflict_and_exclusion_ledgers([pair_level, chain_level])
        reasons = {row["record_id"]: row["reason"] for row in exclusions}
        self.assertIn("pair-level-mixed", reasons)
        self.assertIn("chain-level-mixed", reasons)
        self.assertIn("chain_mismatch", reasons["pair-level-mixed"])
        self.assertIn("chain_mismatch", reasons["chain-level-mixed"])



class TestFrozenSourceCache(unittest.TestCase):
    """Frozen cache manifests must reject stale/mutated payloads."""

    def _make_rows(self) -> list[dict]:
        return [
            {
                "record_id": "a",
                "sequence_hash": "01",
                "source": "flab",
                "source_row_hash": "ha",
            },
            {
                "record_id": "b",
                "sequence_hash": "02",
                "source": "flab",
                "source_row_hash": "hb",
            },
        ]

    def test_manifest_hash_ignores_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=Path(td)
            )
            manifest = load_frozen_source_manifest("flab", cache_root=Path(td))
            manifest_2 = dict(manifest)
            manifest_2["fetch_timestamp_utc"] = "2020-01-01T00:00:00+00:00"
            manifest_2_hash = manifest_content_hash(manifest_2)
            self.assertEqual(manifest_content_hash(manifest), manifest_2_hash)

    def test_manifest_source_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            written = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"] = "abdev"
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows(
                    "flab", cache_root=cache_root, dataset_sha=written["dataset_sha"]
                )

    def test_manifest_loader_rejects_source_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"] = "abdev"
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_manifest_loader_rejects_wrong_type_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"] = 123
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_manifest_loader_rejects_non_string_dataset_sha(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_sha"] = 123
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_manifest_dataset_dir_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            written = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_dir"] = "raw/flab/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_manifest_loader_rejects_missing_dataset_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_dir"] = "raw/flab/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_manifest_rows_file_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            written = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["rows_file"] = (
                f"raw/flab/{written['dataset_sha']}/rows_alt.jsonl.gz"
            )
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_manifest_loader_rejects_invalid_rows_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            manifest_sha = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["rows_file"] = (
                f"raw/flab/{manifest_sha['dataset_sha']}/rows_alt.jsonl.gz"
            )
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_metadata_source_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            written = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            metadata_path = (
                cache_root / "raw" / "flab" / written["dataset_sha"] / "dataset.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source"] = "abdev"
            metadata["metadata_hash"] = _metadata_hash(metadata)
            metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows(
                    "flab", cache_root=cache_root, dataset_sha=written["dataset_sha"]
                )

    def test_metadata_dataset_format_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            written = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            metadata_path = (
                cache_root / "raw" / "flab" / written["dataset_sha"] / "dataset.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["dataset_format"] = "txt"
            metadata["metadata_hash"] = _metadata_hash(metadata)
            metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows(
                    "flab", cache_root=cache_root, dataset_sha=written["dataset_sha"]
                )

    def test_manifest_dataset_format_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_format"] = "txt"
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_metadata_dataset_dir_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            written = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            metadata_path = (
                cache_root / "raw" / "flab" / written["dataset_sha"] / "dataset.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["dataset_dir"] = (
                "raw/flab/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )
            metadata["metadata_hash"] = _metadata_hash(metadata)
            metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows(
                    "flab", cache_root=cache_root, dataset_sha=written["dataset_sha"]
                )

    def test_embedded_source_row_hash_not_trusted(self) -> None:
        row = {
            "record_id": "row-with-embedded-hash",
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "source": "flab",
            "source_row_hash": "trusted-hash",
        }
        self.assertNotEqual(_row_hash(row), "trusted-hash")
        self.assertEqual(
            _row_hash(row), hash_payload(row, exclude_keys=cache_module.ROW_HASH_EXCLUDE_KEYS)
        )

    def test_rewriting_same_rows_remains_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            first = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            second = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            self.assertEqual(first["dataset_sha"], second["dataset_sha"])
            first_rows, _ = load_frozen_source_rows("flab", cache_root=cache_root, dataset_sha=first["dataset_sha"])
            second_rows, _ = load_frozen_source_rows("flab", cache_root=cache_root, dataset_sha=second["dataset_sha"])
            self.assertEqual(first_rows, second_rows)

    def test_cached_bundle_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            meta = write_frozen_source_bundle(
                "flab",
                rows,
                source_urls=["https://example.invalid"],
                cache_root=Path(td),
            )
            rows_path = (
                Path(td) / "raw/flab" / meta["dataset_sha"] / "rows.jsonl.gz"
            )
            payload = rows_path.read_bytes()
            with gzip.open(rows_path, "wt", encoding="utf-8") as f:
                text = gzip.decompress(payload).decode("utf-8") + " "
                f.write(text)
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows("flab", cache_root=Path(td), dataset_sha=meta["dataset_sha"])
            with self.assertRaises(RuntimeError):
                assert_frozen_cache_match(
                    "flab",
                    dataset_sha=meta["dataset_sha"],
                    cache_root=Path(td),
                )

    def test_modified_payload_rejects_even_if_embedded_hash_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            meta = write_frozen_source_bundle(
                "flab",
                rows,
                source_urls=["https://example.invalid"],
                cache_root=cache_root,
            )
            rows_path = (
                cache_root / "raw/flab" / meta["dataset_sha"] / "rows.jsonl.gz"
            )
            with gzip.open(rows_path, "rt", encoding="utf-8") as f:
                rows_payload = [json.loads(line) for line in f if line.strip()]
            rows_payload[0]["source_row_hash"] = "pinned-hash"
            rows_payload[0]["endpoint_unit"] = "ms"
            with gzip.open(rows_path, "wt", encoding="utf-8") as f:
                for row in rows_payload:
                    f.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                    f.write("\n")
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows("flab", cache_root=cache_root, dataset_sha=meta["dataset_sha"])

    def test_non_dict_frozen_row_is_rejected_with_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            meta = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            rows_path = cache_root / "raw/flab" / meta["dataset_sha"] / "rows.jsonl.gz"
            with gzip.open(rows_path, "rt", encoding="utf-8") as f:
                tampered_rows = [json.loads(line) for line in f if line.strip()]
            tampered_rows[0] = ["non-dict-row"]
            tampered_payload = "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) for row in tampered_rows
            ).encode("utf-8")

            with gzip.open(rows_path, "wb") as f:
                f.write(tampered_payload)

            new_payload_hash = cache_module.sha256_hex(tampered_payload)
            metadata_path = cache_root / "raw" / "flab" / meta["dataset_sha"] / "dataset.json"
            manifest_path = cache_root / "manifests" / "flab.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            metadata["rows_payload_hash"] = new_payload_hash
            metadata["row_hashes"] = sorted(set(_row_hash(row) for row in tampered_rows))
            metadata["metadata_hash"] = _metadata_hash(metadata)
            metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))

            manifest["rows_payload_hash"] = new_payload_hash
            manifest["row_hashes"] = sorted(set(_row_hash(row) for row in tampered_rows))
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )

            with self.assertRaises(RuntimeError):
                load_frozen_source_rows(
                    "flab", cache_root=cache_root, dataset_sha=meta["dataset_sha"]
                )

    def test_malformed_jsonl_is_rejected_with_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            meta = write_frozen_source_bundle(
                "flab", rows, source_urls=["https://example.invalid"], cache_root=cache_root
            )
            rows_path = cache_root / "raw/flab" / meta["dataset_sha"] / "rows.jsonl.gz"
            with gzip.open(rows_path, "rt", encoding="utf-8") as f:
                rows_text = [line.rstrip("\n") for line in f if line.strip()]
            rows_text[0] = rows_text[0] + " [invalid"
            tampered_payload = "\n".join(rows_text).encode("utf-8")

            with gzip.open(rows_path, "wb") as f:
                f.write(tampered_payload)

            payload_hash = cache_module.sha256_hex(tampered_payload)
            metadata_path = cache_root / "raw" / "flab" / meta["dataset_sha"] / "dataset.json"
            manifest_path = cache_root / "manifests" / "flab.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            metadata["rows_payload_hash"] = payload_hash
            metadata["metadata_hash"] = _metadata_hash(metadata)
            manifest["rows_payload_hash"] = payload_hash
            manifest["manifest_content_hash"] = manifest_content_hash(manifest)
            metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )

            with self.assertRaises(RuntimeError):
                load_frozen_source_rows(
                    "flab", cache_root=cache_root, dataset_sha=meta["dataset_sha"]
                )

    def test_schema_hash_change_invalidates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            meta = write_frozen_source_bundle(
                "flab",
                rows,
                source_urls=["https://example.invalid"],
                cache_root=cache_root,
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["row_schema_hash"] = "deadbeef"
            payload["manifest_content_hash"] = manifest_content_hash(payload)
            manifest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows("flab", cache_root=cache_root, dataset_sha=meta["dataset_sha"])

    def test_manifest_metadata_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            meta = write_frozen_source_bundle(
                "flab",
                rows,
                source_urls=["https://example.invalid"],
                cache_root=cache_root,
            )
            metadata_path = (
                cache_root / "raw/flab" / meta["dataset_sha"] / "dataset.json"
            )
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["metadata_hash"] = "bad-metadata-hash"
            metadata_path.write_text(json.dumps(payload, sort_keys=True, indent=2))
            with self.assertRaises(RuntimeError):
                load_frozen_source_rows("flab", cache_root=cache_root, dataset_sha=meta["dataset_sha"])

    def test_path_traversal_source_name_rejected(self) -> None:
        rows = self._make_rows()
        with tempfile.TemporaryDirectory() as td:
            for source in ["../bad", "a/b", "a\\b", "FLAB", "bad source"]:
                with self.assertRaises(ValueError):
                    write_frozen_source_bundle(
                        source, rows, source_urls=[], cache_root=Path(td)
                    )

    def test_writer_rejects_non_dict_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            valid_row = {"record_id": "a", "sequence_hash": "01", "source": "flab"}
            with self.assertRaisesRegex(RuntimeError, r"rows\[0\].*must be a dict"):
                write_frozen_source_bundle(
                    "flab", [1], source_urls=[], cache_root=Path(td)
                )
            with self.assertRaisesRegex(RuntimeError, r"rows\[1\].*must be a dict"):
                write_frozen_source_bundle(
                    "flab", [valid_row, 2], source_urls=[], cache_root=Path(td)
                )

    def test_writer_rejects_non_string_row_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad_rows = [{1: "bad-key", "record_id": "a", "source": "flab"}]
            with self.assertRaisesRegex(RuntimeError, r"rows\[0\].*keys must be strings"):
                write_frozen_source_bundle("flab", bad_rows, source_urls=[], cache_root=Path(td))

    def test_writer_rejects_non_string_source_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            with self.assertRaisesRegex(
                RuntimeError, r"source_urls\[0\].*non-empty string"
            ):
                write_frozen_source_bundle(
                    "flab", rows, source_urls=[123], cache_root=Path(td)
                )

    def test_writer_rejects_non_list_rows_and_source_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, r"rows must be a list"):
                write_frozen_source_bundle("flab", "not-a-list", source_urls=[], cache_root=Path(td))
            with self.assertRaisesRegex(RuntimeError, r"source_urls must be a list"):
                write_frozen_source_bundle(
                    "flab", self._make_rows(), source_urls="not-a-list", cache_root=Path(td)
                )

    def test_manifest_with_unsafe_dataset_sha_is_rejected(self) -> None:
        rows = self._make_rows()
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            meta = write_frozen_source_bundle(
                "flab",
                rows,
                source_urls=["https://example.invalid"],
                cache_root=cache_root,
            )
            manifest_path = cache_root / "manifests" / "flab.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["dataset_sha"] = "../" + meta["dataset_sha"][:2]
            payload["manifest_content_hash"] = manifest_content_hash(payload)
            manifest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_path_traversal_dataset_id_rejected(self) -> None:
        rows = self._make_rows()
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            meta = write_frozen_source_bundle(
                "flab",
                rows,
                source_urls=["https://example.invalid"],
                cache_root=cache_root,
            )
            invalid_dataset_ids = [
                "../x",
                "x/y",
                "",
                meta["dataset_sha"].upper(),
                meta["dataset_sha"][:-1],
                123,
            ]
            for dataset_id in invalid_dataset_ids:
                with self.assertRaises(RuntimeError):
                    load_frozen_source_rows(
                        "flab", dataset_sha=dataset_id, cache_root=cache_root
                    )
            with self.assertRaises(ValueError):
                cache_module._dataset_dir(cache_root, "flab", None)

    def test_writer_rejects_symlinked_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            outside_root = Path(td) / "outside"
            cache_root.mkdir()
            outside_root.mkdir()

            (cache_root / "raw").symlink_to(outside_root)

            with self.assertRaises(RuntimeError):
                write_frozen_source_bundle(
                    "flab",
                    self._make_rows(),
                    source_urls=["https://example.invalid"],
                    cache_root=cache_root,
                )

            self.assertEqual(len(list(outside_root.iterdir())), 0)

    def test_writer_rejects_symlinked_manifests_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td) / "cache"
            outside_root = Path(td) / "outside"
            cache_root.mkdir()
            outside_root.mkdir()

            (cache_root / "manifests").symlink_to(outside_root)

            with self.assertRaises(RuntimeError):
                write_frozen_source_bundle(
                    "flab",
                    self._make_rows(),
                    source_urls=["https://example.invalid"],
                    cache_root=cache_root,
                )

            self.assertEqual(len(list(outside_root.iterdir())), 0)

    def test_reader_rejects_symlinked_manifest_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            metadata = write_frozen_source_bundle(
                "flab",
                rows,
                source_urls=["https://example.invalid"],
                cache_root=cache_root,
            )

            manifest_path = cache_root / "manifests" / "flab.json"
            target_root = cache_root / "outside"
            target_root.mkdir()
            target = target_root / "flab.json"
            target.write_text(json.dumps({"rows": []}, sort_keys=True), encoding="utf-8")

            manifest_path.unlink()
            manifest_path.symlink_to(target)

            with self.assertRaises(RuntimeError):
                load_frozen_source_manifest("flab", cache_root=cache_root)

    def test_reader_rejects_symlinked_dataset_metadata_or_rows(self) -> None:
        for variant in ("dataset", "metadata", "rows"):
            with self.subTest(variant=variant):
                with tempfile.TemporaryDirectory() as td:
                    cache_root = Path(td)
                    written = write_frozen_source_bundle(
                        "flab",
                        self._make_rows(),
                        source_urls=["https://example.invalid"],
                        cache_root=cache_root,
                    )

                    dataset_dir = cache_root / "raw" / "flab" / written["dataset_sha"]
                    outside_root = cache_root / "outside"
                    outside_root.mkdir()

                    if variant == "dataset":
                        outside = outside_root / "dataset"
                        dataset_dir.rename(outside)
                        dataset_dir.symlink_to(outside)
                    elif variant == "metadata":
                        target = outside_root / "dataset.json"
                        target.write_text("{}", encoding="utf-8")
                        (dataset_dir / "dataset.json").unlink()
                        (dataset_dir / "dataset.json").symlink_to(target)
                    else:
                        target = outside_root / "rows.jsonl.gz"
                        target.touch()
                        (dataset_dir / "rows.jsonl.gz").unlink()
                        (dataset_dir / "rows.jsonl.gz").symlink_to(target)

                    with self.assertRaises(RuntimeError):
                        load_frozen_source_rows(
                            "flab",
                            cache_root=cache_root,
                            dataset_sha=written["dataset_sha"],
                        )

    def test_relative_cache_root_works(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "cache"
            original_cwd = Path.cwd()
            rows = self._make_rows()
            try:
                os.chdir(td)
                written = write_frozen_source_bundle(
                    "flab",
                    rows,
                    source_urls=["https://example.invalid"],
                    cache_root=Path("cache"),
                )

                _, _ = load_frozen_source_rows(
                    "flab",
                    cache_root=Path("cache"),
                    dataset_sha=written["dataset_sha"],
                )
            finally:
                os.chdir(original_cwd)

            manifest = cache_dir / "manifests" / "flab.json"
            self.assertTrue(manifest.exists())

    def test_atomic_bundle_write_no_partial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rows = self._make_rows()
            cache_root = Path(td)
            with patch.object(cache_module.os, "replace", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    write_frozen_source_bundle(
                        "flab",
                        rows,
                        source_urls=["https://example.invalid"],
                        cache_root=cache_root,
                    )
            temp_candidates = list(cache_root.rglob("*.tmp.*"))
            self.assertEqual(len(temp_candidates), 0)


if __name__ == "__main__":
    unittest.main()
