"""Source registry: sources bucket by CORRECT role, failures are non-fatal, and
supervised stays clean (no background/auxiliary contamination)."""
from __future__ import annotations

import unittest

from data.sources_registry import assemble, REGISTRY, SourceSpec
from data.cohort import build_full_cohort


class TestSourcesRegistry(unittest.TestCase):
    def test_roles_are_valid_and_flab_is_supervised(self) -> None:
        roles = {s.name: s.role for s in REGISTRY}
        self.assertEqual(roles["flab"], "supervised")
        self.assertEqual(roles["gdpa"], "supervised")
        self.assertEqual(roles["oas"], "background_ood")
        self.assertEqual(roles["canya"], "auxiliary")
        for s in REGISTRY:
            self.assertIn(s.role, ("supervised", "auxiliary", "background_ood"))

    def test_assemble_buckets_and_stamps_rows(self) -> None:
        # Only FLAb is local + stdlib here; network/absent sources skip gracefully.
        buckets = assemble()
        self.assertEqual(set(buckets), {"supervised", "auxiliary", "background_ood"})
        self.assertGreater(len(buckets["supervised"]), 0)  # FLAb is present
        for r in buckets["supervised"]:
            self.assertEqual(r["supervision_status"], "supervised")
            self.assertTrue(r.get("source"))

    def test_missing_source_is_non_fatal(self) -> None:
        # A bogus include set yields empty buckets, never an exception.
        buckets = assemble(include={"does_not_exist"})
        self.assertEqual(buckets["supervised"], [])

    def test_full_cohort_attaches_pairing_and_isolates_supervised(self) -> None:
        buckets = build_full_cohort()
        for r in buckets["supervised"]:
            self.assertIn("pair_id", r)  # pairing attached
        # supervised must not contain background/auxiliary rows
        sup_sources = {r["source"] for r in buckets["supervised"]}
        self.assertNotIn("oas", sup_sources)
        self.assertNotIn("canya", sup_sources)


if __name__ == "__main__":
    unittest.main()
