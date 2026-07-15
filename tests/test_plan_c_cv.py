"""Tests for leakage-safe nested grouped CV (model/plan_c_cv.py)."""

from __future__ import annotations

import unittest

from model import plan_c_cv


def _rows():
    """Synthetic rows spanning several composite groups with two labels."""
    rows = []
    # 6 identity components across 3 studies/campaigns, varied sizes.
    plan = [
        ("s1", "c1", "comp-a", 3, 1),
        ("s1", "c2", "comp-b", 2, 0),
        ("s2", "c3", "comp-c", 4, 1),
        ("s2", "c4", "comp-d", 1, 0),
        ("s3", "c5", "comp-e", 2, 1),
        ("s3", "c6", "comp-f", 2, 0),
    ]
    for study, campaign, comp, count, label in plan:
        for k in range(count):
            rows.append(
                {
                    "record_id": f"{comp}-{k}",
                    "study_id": study,
                    "campaign_id": campaign,
                    "identity_component_id": comp,
                    "label": label,
                }
            )
    return rows


class TestCompositeKey(unittest.TestCase):
    def test_key_and_fallbacks(self):
        self.assertEqual(
            plan_c_cv.composite_group_key(
                {"study_id": "s", "campaign_id": "c", "identity_component_id": "k"}
            ),
            "s|c|k",
        )
        # campaign -> study, component -> record_id
        self.assertEqual(
            plan_c_cv.composite_group_key({"study_id": "s", "record_id": "r7"}),
            "s|s|r7",
        )

    def test_missing_study_raises(self):
        with self.assertRaises(ValueError):
            plan_c_cv.composite_group_key({"record_id": "r1"})


class TestGroupedFolds(unittest.TestCase):
    def setUp(self):
        self.rows = _rows()
        self.keys = [plan_c_cv.composite_group_key(r) for r in self.rows]
        self.labels = [r["label"] for r in self.rows]

    def test_every_group_in_exactly_one_test_fold(self):
        folds = plan_c_cv.grouped_folds(self.keys, self.labels, n_splits=5)
        counts = {}
        for _train, test in folds:
            for key in {self.keys[i] for i in test}:
                counts[key] = counts.get(key, 0) + 1
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(set(counts), set(self.keys))

    def test_no_group_leakage_passes(self):
        folds = plan_c_cv.grouped_folds(self.keys, self.labels, n_splits=5)
        plan_c_cv.assert_no_group_leakage(folds, self.keys)  # must not raise

    def test_assert_raises_on_corrupted_fold(self):
        folds = plan_c_cv.grouped_folds(self.keys, self.labels, n_splits=5)
        train, test = folds[0]
        corrupted = [(train + [test[0]], test)] + folds[1:]  # leak one test idx into train
        with self.assertRaises(AssertionError):
            plan_c_cv.assert_no_group_leakage(corrupted, self.keys)

    def test_more_splits_than_groups_reduces_deterministically(self):
        # 6 groups; asking for 20 splits must yield exactly 6 folds, and repeat.
        f1 = plan_c_cv.grouped_folds(self.keys, self.labels, n_splits=20)
        f2 = plan_c_cv.grouped_folds(self.keys, self.labels, n_splits=20)
        self.assertEqual(len(f1), 6)
        self.assertEqual(f1, f2)

    def test_fewer_than_two_groups_raises(self):
        keys = ["g|g|only"] * 4
        with self.assertRaises(ValueError):
            plan_c_cv.grouped_folds(keys, [0, 1, 0, 1], n_splits=5)


class TestNestedFolds(unittest.TestCase):
    def test_inner_folds_are_group_safe_and_global(self):
        rows = _rows()
        keys = [plan_c_cv.composite_group_key(r) for r in rows]
        labels = [r["label"] for r in rows]
        nested = plan_c_cv.nested_folds(keys, labels, n_splits=3, inner_splits=3)
        self.assertEqual(len(nested), 3)
        for block in nested:
            # inner train/test indices are drawn from the outer train block only
            train_set = set(block["train"])
            for inner_train, inner_test in block["inner"]:
                self.assertTrue(set(inner_train + inner_test) <= train_set)
                plan_c_cv.assert_no_group_leakage([(inner_train, inner_test)], keys)


if __name__ == "__main__":
    unittest.main()
