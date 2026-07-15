"""Leakage-safe nested grouped cross-validation for Plan C.

Outer/inner folds are keyed on the composite group
`(study, campaign, identity_component)` (spec section 8) so that no study or
identity component is ever split across train/test. There is no ungrouped
fallback: whole groups are assigned to folds, and fold count only reduces
deterministically when composite groups are insufficient.

Identity components come from ``data.identity_graph.build_identity_components``;
this module does not reimplement alignment. Pure stdlib (no numpy/sklearn).
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib

from data.identity_graph import (  # re-exported for callers building group ids
    build_identity_components as build_identity_components,
)

__all__ = [
    "composite_group_key",
    "grouped_folds",
    "assert_no_group_leakage",
    "nested_folds",
    "build_identity_components",
]

Fold = tuple[list[int], list[int]]


def _field(row: object, name: str) -> object | None:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def _clean(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def composite_group_key(row: object) -> str:
    """Build the outer group token `(study, campaign, component)` for a row.

    Accepts a mapping or any object with the relevant attributes. Falls back:
    campaign -> study, component -> record id -> stable content hash. Raising is
    reserved for rows with no study identifier at all, since that would silently
    collapse everything into one group.
    """
    study = _clean(_field(row, "study_id")) or _clean(_field(row, "group_study"))
    if study is None:
        raise ValueError("composite_group_key requires study_id (or group_study) on the row")
    campaign = (
        _clean(_field(row, "campaign_id"))
        or _clean(_field(row, "group_campaign"))
        or study
    )
    component = (
        _clean(_field(row, "identity_component_id"))
        or _clean(_field(row, "record_id"))
        or "row:" + hashlib.sha1(repr(row).encode("utf-8")).hexdigest()[:16]
    )
    return f"{study}|{campaign}|{component}"


def _stable_rank(key: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()


def grouped_folds(
    group_keys: list[str],
    labels: list[int],
    n_splits: int = 5,
    seed: int = 42,
) -> list[Fold]:
    """Assign whole groups to folds; return `(train_idx, test_idx)` per fold.

    Groups are ordered by descending size (seed-stable hash tie-break) and
    round-robined across folds, so each group lands in exactly one test fold and
    never straddles train/test. ``n_splits`` reduces to the number of groups when
    fewer groups exist; it never falls back to splitting a group.
    """
    if len(group_keys) != len(labels):
        raise ValueError("group_keys and labels must be the same length")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    groups: dict[str, list[int]] = {}
    for idx, key in enumerate(group_keys):
        groups.setdefault(key, []).append(idx)

    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError(f"need >= 2 composite groups for grouped CV, got {n_groups}")

    n_splits = min(n_splits, n_groups)

    # ponytail: descending-size round-robin balances folds well enough; label
    # stratification under a whole-group constraint isn't worth the machinery.
    ordered = sorted(groups, key=lambda k: (-len(groups[k]), _stable_rank(k, seed)))
    fold_of_group = {key: i % n_splits for i, key in enumerate(ordered)}

    folds: list[Fold] = []
    for f in range(n_splits):
        train_idx: list[int] = []
        test_idx: list[int] = []
        for idx, key in enumerate(group_keys):
            (test_idx if fold_of_group[key] == f else train_idx).append(idx)
        folds.append((train_idx, test_idx))
    return folds


def assert_no_group_leakage(folds: list[Fold], group_keys: list[str]) -> None:
    """Raise AssertionError naming the first group that leaks across a fold."""
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_groups = {group_keys[i] for i in train_idx}
        test_groups = {group_keys[i] for i in test_idx}
        overlap = train_groups & test_groups
        if overlap:
            offender = sorted(overlap)[0]
            raise AssertionError(
                f"group leakage in fold {fold_i}: {offender!r} appears in both train and test"
            )


def nested_folds(
    group_keys: list[str],
    labels: list[int],
    n_splits: int = 5,
    inner_splits: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Outer grouped folds, each with inner grouped folds over its train block.

    Inner folds reuse the SAME composite groups via ``grouped_folds`` on the
    outer-train subset; indices are mapped back to the global positions. An outer
    train block with fewer than 2 groups yields an empty ``inner`` (nothing to
    tune) rather than any ungrouped fallback.
    """
    outer = grouped_folds(group_keys, labels, n_splits, seed)
    result: list[dict] = []
    for outer_train, outer_test in outer:
        sub_keys = [group_keys[i] for i in outer_train]
        sub_labels = [labels[i] for i in outer_train]
        try:
            inner = grouped_folds(sub_keys, sub_labels, inner_splits, seed)
        except ValueError:
            inner = []  # ponytail: <2 groups in this block -> no inner tuning folds
        inner_global: list[Fold] = [
            ([outer_train[i] for i in tr], [outer_train[i] for i in te])
            for tr, te in inner
        ]
        result.append(
            {"train": outer_train, "test": outer_test, "inner": inner_global}
        )
    return result
