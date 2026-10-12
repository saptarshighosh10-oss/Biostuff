"""Evaluate polyreactivity transfer from library antibodies to diverse antibodies.

Unlike ``head_c_polyreactivity``, this module never evaluates a source using
examples from that source for training.  The diverse AIMS sources are held out
entirely, and any training row sharing their VH sequence is removed.
"""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Iterable

from model.head_c_polyreactivity import _binary_labels, _feat, _load, _model


LIBRARY_SOURCES = ("emi", "igg", "iso", "tessier_lib1", "tessier_lib2")
DIVERSE_SOURCES = ("aims_mouse", "aims_flu")


def _unique_vh(rows: Iterable[dict]) -> list[dict]:
    """Keep one row per VH, matching Head C's sequence-deduplication policy."""
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["variant_sequence"], row)
    return list(unique.values())


def evaluate_transfer(
    train_sources: Iterable[str] = LIBRARY_SOURCES,
    test_sources: Iterable[str] = DIVERSE_SOURCES,
    by_source: dict[str, list[dict]] | None = None,
) -> dict:
    """Train on library sources and score completely held-out diverse sources."""
    from sklearn.metrics import roc_auc_score

    train_sources = tuple(train_sources)
    test_sources = tuple(test_sources)
    if set(train_sources) & set(DIVERSE_SOURCES):
        raise ValueError("diverse holdout sources must never be used for training")
    by_source = _load() if by_source is None else by_source
    missing = set(train_sources + test_sources) - set(by_source)
    if missing:
        raise ValueError(f"sources absent from dataset: {', '.join(sorted(missing))}")

    raw_training = [row for source in train_sources for row in by_source[source]]
    per_source: dict[str, dict] = {}
    for source in test_sources:
        test_rows = _unique_vh(by_source[source])
        held_vh = {row["variant_sequence"] for row in test_rows}
        overlap_rows = sum(row["variant_sequence"] in held_vh for row in raw_training)
        train_rows = _unique_vh(
            row for row in raw_training if row["variant_sequence"] not in held_vh
        )
        train_labels = _binary_labels(train_rows)
        test_labels = _binary_labels(test_rows)
        if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
            per_source[source] = {
                "n": len(test_rows),
                "train_n": len(train_rows),
                "skip": "training or test set has fewer than two label classes",
            }
            continue

        model = _model()
        model.fit([_feat(row) for row in train_rows], train_labels)
        predictions = model.predict_proba([_feat(row) for row in test_rows])[:, 1]
        per_source[source] = {
            "n": len(test_rows),
            "positives": sum(test_labels),
            "train_n": len(train_rows),
            "training_rows_removed_for_vh_overlap": overlap_rows,
            "auc": float(roc_auc_score(test_labels, predictions)),
        }

    aucs = [result["auc"] for result in per_source.values() if "auc" in result]
    return {
        "train_sources": list(train_sources),
        "test_sources": list(test_sources),
        "mean_auc": mean(aucs) if aucs else None,
        "per_source": per_source,
    }


def _parse_sources(value: str) -> tuple[str, ...]:
    sources = tuple(source.strip() for source in value.split(",") if source.strip())
    if not sources:
        raise argparse.ArgumentTypeError("provide at least one comma-separated source")
    return sources


def print_report(report: dict) -> None:
    """Print the compact report used by the command line and pipeline gate."""
    print("Head C transfer — library training, diverse held-out test")
    print(f"  {'source':<14}{'n':>6}{'pos':>6}{'train n':>10}{'AUC':>8}")
    for source, result in report["per_source"].items():
        if "auc" in result:
            note = " (small holdout)" if result["n"] < 100 else ""
            print(
                f"  {source:<14}{result['n']:>6}{result['positives']:>6}"
                f"{result['train_n']:>10}{result['auc']:>8.3f}{note}"
            )
        else:
            print(f"  {source:<14}{result['n']:>6}  skip ({result['skip']})")
    if report["mean_auc"] is None:
        print("\ndiverse AUC when trained on library data: unavailable")
    else:
        print(
            f"\ndiverse AUC when trained on library data: {report['mean_auc']:.3f} "
            "(baseline within-source diverse was ~0.617)"
        )
    if "aims_flu" in report["per_source"]:
        print("Note: aims_flu has n=97, so its AUC is a small-holdout estimate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polyreactivity library-to-diverse transfer test")
    parser.add_argument("--train", type=_parse_sources, default=LIBRARY_SOURCES)
    parser.add_argument("--test", type=_parse_sources, default=DIVERSE_SOURCES)
    args = parser.parse_args()
    print_report(evaluate_transfer(args.train, args.test))
