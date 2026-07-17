"""Pure routing and equal-weight blend selection for developability experts."""

from __future__ import annotations

import itertools
import statistics


def positional_identity(query: str, reference: str) -> float:
    """Return aligned identity for equal-length sequences, otherwise zero."""
    query = str(query or "").strip().upper()
    reference = str(reference or "").strip().upper()
    if not query or len(query) != len(reference):
        return 0.0
    return sum(left == right for left, right in zip(query, reference)) / len(query)


def choose_specialist(
    vh: str,
    references: dict[str, list[str]],
    threshold: float = 0.90,
) -> dict | None:
    """Choose the closest qualifying family without accepting a source label."""
    eligible = []
    for name in sorted(references):
        identity = max(
            (positional_identity(vh, reference) for reference in references[name]),
            default=0.0,
        )
        if identity >= threshold:
            eligible.append((identity, name))
    if not eligible:
        return None
    identity, name = min(eligible, key=lambda item: (-item[0], item[1]))
    return {"name": name, "identity": identity}


def candidate_combinations(names: list[str]) -> list[tuple[str, ...]]:
    """Return deterministic non-empty equal-weight candidate combinations."""
    ordered = sorted(set(names))
    return [
        combination
        for size in range(1, len(ordered) + 1)
        for combination in itertools.combinations(ordered, size)
    ]


def select_blend(
    predictions: dict[str, dict[str, dict[str, tuple[int, float]]]],
    min_coverage: float = 0.80,
) -> dict:
    """Select by unweighted inner-source AUC; prefer fewer components on ties."""
    from sklearn.metrics import roc_auc_score

    if not predictions:
        raise ValueError("blend selection requires at least one source")
    common_names = sorted(set.intersection(*(set(items) for items in predictions.values())))
    eligible = []
    for name in common_names:
        if all(
            len(items[name]) >= min_coverage * max(map(len, items.values()))
            for items in predictions.values()
        ):
            eligible.append(name)
    if not eligible:
        raise ValueError("no candidate has sufficient common prediction coverage")

    ranked = []
    for members in candidate_combinations(eligible):
        source_auc = {}
        for source, items in predictions.items():
            record_ids = sorted(set.intersection(*(set(items[name]) for name in members)))
            labels = [items[members[0]][record_id][0] for record_id in record_ids]
            if len(set(labels)) != 2:
                continue
            if any(
                items[name][record_id][0] != label
                for name in members
                for record_id, label in zip(record_ids, labels)
            ):
                raise ValueError(f"candidate labels disagree for source {source}")
            scores = [
                statistics.fmean(items[name][record_id][1] for name in members)
                for record_id in record_ids
            ]
            source_auc[source] = float(roc_auc_score(labels, scores))
        if source_auc:
            ranked.append(
                (statistics.fmean(source_auc.values()), len(members), members, source_auc)
            )
    if not ranked:
        raise ValueError("no candidate combination has two-class source predictions")
    macro_auc, _, members, source_auc = min(
        ranked, key=lambda item: (-item[0], item[1], item[2])
    )
    return {
        "members": list(members),
        "macro_source_auc": macro_auc,
        "source_auc": source_auc,
        "eligible_candidates": eligible,
    }
