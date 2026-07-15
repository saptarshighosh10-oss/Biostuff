"""Plan C evaluation metrics (design §9), pure stdlib.

No numpy/scipy/sklearn — the eval environment has none. Everything here is
plain Python over lists of floats/ints so it runs anywhere Python 3 does.

Scope discipline carried from the spec:
- Spearman is per assay endpoint + study; only macro-aggregate across
  commensurate endpoints (that aggregation lives at the caller, not here).
- MAE is same-physical-unit only (see `mae` docstring).
- CIs come from resampling GROUPS, never rows (`group_bootstrap_ci`).
- Calibration is evaluated only around a predeclared physical threshold and
  abstains below a declared minimum count (`calibration_at_threshold`).
"""

from __future__ import annotations

import math
import random
from typing import Callable


def _average_ranks(values: list[float]) -> list[float]:
    """1-based ranks with ties broken by average rank (standard for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # mean of 0-based positions i..j, +1 for 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0.0 or vb == 0.0:
        return 0.0  # zero variance in either input is degenerate
    return cov / math.sqrt(va * vb)


def spearman_rho(y_true: list[float], y_pred: list[float]) -> float:
    """Spearman rank correlation with average-rank tie handling.

    Returns 0.0 for degenerate input (n < 2, or zero variance / all-tied ranks
    on either side).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if len(y_true) < 2:
        return 0.0
    return _pearson(_average_ranks(y_true), _average_ranks(y_pred))


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean absolute error.

    WARNING: valid only within ONE physical unit / assay family. Do NOT pool
    MAE across unlike units or assays — a titer error in mg/mL and a Tm error
    in °C are not commensurate and their mean is meaningless (design §9.2).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def topk_enrichment(
    y_true_binary: list[int], scores: list[float], k_fraction: float
) -> float:
    """Enrichment factor of positives in the top-`k_fraction` by score.

    Ranks items by `scores` descending, takes the top k = round(k_fraction*n),
    and returns (fraction of positives captured in the top-k) / base_rate,
    where base_rate = positives / n (design §9.4, K in {0.01, 0.05, 0.10}).

    With all positives ranked at the top this equals 1/base_rate. Returns 0.0
    when there are no positives or no items.
    """
    if len(y_true_binary) != len(scores):
        raise ValueError("y_true_binary and scores must be the same length")
    n = len(scores)
    positives = sum(1 for y in y_true_binary if y)
    if n == 0 or positives == 0:
        return 0.0
    k = max(1, round(k_fraction * n))
    # ponytail: stable descending sort; ties at the top-k boundary are not
    # broken specially. Add tie-aware averaging only if a real dataset needs it.
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    captured = sum(1 for i in order[:k] if y_true_binary[i])
    recall = captured / positives
    base_rate = positives / n
    return recall / base_rate


def group_bootstrap_ci(
    values_by_group: dict[str, list[tuple[float, float]]],
    metric_fn: Callable[[list[float], list[float]], float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Group-level bootstrap CI (design §9.3): resample GROUPS, not rows.

    Each group holds a list of (y_true, y_pred) pairs. The point estimate is
    `metric_fn` over all pooled pairs. Each of `n_boot` iterations resamples the
    groups with replacement, pools their pairs, and recomputes `metric_fn`.

    Returns (point_estimate, ci_low, ci_high) as a (1-alpha) percentile
    interval. Deterministic for a given `seed`.
    """
    groups = list(values_by_group.keys())
    if not groups:
        raise ValueError("values_by_group must contain at least one group")

    def _pooled_metric(chosen: list[str]) -> float:
        yt: list[float] = []
        yp: list[float] = []
        for g in chosen:
            for t, p in values_by_group[g]:
                yt.append(t)
                yp.append(p)
        return metric_fn(yt, yp)

    point = _pooled_metric(groups)

    rng = random.Random(seed)
    boots = [
        _pooled_metric([rng.choice(groups) for _ in range(len(groups))])
        for _ in range(n_boot)
    ]
    boots.sort()

    def _pct(q: float) -> float:
        idx = int(round(q * (n_boot - 1)))
        return boots[min(max(idx, 0), n_boot - 1)]

    return point, _pct(alpha / 2.0), _pct(1.0 - alpha / 2.0)


def calibration_at_threshold(
    y_true_binary: list[int],
    probs: list[float],
    threshold: float,
    min_count: int = 20,
) -> dict:
    """Calibration around ONE predeclared decision threshold (design §9.5).

    Predicts positive when prob >= threshold and reports:
      - predicted_positive_rate: fraction of items predicted positive
      - observed_positive_rate_in_predicted_pos: true-positive fraction among
        those predicted positive (None if none are predicted positive)

    Abstains ({"abstain": True, "reason": ...}) when n < `min_count`, because
    calibration on tiny samples is noise, not a diagnostic.
    """
    if len(y_true_binary) != len(probs):
        raise ValueError("y_true_binary and probs must be the same length")
    n = len(probs)
    if n < min_count:
        return {
            "abstain": True,
            "reason": f"n={n} below min_count={min_count}",
        }
    predicted_pos = [i for i, p in enumerate(probs) if p >= threshold]
    n_pred = len(predicted_pos)
    if n_pred == 0:
        observed = None  # no predicted positives -> rate undefined
    else:
        observed = sum(1 for i in predicted_pos if y_true_binary[i]) / n_pred
    return {
        "threshold": threshold,
        "predicted_positive_rate": n_pred / n,
        "observed_positive_rate_in_predicted_pos": observed,
        "n": n,
    }
