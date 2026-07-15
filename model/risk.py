"""Phase 4 risk outputs (design §10).

The model does NOT emit a universal failure probability. It emits an
assay-contextual, threshold-conditioned risk score:

  - `risk_score_global`  — monotonic score oriented so higher = more likely to
                           fail the named assay (raw regressor output flipped to
                           risk orientation by endpoint direction).
  - `risk_decision_margin` — signed distance from the predeclared physical
                           threshold in risk space (>=0 → flagged).
  - `calibrated_decision_score` — optional calibrated P(fail) at that threshold,
                           only when a calibrator fitted at the threshold exists.
  - provenance: `assay_name`, `pair_id`, `identity_component_id`, `split_group`.

Abstain-by-default when the caller says the row is out of validated coverage
(low group support, VHH format shift, OOD distance) — see design §9/§12.
"""

from __future__ import annotations

from typing import Callable


def to_risk_orientation(value: float, direction: str) -> float:
    """Flip a raw endpoint prediction so higher always means more failure-prone.
    `higher_bad` passes through; `lower_bad` negates."""
    return value if direction == "higher_bad" else -value


def build_risk_output(
    *,
    assay_name: str,
    raw_prediction: float,
    endpoint_direction: str,
    threshold: float,
    calibrator: Callable[[float], float] | None = None,
    abstain: bool = False,
    abstain_reason: str = "",
    pair_id: str | None = None,
    identity_component_id: str | None = None,
    split_group: str | None = None,
) -> dict:
    """Assemble one risk record. Never returns a bare probability.

    `threshold` is the predeclared physical decision threshold in the assay's
    native units (same space as `raw_prediction`); it is converted to risk space
    with the same orientation so the margin is comparable across assays."""
    risk = to_risk_orientation(raw_prediction, endpoint_direction)
    threshold_risk = to_risk_orientation(threshold, endpoint_direction)
    margin = risk - threshold_risk

    out = {
        "assay_name": assay_name,
        "endpoint_direction": endpoint_direction,
        "risk_score_global": round(float(risk), 6),
        "risk_decision_margin": round(float(margin), 6),
        "pair_id": pair_id,
        "identity_component_id": identity_component_id,
        "split_group": split_group,
    }

    if abstain:
        out["decision"] = "abstain"
        out["decision_reason"] = abstain_reason or "outside validated coverage"
        out["calibrated_decision_score"] = None
        return out

    out["decision"] = "flag_failure_risk" if margin >= 0 else "pass"
    out["decision_reason"] = (
        "prediction crosses the predeclared assay threshold"
        if margin >= 0 else "prediction below the assay failure threshold"
    )
    # Calibrated P(fail) only when a threshold-fitted calibrator is supplied.
    out["calibrated_decision_score"] = (
        round(float(calibrator(raw_prediction)), 6) if calibrator is not None else None
    )
    return out
