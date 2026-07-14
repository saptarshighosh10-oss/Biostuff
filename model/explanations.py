"""Deterministic, evidence-bounded explanations for model predictions."""

from __future__ import annotations


def _mutation_text(mutations: list[tuple[int, str, str]]) -> str:
    return ", ".join(f"{old}{position + 1}{new}" for position, old, new in mutations)


def explain_failure(
    *,
    mutations: list[tuple[int, str, str]],
    features: dict,
    probability: float,
    confidence: str,
    top_features: list[dict],
) -> dict:
    """Return facts and cautious hypotheses; never claim a proven mechanism."""
    evidence: list[str] = []
    hypotheses: list[str] = []

    if mutations:
        evidence.append(f"variant mutations: {_mutation_text(mutations)}")
    if features.get("mutation_hits_hotspot"):
        evidence.append("a mutation overlaps a consensus aggregation hotspot")
        hypotheses.append("the substitution may strengthen a locally aggregation-prone region")
    if features.get("mean_hydrophobicity_delta", 0.0) >= 0.5:
        evidence.append("mutations increase mean hydrophobicity")
        hypotheses.append("the added hydrophobicity may expose a self-association surface")
    if abs(features.get("total_charge_delta", 0.0)) >= 1.0:
        evidence.append("mutations change the estimated net charge")
        hypotheses.append("the charge change may reduce electrostatic repulsion")

    drivers = {item["feature"] for item in top_features[:3]}
    if "hydrophobic_patch_score" in drivers or features.get("hydrophobic_patch_score", 0.0) >= 2.5:
        evidence.append("the sequence contains a strong hydrophobic patch")
        hypotheses.append("a hydrophobic patch may increase aggregation propensity")
    if "max_beta_aromatic_patch" in drivers or features.get("max_beta_aromatic_patch", 0.0) >= 1.6:
        evidence.append("the sequence contains a beta/aromatic-rich local patch")
        hypotheses.append("beta-sheet and aromatic interactions may support aggregate nucleation")

    evidence.append(f"model failure probability: {probability:.3f}")
    if not evidence:
        evidence.append("no specific mechanistic feature crossed the explanation thresholds")
    if not hypotheses:
        hypotheses.append("the model has no strong mechanistic signal beyond its learned sequence pattern")

    certainty = "inferred" if hypotheses else "unknown"
    description = "; ".join(hypotheses[:2]) + ". This is a computational hypothesis, not a proven cause."
    return {
        "certainty": certainty,
        "evidence": evidence,
        "likely_mechanisms": hypotheses,
        "description": description,
        "confidence": confidence,
    }
