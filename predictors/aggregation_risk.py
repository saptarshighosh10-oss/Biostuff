"""
Combined aggregation risk score from all available signals.
Returns a single 0-1 score and a flag for wet-lab testing.
"""

from features import sequence as seq_features
from predictors.camsol import mean_camsol_score

# Thresholds tuned to flag the top ~20% of variants for testing
HYDROPHOBIC_PATCH_HIGH = 3.0
CAMSOL_RISK_THRESHOLD = -0.5
COMBINED_RISK_THRESHOLD = 0.6


def fast_risk_score(sequence: str) -> dict:
    """
    Compute aggregation risk from sequence features only (no API calls).
    Use this for bulk screening before calling ESMFold.
    """
    feats = seq_features.compute_all(sequence)
    camsol = mean_camsol_score(sequence)

    # normalize to 0-1
    hydro_risk = min(1.0, max(0.0, feats["hydrophobic_patch_score"] / 5.0))
    charge_risk = min(1.0, max(0.0, (2.0 - feats["charge_density"] * 10) / 4.0))
    camsol_risk = min(1.0, max(0.0, (-camsol + 2.0) / 4.0))

    # weighted combination
    combined = 0.4 * hydro_risk + 0.3 * charge_risk + 0.3 * camsol_risk

    return {
        "sequence_features": feats,
        "camsol_score": camsol,
        "hydrophobicity_risk": hydro_risk,
        "charge_risk": charge_risk,
        "camsol_risk": camsol_risk,
        "combined_risk": combined,
        "flag_for_esmfold": combined > COMBINED_RISK_THRESHOLD,
    }


def full_risk_score(sequence: str, use_esmfold: bool = True) -> dict:
    """
    Full risk assessment including ESMFold structure prediction.
    Only call ESMFold for variants already flagged by fast_risk_score.
    """
    result = fast_risk_score(sequence)

    if use_esmfold and result["flag_for_esmfold"]:
        from predictors.esmfold import fold_sequence
        from features import structure as struct_features
        from predictors.ensemble import _compute_disagreement

        fold = fold_sequence(sequence)
        if fold:
            struct = struct_features.compute_all(fold["plddt_scores"])
            result["structure_features"] = struct
            result["pdb_string"] = fold["pdb_string"]
            result["disagreement_score"] = _compute_disagreement({
                "camsol_score": result["camsol_score"],
                "structure_features": struct,
            })
            result["flag_for_wetlab"] = result["disagreement_score"] > 0.5
        else:
            result["structure_features"] = None
            result["disagreement_score"] = None
            result["flag_for_wetlab"] = result["flag_for_esmfold"]
    else:
        result["structure_features"] = None
        result["disagreement_score"] = None
        result["flag_for_wetlab"] = result["flag_for_esmfold"]

    return result
