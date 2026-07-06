"""
Run all predictors and compute a disagreement score.
High disagreement = flag for wet-lab testing.
"""

from predictors.camsol import mean_camsol_score
from predictors.esmfold import fold_sequence
from features import sequence as seq_features
from features import structure as struct_features


def run_all(sequence: str) -> dict:
    """
    Run all available predictors on a sequence.
    Returns raw scores + disagreement flag.
    """
    results = {}

    # sequence-based (fast, no API)
    results["sequence_features"] = seq_features.compute_all(sequence)
    results["camsol_score"] = mean_camsol_score(sequence)

    # structure-based (ESMFold API call)
    fold_result = fold_sequence(sequence)
    if fold_result:
        plddt = fold_result["plddt_scores"]
        results["structure_features"] = struct_features.compute_all(plddt)
        results["pdb_string"] = fold_result["pdb_string"]
    else:
        results["structure_features"] = None
        results["pdb_string"] = None

    results["disagreement_score"] = _compute_disagreement(results)
    results["flag_for_testing"] = results["disagreement_score"] > 0.5

    return results


def _compute_disagreement(results: dict) -> float:
    """
    Disagreement score between 0 and 1.
    High = predictors are contradicting each other = worth testing.

    Logic:
    - CamSol says soluble (positive) but pLDDT variance is high = disagreement
    - CamSol says aggregation-prone (negative) but structure looks confident = disagreement
    """
    camsol = results.get("camsol_score", 0.0)
    struct = results.get("structure_features")

    if struct is None:
        # can't compute disagreement without structure
        return 0.0

    plddt_var = struct.get("plddt_variance", 0.0)
    low_conf = struct.get("low_confidence_fraction", 0.0)

    # normalize camsol to 0-1 range (rough)
    camsol_risk = max(0.0, min(1.0, (-camsol + 2) / 4))

    # structure risk from pLDDT
    struct_risk = min(1.0, plddt_var / 200.0 + low_conf)

    # disagreement = how far apart are the two risk estimates
    return abs(camsol_risk - struct_risk)
