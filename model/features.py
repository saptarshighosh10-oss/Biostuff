"""
Feature extraction for the failure model.
Produces a flat ~21-feature vector from any candidate dict or raw sequence.
Missing fields (e.g. no ESMFold, no Phase 2) default to 0 so the model still runs.
"""

from features.sequence import compute_all as seq_compute_all
from predictors.camsol import mean_camsol_score

HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGE = {"R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}

FEATURE_NAMES = [
    # sequence-based (always computable)
    "hydrophobicity_mean",
    "net_charge",
    "charge_density",
    "hydrophobic_patch_score",
    "camsol_score",
    # mutation-based
    "n_mutations",
    "mean_hydrophobicity_delta",
    "total_charge_delta",
    "max_hydrophobicity_delta",
    # Phase 1 risk (0 if unavailable)
    "hydrophobicity_risk",
    "charge_risk",
    "camsol_risk",
    "combined_risk",
    "plddt_mean",
    "plddt_variance",
    "low_confidence_fraction",
    "disagreement_score",
    # Phase 2 multi-predictor (0 if unavailable)
    "tango_mean",
    "aggrescan_mean",
    "zyggregator_mean",
    "n_consensus_hotspots",
    "mutation_hits_hotspot",
]


def extract_features(candidate: dict) -> dict:
    """
    Extract named features from a candidate dict (Phase 1 or Phase 2 output).
    Any missing field is filled with 0.
    """
    seq = candidate.get("variant_sequence") or candidate.get("sequence", "")
    mutations = candidate.get("mutations", [])
    risk = candidate.get("risk", {})
    struct = risk.get("structure_features") or {}
    mp = candidate.get("multi_predictor", {})

    # Compute the multi-predictor features in-process if absent, so training
    # and inference featurize identically (no train/serve skew). These are the
    # TANGO/AGGRESCAN/Zyggregator aggregation signals — the most domain-relevant
    # features, previously zeroed for every training sample.
    if not mp and seq:
        from predictors.multi_predictor import run_all_predictors
        mp = run_all_predictors(seq)

    # sequence features
    seq_feats = seq_compute_all(seq) if seq else {}
    camsol = mean_camsol_score(seq) if seq else 0.0

    # mutation deltas
    hydro_deltas = [
        HYDROPHOBICITY.get(mut, 0.0) - HYDROPHOBICITY.get(orig, 0.0)
        for _, orig, mut in mutations
    ]
    charge_deltas = [
        CHARGE.get(mut, 0.0) - CHARGE.get(orig, 0.0)
        for _, orig, mut in mutations
    ]

    # Phase 2 multi-predictor
    tango = mp.get("tango", {})
    aggrescan = mp.get("aggrescan", {})
    zygg = mp.get("zyggregator", {})
    consensus = mp.get("consensus_hotspots", [])
    from predictors.multi_predictor import mutation_hits_hotspot
    hits_hotspot = 1.0 if (consensus and mutation_hits_hotspot(mutations, consensus)) else 0.0

    return {
        "hydrophobicity_mean": seq_feats.get("hydrophobicity_mean", 0.0),
        "net_charge": seq_feats.get("net_charge", 0.0),
        "charge_density": seq_feats.get("charge_density", 0.0),
        "hydrophobic_patch_score": seq_feats.get("hydrophobic_patch_score", 0.0),
        "camsol_score": camsol,
        "n_mutations": float(len(mutations)),
        "mean_hydrophobicity_delta": sum(hydro_deltas) / len(hydro_deltas) if hydro_deltas else 0.0,
        "total_charge_delta": sum(charge_deltas),
        "max_hydrophobicity_delta": max(hydro_deltas, default=0.0),
        "hydrophobicity_risk": risk.get("hydrophobicity_risk", 0.0),
        "charge_risk": risk.get("charge_risk", 0.0),
        "camsol_risk": risk.get("camsol_risk", 0.0),
        "combined_risk": risk.get("combined_risk", 0.0),
        "plddt_mean": struct.get("plddt_mean", 0.0),
        "plddt_variance": struct.get("plddt_variance", 0.0),
        "low_confidence_fraction": struct.get("low_confidence_fraction", 0.0),
        "disagreement_score": risk.get("disagreement_score", 0.0),
        "tango_mean": tango.get("mean_score", 0.0),
        "aggrescan_mean": aggrescan.get("mean_score", 0.0),
        "zyggregator_mean": zygg.get("mean_score", 0.0),
        "n_consensus_hotspots": float(len(consensus)),
        "mutation_hits_hotspot": hits_hotspot,
    }


def features_to_vector(feat_dict: dict) -> list[float]:
    """Return feature values in the canonical FEATURE_NAMES order."""
    return [feat_dict.get(name, 0.0) for name in FEATURE_NAMES]


def extract_from_sequence(sequence: str, mutations: list | None = None) -> dict:
    """
    Extract features from a raw sequence with no Phase 1/2 data.
    Used for fast inference on new sequences. extract_features now computes the
    multi-predictor signals in-process, so this is a thin wrapper — the training
    and inference paths share one identical featurization.
    """
    return extract_features({
        "variant_sequence": sequence,
        "mutations": mutations or [],
    })
