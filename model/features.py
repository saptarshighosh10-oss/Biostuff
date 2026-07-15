"""
Feature extraction for the failure model.
Produces a flat feature vector from any candidate dict or raw sequence.
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
    "aromatic_fraction",
    "beta_sheet_propensity",
    "longest_hydrophobic_run",
    "gatekeeper_density",
    "max_beta_aromatic_patch",
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
    # Optional auxiliary Head A feature; zero when the separate head is absent.
    "proteingym_fitness_score",
]

# Head A (general ProteinGym fitness) may only use features that mean the SAME
# thing for a raw sequence with no mutations and no structure data — otherwise
# its score is computed on out-of-distribution zeros when applied to antibodies.
# Excluded: mutation-delta features (mutations=[] at serve), Phase-1 risk /
# structure features (always absent for ProteinGym rows), and its own output
# (circular). What remains is computable identically from any bare sequence.
_HEAD_A_EXCLUDED = {
    "n_mutations", "mean_hydrophobicity_delta", "total_charge_delta",
    "max_hydrophobicity_delta", "mutation_hits_hotspot",
    "hydrophobicity_risk", "charge_risk", "camsol_risk", "combined_risk",
    "plddt_mean", "plddt_variance", "low_confidence_fraction", "disagreement_score",
    "proteingym_fitness_score",
}
HEAD_A_FEATURE_NAMES = [n for n in FEATURE_NAMES if n not in _HEAD_A_EXCLUDED]


def feature_schema_hash(names: list[str] = FEATURE_NAMES) -> str:
    """Stable hash of a feature-name list, stored on saved models and asserted
    on load so a schema drift (e.g. the 27→28 column change) can't silently
    mis-score a stale artifact."""
    import hashlib
    return hashlib.sha256("\x00".join(names).encode()).hexdigest()[:16]


# Lazily-loaded, cached Head A. Shared by the training path and every serve
# path (extract_from_sequence) so ``proteingym_fitness_score`` is computed
# identically in both — the derived feature is never a train-only value.
_FITNESS_HEAD = None
_FITNESS_HEAD_TRIED = False


def general_fitness_score(sequence: str) -> float:
    """Score a raw sequence with the separate Head A; 0.0 if the head artifact
    is absent. No recursion: Head A featurizes over HEAD_A_FEATURE_NAMES via
    extract_features, which only *reads* proteingym_fitness_score from the dict
    and never calls back here."""
    global _FITNESS_HEAD, _FITNESS_HEAD_TRIED
    if not sequence:
        return 0.0
    if not _FITNESS_HEAD_TRIED:
        _FITNESS_HEAD_TRIED = True
        try:
            from model.pretrain_proteingym_fitness import load_general_fitness_model
            _FITNESS_HEAD = load_general_fitness_model()
        except Exception:  # ponytail: head is optional; any load failure → 0.0
            _FITNESS_HEAD = None
    if _FITNESS_HEAD is None:
        return 0.0
    return _FITNESS_HEAD.score_general_fitness(sequence)


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
        "aromatic_fraction": seq_feats.get("aromatic_fraction", 0.0),
        "beta_sheet_propensity": seq_feats.get("beta_sheet_propensity", 0.0),
        "longest_hydrophobic_run": seq_feats.get("longest_hydrophobic_run", 0.0),
        "gatekeeper_density": seq_feats.get("gatekeeper_density", 0.0),
        "max_beta_aromatic_patch": seq_feats.get("max_beta_aromatic_patch", 0.0),
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
        "proteingym_fitness_score": float(candidate.get("proteingym_fitness_score", 0.0) or 0.0),
    }


def features_to_vector(feat_dict: dict, names: list[str] = FEATURE_NAMES) -> list[float]:
    """Return feature values in the given name order (canonical Head-B schema by
    default; pass HEAD_A_FEATURE_NAMES for the general fitness head)."""
    return [feat_dict.get(name, 0.0) for name in names]


def extract_from_sequence(sequence: str, mutations: list | None = None) -> dict:
    """
    Extract features from a raw sequence with no Phase 1/2 data.
    Used for fast inference on new sequences. Populates proteingym_fitness_score
    via the shared Head A loader so the serve path featurizes identically to the
    training path (no train/serve skew — the value is not train-only).
    """
    return extract_features({
        "variant_sequence": sequence,
        "mutations": mutations or [],
        "proteingym_fitness_score": general_fitness_score(sequence),
    })
