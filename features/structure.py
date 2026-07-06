"""
Structure-based features — stubs for now.
These require ESMFold output (pLDDT scores, coordinates).
Run ESMFold via free API; parse results here locally.
"""


def plddt_variance(plddt_scores: list[float]) -> float:
    """
    Variance in per-residue confidence from ESMFold.
    High variance = structurally uncertain regions = aggregation risk.
    """
    if not plddt_scores:
        return 0.0
    mean = sum(plddt_scores) / len(plddt_scores)
    return sum((s - mean) ** 2 for s in plddt_scores) / len(plddt_scores)


def mean_plddt(plddt_scores: list[float]) -> float:
    """Overall structure confidence."""
    return sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0


def low_confidence_fraction(plddt_scores: list[float], threshold: float = 70.0) -> float:
    """Fraction of residues below confidence threshold."""
    if not plddt_scores:
        return 0.0
    low = sum(1 for s in plddt_scores if s < threshold)
    return low / len(plddt_scores)


def compute_all(plddt_scores: list[float]) -> dict:
    """Compute all structure features from ESMFold pLDDT output."""
    return {
        "plddt_mean": mean_plddt(plddt_scores),
        "plddt_variance": plddt_variance(plddt_scores),
        "low_confidence_fraction": low_confidence_fraction(plddt_scores),
    }
