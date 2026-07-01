"""
CamSol intrinsic solubility score — sequence-based, CPU only.
Approximation of the published method (Sormanni et al. 2015).
For production use, run the official CamSol server.
"""

from features.sequence import HYDROPHOBICITY, CHARGE


def camsol_score(seq: str, window: int = 9) -> list[float]:
    """
    Per-residue CamSol-like solubility score using a sliding window.
    Negative = aggregation prone. Positive = soluble.
    Combines hydrophobicity and charge contributions.
    """
    seq = seq.upper()
    n = len(seq)
    scores = []

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        window_seq = seq[start:end]

        hydro = sum(HYDROPHOBICITY.get(aa, 0.0) for aa in window_seq) / len(window_seq)
        charge = sum(abs(CHARGE.get(aa, 0.0)) for aa in window_seq) / len(window_seq)

        # charged residues improve solubility; hydrophobic ones reduce it
        scores.append(charge - hydro)

    return scores


def mean_camsol_score(seq: str) -> float:
    """Overall sequence solubility estimate."""
    scores = camsol_score(seq)
    return sum(scores) / len(scores) if scores else 0.0
