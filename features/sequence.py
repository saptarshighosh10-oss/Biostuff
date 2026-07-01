"""
Sequence-based physical property features.
All CPU, no GPU needed, no external API calls.
"""

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Charge at pH 7.4
CHARGE = {
    "R": 1.0, "K": 1.0, "H": 0.1,
    "D": -1.0, "E": -1.0,
}


def hydrophobicity_score(seq: str) -> float:
    """Mean hydrophobicity across the sequence."""
    scores = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq.upper()]
    return sum(scores) / len(scores) if scores else 0.0


def net_charge(seq: str) -> float:
    """Approximate net charge at pH 7.4."""
    return sum(CHARGE.get(aa, 0.0) for aa in seq.upper())


def charge_density(seq: str) -> float:
    """Net charge normalized by sequence length."""
    return net_charge(seq) / len(seq) if seq else 0.0


def hydrophobic_patch_score(seq: str, window: int = 5) -> float:
    """
    Max mean hydrophobicity over any sliding window.
    High scores indicate exposed hydrophobic patches — a key aggregation driver.
    """
    if len(seq) < window:
        return hydrophobicity_score(seq)

    scores = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq.upper()]
    windows = [scores[i:i + window] for i in range(len(scores) - window + 1)]
    return max(sum(w) / window for w in windows)


def compute_all(seq: str) -> dict:
    """Compute all sequence features at once."""
    return {
        "hydrophobicity_mean": hydrophobicity_score(seq),
        "net_charge": net_charge(seq),
        "charge_density": charge_density(seq),
        "hydrophobic_patch_score": hydrophobic_patch_score(seq),
        "length": len(seq),
    }
