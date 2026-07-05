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

# Chou-Fasman beta-sheet propensity — high-propensity residues favor the
# cross-beta sheet structure that underlies amyloid/aggregate cores.
BETA_SHEET_PROPENSITY = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Aromatic residues drive pi-stacking, a common aggregation-nucleating
# interaction in cross-beta cores (e.g. the FF motif in amyloid-beta KLVFFAE).
AROMATIC = set("FWY")

# Strongly hydrophobic/beta-branched residues most likely to form buried
# aggregation-prone stretches.
STRONG_HYDROPHOBIC = set("VILMFWYC")

# "Gatekeeper" residues (proline, glycine, charged) are known to interrupt
# aggregation-prone stretches by breaking beta-sheet propensity or adding
# repulsive charge — their local density is a protective signal.
GATEKEEPERS = set("PGDEKR")


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


def aromatic_fraction(seq: str) -> float:
    """Fraction of residues that are aromatic (F/W/Y) — pi-stacking drives aggregation."""
    if not seq:
        return 0.0
    return sum(1 for aa in seq.upper() if aa in AROMATIC) / len(seq)


def beta_sheet_propensity(seq: str) -> float:
    """Mean Chou-Fasman beta-sheet propensity — high = favors cross-beta aggregation."""
    scores = [BETA_SHEET_PROPENSITY.get(aa, 1.0) for aa in seq.upper()]
    return sum(scores) / len(scores) if scores else 0.0


def longest_hydrophobic_run(seq: str) -> float:
    """Longest contiguous stretch of strongly hydrophobic residues."""
    longest = current = 0
    for aa in seq.upper():
        if aa in STRONG_HYDROPHOBIC:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest)


def gatekeeper_density(seq: str) -> float:
    """Fraction of proline/glycine/charged 'gatekeeper' residues — protective, inverse risk."""
    if not seq:
        return 0.0
    return sum(1 for aa in seq.upper() if aa in GATEKEEPERS) / len(seq)


def max_beta_aromatic_patch(seq: str, window: int = 5) -> float:
    """
    Max sliding-window score combining beta-sheet propensity and aromatic content —
    captures aromatic cross-beta "zipper" motifs (e.g. KLVFFAE in amyloid-beta).
    """
    seq = seq.upper()
    if len(seq) < window:
        return beta_sheet_propensity(seq) + aromatic_fraction(seq)

    best = 0.0
    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        beta = sum(BETA_SHEET_PROPENSITY.get(aa, 1.0) for aa in w) / window
        arom = sum(1 for aa in w if aa in AROMATIC) / window
        best = max(best, beta + arom)
    return best


def compute_all(seq: str) -> dict:
    """Compute all sequence features at once."""
    return {
        "hydrophobicity_mean": hydrophobicity_score(seq),
        "net_charge": net_charge(seq),
        "charge_density": charge_density(seq),
        "hydrophobic_patch_score": hydrophobic_patch_score(seq),
        "aromatic_fraction": aromatic_fraction(seq),
        "beta_sheet_propensity": beta_sheet_propensity(seq),
        "longest_hydrophobic_run": longest_hydrophobic_run(seq),
        "gatekeeper_density": gatekeeper_density(seq),
        "max_beta_aromatic_patch": max_beta_aromatic_patch(seq),
        "length": len(seq),
    }
