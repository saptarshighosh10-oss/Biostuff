"""
BLOSUM62-guided sequence perturbation.
Generates mutations that are statistically plausible (similar to ProteinMPNN proposals)
rather than random noise. CDR positions are perturbed preferentially.
"""

import random
from features.cdr import cdr_positions

# BLOSUM62 substitution scores — higher = more conservative
# Only includes substitutions with score >= 0 (plausible replacements)
BLOSUM62_SUBSTITUTIONS: dict[str, list[tuple[str, int]]] = {
    "A": [("S", 1), ("T", 0), ("V", 0), ("G", 0)],
    "R": [("K", 2), ("Q", 1), ("H", 0)],
    "N": [("D", 1), ("S", 1), ("T", 0)],
    "D": [("E", 2), ("N", 1)],
    "C": [("S", 0)],
    "Q": [("E", 2), ("R", 1), ("K", 1), ("H", 0)],
    "E": [("D", 2), ("Q", 2), ("K", 1)],
    "G": [("A", 0)],
    "H": [("Y", 2), ("N", 1), ("Q", 0), ("R", 0)],
    "I": [("L", 2), ("V", 3), ("M", 1)],
    "L": [("I", 2), ("V", 1), ("M", 2), ("F", 0)],
    "K": [("R", 2), ("Q", 1), ("E", 1)],
    "M": [("L", 2), ("I", 1), ("V", 1)],
    "F": [("Y", 3), ("W", 1), ("L", 0)],
    "P": [("A", 0)],
    "S": [("T", 1), ("A", 1), ("N", 1)],
    "T": [("S", 1), ("A", 0), ("V", 0)],
    "W": [("Y", 2), ("F", 1)],
    "Y": [("F", 3), ("H", 2), ("W", 2)],
    "V": [("I", 3), ("L", 1), ("M", 1), ("T", 0), ("A", 0)],
}


def _sample_substitution(aa: str, seed: int | None = None) -> str:
    """Sample a plausible substitution for an amino acid using BLOSUM62 weights."""
    rng = random.Random(seed)
    candidates = BLOSUM62_SUBSTITUTIONS.get(aa.upper(), [])
    if not candidates:
        return aa
    amino_acids, weights = zip(*candidates)
    # higher BLOSUM62 score = more likely to be sampled
    adjusted_weights = [w + 1 for w in weights]
    return rng.choices(amino_acids, weights=adjusted_weights, k=1)[0]


def generate_variant(
    sequence: str,
    n_mutations: int = 1,
    prefer_cdrs: bool = True,
    seed: int | None = None,
) -> tuple[str, list[tuple[int, str, str]]]:
    """
    Generate one mutated variant of the sequence.

    Args:
        sequence: parent sequence
        n_mutations: number of mutations to introduce (1-3 recommended)
        prefer_cdrs: if True, bias mutations toward CDR positions
        seed: random seed for reproducibility

    Returns:
        (mutant_sequence, mutations) where mutations = [(pos, original, mutant), ...]
    """
    rng = random.Random(seed)
    seq = list(sequence.upper())
    n = len(seq)

    cdr_pos = cdr_positions(sequence) if prefer_cdrs else set()

    # build position weights: CDR positions 3x more likely to be mutated
    weights = [3.0 if i in cdr_pos else 1.0 for i in range(n)]

    # don't mutate Cys (framework-stabilizing disulfides)
    weights = [0.0 if seq[i] == "C" else w for i, w in enumerate(weights)]

    total = sum(weights)
    if total == 0:
        return sequence, []

    norm_weights = [w / total for w in weights]

    positions = rng.choices(range(n), weights=norm_weights, k=min(n_mutations, n))
    positions = list(set(positions))

    mutations = []
    for pos in positions:
        original = seq[pos]
        mutant = _sample_substitution(original, seed=seed)
        if mutant != original:
            seq[pos] = mutant
            mutations.append((pos, original, mutant))

    return "".join(seq), mutations


def generate_variants(
    sequence: str,
    n_variants: int = 5,
    mutations_per_variant: int = 2,
    seed: int = 42,
) -> list[tuple[str, list[tuple[int, str, str]]]]:
    """
    Generate multiple variants for a single parent sequence.
    Each variant gets a different seed to ensure diversity.
    """
    return [
        generate_variant(sequence, n_mutations=mutations_per_variant, seed=seed + i)
        for i in range(n_variants)
    ]
