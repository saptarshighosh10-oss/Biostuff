"""
Sequence perturbation for variant generation.

Two modes:
  ESM-2 guided (preferred): uses protein language model probabilities to propose
    mutations that are evolutionarily plausible — directly inspired by EVOLVEpro.
  BLOSUM62 fallback: used when ESM-2 is not installed.

ESM-2 install: pip install fair-esm
"""

import random
from features.cdr import cdr_positions

# ── BLOSUM62 fallback ────────────────────────────────────────────────────────

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

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _blosum62_substitution(aa: str, rng: random.Random) -> str:
    candidates = BLOSUM62_SUBSTITUTIONS.get(aa.upper(), [])
    if not candidates:
        return aa
    amino_acids, weights = zip(*candidates)
    return rng.choices(amino_acids, weights=[w + 1 for w in weights], k=1)[0]


# ── ESM-2 guided mutations ───────────────────────────────────────────────────

def load_esm2():
    """
    Load the smallest ESM-2 model (8M params, ~25MB).
    Returns (model, alphabet) or (None, None) if fair-esm not installed.
    """
    try:
        import esm
        import torch
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model, alphabet
    except ImportError:
        return None, None


def _esm2_substitutions(
    model, alphabet, sequence: str, position: int, top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Get ESM-2's top-k predicted amino acids at a masked position.
    High probability = evolutionarily plausible substitution.
    """
    import torch

    seq_masked = sequence[:position] + "<mask>" + sequence[position + 1:]
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", seq_masked)])

    if next(model.parameters()).is_cuda:
        tokens = tokens.cuda()

    with torch.no_grad():
        results = model(tokens, repr_layers=[])
        logits = results["logits"][0, position + 1]  # +1 for <cls> token
        probs = torch.softmax(logits, dim=-1)

    original = sequence[position].upper()
    aa_probs = []
    for aa in AMINO_ACIDS:
        if aa == original or aa == "C":  # don't suggest Cys mutations
            continue
        idx = alphabet.get_idx(aa)
        aa_probs.append((aa, probs[idx].item()))

    aa_probs.sort(key=lambda x: x[1], reverse=True)
    return aa_probs[:top_k]


# ── Position sampling (shared) ───────────────────────────────────────────────

def _sample_positions(sequence: str, n: int, prefer_cdrs: bool, rng: random.Random) -> list[int]:
    seq = sequence.upper()
    cdr_pos = cdr_positions(sequence) if prefer_cdrs else set()
    weights = [3.0 if i in cdr_pos else 1.0 for i in range(len(seq))]
    weights = [0.0 if seq[i] == "C" else w for i, w in enumerate(weights)]
    total = sum(weights)
    if total == 0:
        return []
    norm = [w / total for w in weights]
    return list(set(rng.choices(range(len(seq)), weights=norm, k=n)))


# ── Public API ───────────────────────────────────────────────────────────────

def generate_variant(
    sequence: str,
    n_mutations: int = 2,
    prefer_cdrs: bool = True,
    seed: int | None = None,
    esm2_model=None,
    esm2_alphabet=None,
) -> tuple[str, list[tuple[int, str, str]]]:
    """
    Generate one mutated variant.
    Uses ESM-2 if model is provided, otherwise falls back to BLOSUM62.
    """
    rng = random.Random(seed)
    seq = list(sequence.upper())
    positions = _sample_positions(sequence, n_mutations, prefer_cdrs, rng)

    mutations = []
    for pos in positions:
        original = seq[pos]

        if esm2_model is not None:
            subs = _esm2_substitutions(esm2_model, esm2_alphabet, "".join(seq), pos)
            if subs:
                candidates, probs = zip(*subs)
                mutant = rng.choices(candidates, weights=probs, k=1)[0]
            else:
                mutant = _blosum62_substitution(original, rng)
        else:
            mutant = _blosum62_substitution(original, rng)

        if mutant != original:
            seq[pos] = mutant
            mutations.append((pos, original, mutant))

    return "".join(seq), mutations


def generate_variants(
    sequence: str,
    n_variants: int = 5,
    mutations_per_variant: int = 2,
    seed: int = 42,
    esm2_model=None,
    esm2_alphabet=None,
) -> list[tuple[str, list[tuple[int, str, str]]]]:
    """Generate multiple variants, each with a different seed for diversity."""
    return [
        generate_variant(
            sequence,
            n_mutations=mutations_per_variant,
            seed=seed + i,
            esm2_model=esm2_model,
            esm2_alphabet=esm2_alphabet,
        )
        for i in range(n_variants)
    ]
