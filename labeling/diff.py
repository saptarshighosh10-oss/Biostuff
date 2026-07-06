"""
Mutation diff between a failed variant and its nearest working neighbor.
Only trust labels when mutation distance is small (few residues).
"""

MAX_ATTRIBUTABLE_MUTATIONS = 3


def sequence_diff(seq_a: str, seq_b: str) -> list[tuple[int, str, str]]:
    """
    Compare two same-length sequences residue by residue.
    Returns list of (position, original_aa, mutated_aa).
    """
    if len(seq_a) != len(seq_b):
        raise ValueError("Sequences must be the same length for diff.")

    return [
        (i, a, b)
        for i, (a, b) in enumerate(zip(seq_a.upper(), seq_b.upper()))
        if a != b
    ]


def mutation_distance(seq_a: str, seq_b: str) -> int:
    """Number of differing positions between two sequences."""
    return len(sequence_diff(seq_a, seq_b))


def is_attributable(seq_failed: str, seq_working: str) -> bool:
    """
    True if mutation distance is small enough that cause is attributable.
    Too many mutations = can't isolate the reason for failure.
    """
    return mutation_distance(seq_failed, seq_working) <= MAX_ATTRIBUTABLE_MUTATIONS
