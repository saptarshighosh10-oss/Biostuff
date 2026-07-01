"""
CDR extraction heuristic for antibody VH/VL sequences.
Uses conserved Cys landmarks to approximate Chothia CDR positions.
For production use, replace with ANARCI (pip install anarci).
"""

from dataclasses import dataclass


@dataclass
class CDRRegions:
    cdr1: str
    cdr2: str
    cdr3: str
    cdr1_start: int
    cdr2_start: int
    cdr3_start: int


def find_conserved_cysteines(seq: str) -> list[int]:
    """
    Find positions of conserved Cys residues that bracket the variable domain.
    VH/VL sequences typically have Cys at ~position 23 and ~92 (Kabat numbering).
    """
    positions = [i for i, aa in enumerate(seq) if aa == "C"]
    if len(positions) >= 2:
        return [positions[0], positions[-1]]
    return positions


def extract_cdrs(seq: str, chain_type: str = "unknown") -> CDRRegions | None:
    """
    Approximate CDR extraction using conserved Cys landmark positions.
    Offsets are based on Chothia canonical loop positions relative to Cys23.

    Returns None if conserved cysteines can't be located.
    """
    seq = seq.upper()
    cys_positions = find_conserved_cysteines(seq)

    if len(cys_positions) < 2:
        return None

    cys1 = cys_positions[0]

    # CDR offsets relative to first conserved Cys (Chothia approximate)
    cdr1_start = cys1 + 1
    cdr1_end = cdr1_start + 10

    cdr2_start = cys1 + 15
    cdr2_end = cdr2_start + 7

    cdr3_start = cys1 + 35
    cdr3_end = cdr3_start + 12

    n = len(seq)

    def safe_slice(start: int, end: int) -> str:
        return seq[max(0, start):min(n, end)]

    return CDRRegions(
        cdr1=safe_slice(cdr1_start, cdr1_end),
        cdr2=safe_slice(cdr2_start, cdr2_end),
        cdr3=safe_slice(cdr3_start, cdr3_end),
        cdr1_start=cdr1_start,
        cdr2_start=cdr2_start,
        cdr3_start=cdr3_start,
    )


def cdr_positions(seq: str) -> set[int]:
    """Return set of sequence positions that fall within CDR regions."""
    cdrs = extract_cdrs(seq)
    if cdrs is None:
        return set()
    positions = set()
    for start, region in [
        (cdrs.cdr1_start, cdrs.cdr1),
        (cdrs.cdr2_start, cdrs.cdr2),
        (cdrs.cdr3_start, cdrs.cdr3),
    ]:
        positions.update(range(start, start + len(region)))
    return positions
