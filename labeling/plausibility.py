"""
Physical plausibility checks for auto-labeling.
A label is only trusted when physical property shift agrees with mutation location.
"""

from features.sequence import HYDROPHOBICITY, CHARGE, hydrophobic_patch_score, net_charge

HYDROPHOBICITY_SHIFT_THRESHOLD = 1.5
CHARGE_SHIFT_THRESHOLD = 1.0


def hydrophobicity_shift(seq_a: str, seq_b: str) -> float:
    """Change in max hydrophobic patch score between working and failed sequence."""
    return hydrophobic_patch_score(seq_b) - hydrophobic_patch_score(seq_a)


def charge_shift(seq_a: str, seq_b: str) -> float:
    """Change in net charge between working and failed sequence."""
    return net_charge(seq_b) - net_charge(seq_a)


def is_physically_plausible(
    seq_working: str,
    seq_failed: str,
    mutations: list[tuple[int, str, str]],
) -> tuple[bool, str]:
    """
    Check whether physical property shifts are consistent with the observed mutations.
    Returns (plausible: bool, reason: str).

    A label is plausible if:
    - Hydrophobicity increased significantly at the mutation sites, OR
    - Charge shifted significantly (loss of stabilizing charges)
    """
    hydro_delta = hydrophobicity_shift(seq_working, seq_failed)
    charge_delta = charge_shift(seq_working, seq_failed)

    if hydro_delta > HYDROPHOBICITY_SHIFT_THRESHOLD:
        return True, f"hydrophobic patch increased by {hydro_delta:.2f}"

    if abs(charge_delta) > CHARGE_SHIFT_THRESHOLD:
        direction = "lost" if charge_delta < 0 else "gained"
        return True, f"net charge {direction} {abs(charge_delta):.1f} units"

    return False, "no significant physical property shift detected"
