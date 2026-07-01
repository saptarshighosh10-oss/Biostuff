"""
Auto-labeling with the two-signal requirement.
Only accepts a failure label when diff location AND physical plausibility agree.
Everything else gets flagged as "unresolved".
"""

from labeling.diff import sequence_diff, is_attributable
from labeling.plausibility import is_physically_plausible


LABEL_CONFIRMED_FAILURE = "confirmed_failure"
LABEL_UNRESOLVED = "unresolved"


def auto_label(seq_working: str, seq_failed: str) -> dict:
    """
    Attempt to auto-label a failure.

    Returns:
        label: "confirmed_failure" or "unresolved"
        mutations: list of (position, original, mutated)
        reason: physical explanation if confirmed, else why it's unresolved
    """
    mutations = sequence_diff(seq_working, seq_failed)

    if not mutations:
        return {
            "label": LABEL_UNRESOLVED,
            "mutations": [],
            "reason": "sequences are identical",
        }

    # signal 1: mutation distance small enough to be attributable
    if not is_attributable(seq_working, seq_failed):
        return {
            "label": LABEL_UNRESOLVED,
            "mutations": mutations,
            "reason": f"too many mutations ({len(mutations)}) to attribute cause",
        }

    # signal 2: physical property shift agrees with mutation location
    plausible, reason = is_physically_plausible(seq_working, seq_failed, mutations)

    if not plausible:
        return {
            "label": LABEL_UNRESOLVED,
            "mutations": mutations,
            "reason": reason,
        }

    return {
        "label": LABEL_CONFIRMED_FAILURE,
        "mutations": mutations,
        "reason": reason,
    }
