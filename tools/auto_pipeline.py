"""Gate newly appended polyreactivity sources by diverse-source generalization.

Run manually with ``python -m tools.auto_pipeline``.  It records decisions in
``tools/pipeline_state.json`` and never changes the source dataset.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from model.head_c_polyreactivity import _load
from model.head_c_transfer import DIVERSE_SOURCES, LIBRARY_SOURCES, evaluate_transfer


STATE_PATH = Path(__file__).with_name("pipeline_state.json")
TOLERANCE = 0.005


def load_state() -> dict:
    """Load the small append-only decision state with its required fields."""
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    for field, expected in (("accepted_sources", list), ("quarantined", dict), ("history", list)):
        if not isinstance(state.get(field), expected):
            raise ValueError(f"{STATE_PATH.name}: {field} must be a {expected.__name__}")
    return state


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def library_sources(accepted_sources: list[str]) -> list[str]:
    """Return accepted training sources, explicitly excluding diverse holdouts."""
    return [source for source in accepted_sources if source in LIBRARY_SOURCES]


def mean_diverse_auc(train_sources: list[str], by_source: dict[str, list[dict]]) -> float:
    report = evaluate_transfer(train_sources, DIVERSE_SOURCES, by_source)
    if report["mean_auc"] is None:
        raise ValueError("diverse holdout evaluation produced no scorable AUC")
    return report["mean_auc"]


def main() -> None:
    by_source = _load()
    state = load_state()
    known = set(state["accepted_sources"]) | set(state["quarantined"])
    candidates = sorted(set(by_source) - known)
    print(f"known accepted sources: {', '.join(state['accepted_sources'])}")
    if not candidates:
        print("No new sources to gate.")
        return

    for candidate in candidates:
        if candidate in DIVERSE_SOURCES:
            raise ValueError(f"{candidate} is a diverse holdout and cannot be a training candidate")
        baseline_sources = library_sources(state["accepted_sources"])
        baseline = mean_diverse_auc(baseline_sources, by_source)
        trial = mean_diverse_auc(baseline_sources + [candidate], by_source)
        accepted = trial >= baseline - TOLERANCE
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "source": candidate,
            "baseline_diverse_auc": baseline,
            "trial_diverse_auc": trial,
            "decision": "accepted" if accepted else "quarantined",
        }
        if accepted:
            state["accepted_sources"].append(candidate)
            reason = "held or improved diverse generalization"
        else:
            reason = f"did not help diverse generalization (baseline {baseline:.3f}, trial {trial:.3f})"
            state["quarantined"][candidate] = reason
        record["reason"] = reason
        state["history"].append(record)
        save_state(state)
        print(
            f"{candidate}: {record['decision'].upper()} — baseline {baseline:.3f}, "
            f"trial {trial:.3f}; {reason}"
        )

    print("\nFinal accepted sources: " + ", ".join(state["accepted_sources"]))
    if state["quarantined"]:
        print("Quarantined sources:")
        for source, reason in sorted(state["quarantined"].items()):
            print(f"  {source}: {reason}")


if __name__ == "__main__":
    main()
