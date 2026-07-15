"""Generate small, auditable mutation panels for active learning.

This does not claim a mutation is safe or effective. It creates hypotheses for
the selector and keeps the parent and exact edit so every suggestion is
reviewable before any wet-lab work.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Conservative substitutions keep the first panel interpretable. Expand only
# after checking that the assay cohort supports the new mutation classes.
_GROUPS = ("AVILM", "ST", "DE", "KR", "FYW", "NQ")
_SUBSTITUTIONS = {
    residue: tuple(other for group in _GROUPS if residue in group for other in group if other != residue)
    for residue in "".join(_GROUPS)
}


def generate_candidates(
    parents: list[dict],
    *,
    max_total: int = 300,
    seed: int = 7,
) -> list[dict]:
    """Return deterministic, unique single-substitution candidates."""
    known = {
        str(row.get("variant_sequence") or "").strip().upper()
        for row in parents
    }
    eligible = [sequence for sequence in known if sequence]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    candidates: list[dict] = []
    seen = set(known)

    for parent in eligible:
        positions = list(range(1, len(parent) - 1))
        rng.shuffle(positions)
        for position in positions:
            residue = parent[position]
            replacements = list(_SUBSTITUTIONS.get(residue, ()))
            rng.shuffle(replacements)
            for replacement in replacements:
                sequence = parent[:position] + replacement + parent[position + 1:]
                if sequence in seen:
                    continue
                seen.add(sequence)
                candidates.append({
                    "variant_sequence": sequence,
                    "parent_sequence": parent,
                    "mutation": f"{residue}{position + 1}{replacement}",
                    "mutation_position_1based": position + 1,
                    "mutation_class": "conservative_single_substitution",
                    "candidate_source": "controlled_mutation_generator",
                })
                break
            if len(candidates) >= max_total:
                return candidates
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an auditable mutation panel")
    parser.add_argument("--pool", required=True, help="JSON list of existing sequence rows")
    parser.add_argument("--out", default="results/novel_candidates.json")
    parser.add_argument("--max-total", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    parents = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    candidates = generate_candidates(parents, max_total=args.max_total, seed=args.seed)
    Path(args.out).write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(candidates)} novel candidates to {args.out}")


if __name__ == "__main__":
    main()
