"""
3D structure viewer for candidates with ESMFold structures.

Renders the antibody chain colored by ESMFold confidence (pLDDT), with
mutation positions and Phase 2 consensus aggregation hotspots highlighted.
Uses py3Dmol (3Dmol.js) — renders inline in Colab/Jupyter, or exports a
standalone HTML file. The exported file loads the 3Dmol.js library from a
CDN at view-time, so it needs internet once to open (same as any normal
webpage) — everything else about it is self-contained.

Requires a candidate that went through `--esmfold` in Phase 1, since that's
the only step that generates a structure.

Usage (Colab):
    from model.structure_viewer import view_candidate
    view_candidate(candidates[0])   # displays inline in the notebook

Usage (CLI, exports HTML):
    python -m model.structure_viewer --file results/phase2_candidates.json --index 0
"""

import argparse
import json
from pathlib import Path


def _mutation_resi(mutations: list) -> list[int]:
    """Convert 0-indexed (pos, orig, mut) tuples to 1-indexed PDB residue numbers."""
    return [int(m[0]) + 1 for m in mutations if len(m) >= 1]


def _hotspot_resi_ranges(hotspots: list) -> list[tuple[int, int]]:
    """Convert 0-indexed [start, end] hotspot ranges to 1-indexed PDB residue ranges."""
    ranges = []
    for h in hotspots or []:
        if len(h) == 2:
            ranges.append((int(h[0]) + 1, int(h[1]) + 1))
    return ranges


def build_view(
    pdb_string: str,
    mutations: list | None = None,
    hotspots: list | None = None,
    width: int = 700,
    height: int = 500,
):
    """
    Build a py3Dmol view: cartoon colored by ESMFold pLDDT confidence
    (blue = confident, red = unconfident — the AlphaFold/ESMFold convention),
    mutation positions as magenta spheres, consensus hotspots as orange sticks.
    """
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_string, "pdb")

    # base: cartoon colored by per-residue pLDDT (ESMFold stores it in the B-factor column)
    view.setStyle({}, {"cartoon": {"colorscheme": {"prop": "b", "gradient": "rwb", "min": 50, "max": 90}}})

    mut_resi = _mutation_resi(mutations or [])
    if mut_resi:
        view.addStyle({"resi": mut_resi}, {"sphere": {"color": "magenta", "radius": 1.4}})

    for start, end in _hotspot_resi_ranges(hotspots):
        view.addStyle({"resi": list(range(start, end + 1))}, {"stick": {"color": "orange", "radius": 0.4}})

    view.zoomTo()
    return view


def view_candidate(candidate: dict, width: int = 700, height: int = 500):
    """
    Colab/Jupyter helper: build and immediately .show() the view for one
    candidate dict. Needs 'pdb_string' in candidate['risk'] — only present
    for candidates scored with --esmfold in Phase 1.
    """
    pdb_string = candidate.get("risk", {}).get("pdb_string")
    if not pdb_string:
        print("No ESMFold structure available for this candidate — "
              "re-run Phase 1 with --esmfold, or increase --top so this "
              "candidate falls within the ESMFold-scored range.")
        return None

    mutations = candidate.get("mutations", [])
    hotspots = candidate.get("multi_predictor", {}).get("consensus_hotspots", [])
    view = build_view(pdb_string, mutations, hotspots, width, height)

    print(f"{candidate.get('anchor_pdb', '?')} ({candidate.get('chain_type', '?')}) | mutations={mutations}")
    print("Color: blue = high ESMFold confidence, red = low confidence")
    print("Magenta spheres = mutated positions. Orange = Phase 2 consensus aggregation hotspot.")
    view.show()
    return view


def save_structure_html(candidate: dict, output_path: str, width: int = 700, height: int = 500) -> str | None:
    """Render one candidate's structure to a standalone HTML file."""
    pdb_string = candidate.get("risk", {}).get("pdb_string")
    if not pdb_string:
        print(f"  {candidate.get('anchor_pdb', '?')}: no ESMFold structure available — skipped")
        return None

    mutations = candidate.get("mutations", [])
    hotspots = candidate.get("multi_predictor", {}).get("consensus_hotspots", [])
    view = build_view(pdb_string, mutations, hotspots, width, height)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    view.write_html(output_path, fullpage=False)
    print(f"  Saved 3D structure to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a candidate's ESMFold structure in 3D")
    parser.add_argument("--file", required=True, help="Candidates JSON (needs --esmfold to have run in Phase 1)")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output HTML path (default: results/viz/structure_<index>.html)")
    args = parser.parse_args()

    with open(args.file) as f:
        candidates = json.load(f)
    if args.index >= len(candidates):
        raise SystemExit(f"--index {args.index} out of range (file has {len(candidates)} candidates)")

    output = args.output or f"results/viz/structure_{args.index}.html"
    save_structure_html(candidates[args.index], output)
