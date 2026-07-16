"""Propose single-mutation antibody variants ranked by the saved risk predictor.

This is an inverse-design guide, not validation: it proposes variants whose
*predicted* aggregation, polyreactivity, and hydrophobicity risks are lower.
Only an experimental assay can establish that a proposed variant works.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from statistics import mean

from model.head_b_acsins_charge import charge_features
from model.head_b_gbm import build_feature_vector

AA = "ACDEFGHIKLMNPQRSTVWY"
PROPERTIES = ("aggregation", "polyreactivity", "hydrophobicity")
CSV_COLUMNS = [
    "mutation", "chain", "position", "wildtype_aa", "new_aa",
    "d_aggregation", "d_polyreactivity", "d_hydrophobicity", "combined_delta",
    "new_vh", "new_vl",
]

_EXPECTED_SELF_TEST_PLATFORM_WIDTH = 7


def _sequence(value: str, name: str, allow_empty: bool = False) -> str:
    seq = (value or "").strip().upper()
    if (not seq and not allow_empty) or any(aa not in AA for aa in seq):
        raise ValueError(f"{name} must contain only the 20 standard amino acids")
    return seq


def _row(vh: str, vl: str) -> dict:
    return {"variant_sequence": vh, "vh_sequence": vh, "vl_sequence": vl}


def _platform_feature_width(platform: dict) -> int:
    """Read the trained artifact and return the shared model feature width."""
    heads = platform.get("heads", {})
    missing = [prop for prop in PROPERTIES if prop not in heads]
    if missing:
        raise ValueError(f"platform is missing served properties: {', '.join(missing)}")

    widths: set[int] = set()
    for prop in PROPERTIES:
        assays = heads[prop].get("assays", {})
        if not assays:
            raise ValueError(f"platform has no trained models for '{prop}'")
        for assay_name, assay in assays.items():
            model = assay.get("model")
            if model is None:
                raise ValueError(f"platform assay '{prop}/{assay_name}' is missing a model")
            width = getattr(model, "n_features_in_", None)
            if not isinstance(width, int):
                raise ValueError(
                    f"platform assay '{prop}/{assay_name}' does not expose "
                    "n_features_in_, can't verify scored feature width"
                )
            widths.add(width)

    if len(widths) != 1:
        raise ValueError(
            f"platform feature width is inconsistent across assays: {sorted(widths)}"
        )
    return next(iter(widths))


def _cache_vh_embeddings(vhs: list[str]) -> None:
    """Populate PLM cache so missing embeddings cannot silently become zeros."""
    from features.plm import embed_ablang2, embed_esm2
    from features import plm

    unique = list(dict.fromkeys(vhs))
    if not unique:
        return
    print(f"Caching frozen PLM features for {len(unique)} VH sequence(s)...")
    embed_esm2(unique)
    embed_ablang2(unique)
    for seq in unique:
        for tag in (plm.MODEL_TAG_ESM2, plm.MODEL_TAG_ABLANG2):
            cached = plm.load_cached(tag, seq)
            if not cached:
                raise RuntimeError(
                    f"PLM cache for {tag} is missing after precompute for sequence '{seq[:12]}...'"
                )
            if not isinstance(cached, list):
                raise RuntimeError(
                    f"PLM cache for {tag} and sequence '{seq[:12]}...' is malformed"
                )


def _platform_needs_plm(expected_width: int) -> bool:
    """A biophysical+charge platform is ~35 features wide; PLM adds >1000."""
    return expected_width > 200


def _feature(row: dict, expected_width: int) -> list[float]:
    charge = charge_features(row)
    if not _platform_needs_plm(expected_width):
        # biophysical+charge platform — instant, no PLM embeddings needed
        vector = build_feature_vector(row, use_plm=False) + charge
    else:
        vector = build_feature_vector(row, use_plm=True) + charge
    if len(vector) != expected_width:
        raise RuntimeError(
            f"Feature width is {len(vector)}, but platform.pkl expects {expected_width}. "
            "PLM caching or the saved platform is incompatible; do not trust these scores."
        )
    return vector


def _self_test() -> None:
    class _StubModel:
        n_features_in_ = _EXPECTED_SELF_TEST_PLATFORM_WIDTH

        def predict(self, X):
            # return a deterministic value for smoke-test scoring math
            return [sum(len(row) for row in X) * 0.0 + 0.0]

    platform = {
        "heads": {
            prop: {
                "assays": {
                    "a": {
                        "model": _StubModel(),
                        "direction": "higher_bad",
                    }
                }
            }
            for prop in PROPERTIES
        }
    }

    expected_width = _platform_feature_width(platform)
    assert expected_width == _EXPECTED_SELF_TEST_PLATFORM_WIDTH

    assert _sequence("acdef", "VH") == "ACDEF"
    bad = False
    try:
        _sequence("ACDX", "VH")
    except ValueError:
        bad = True
    assert bad

    cands = candidates("ACD", "", "vh", max_candidates=3)
    assert len(cands) == 3
    assert cands[0]["chain"] == "vh"
    assert all(c["mutation"].startswith("VH:") for c in cands)
    print("design_loop self-test: ok")


def load_platform(path: str) -> dict:
    with open(path, "rb") as fh:
        platform = pickle.load(fh)
    heads = platform.get("heads", {})
    missing = [prop for prop in PROPERTIES if not heads.get(prop, {}).get("assays")]
    if missing:
        raise ValueError(f"platform is missing served property heads: {', '.join(missing)}")
    return platform


def score(row: dict, platform: dict, feature_width: int) -> dict[str, float]:
    """Return one oriented score per property; higher always means worse."""
    vector = _feature(row, feature_width)
    scores = {}
    for prop in PROPERTIES:
        oriented = []
        for assay in platform["heads"][prop]["assays"].values():
            prediction = float(assay["model"].predict([vector])[0])
            oriented.append(prediction if assay["direction"] == "higher_bad" else -prediction)
        scores[prop] = mean(oriented)
    return scores


def candidates(vh: str, vl: str, chain: str, max_candidates: int) -> list[dict]:
    """Deterministic all-amino-acid single substitutions, capped for local runs."""
    out = []
    for label, sequence in (("vh", vh), ("vl", vl)):
        if chain not in (label, "both"):
            continue
        for index, wildtype in enumerate(sequence, start=1):
            for new_aa in AA:
                if new_aa == wildtype:
                    continue
                mutated = sequence[:index - 1] + new_aa + sequence[index:]
                out.append({
                    "mutation": f"{label.upper()}:{wildtype}{index}{new_aa}", "chain": label,
                    "position": index, "wildtype_aa": wildtype, "new_aa": new_aa,
                    "new_vh": mutated if label == "vh" else vh,
                    "new_vl": mutated if label == "vl" else vl,
                })
                if len(out) >= max_candidates:
                    return out
    return out


def design(vh: str, vl: str, platform_path: str, chain: str, max_candidates: int) -> tuple[dict, list[dict]]:
    vh, vl = _sequence(vh, "VH"), _sequence(vl, "VL", allow_empty=True)
    if chain in ("vl", "both") and not vl:
        raise ValueError("VL optimization needs a non-empty --vl sequence")
    platform = load_platform(platform_path)
    expected_width = _platform_feature_width(platform)
    proposed = candidates(vh, vl, chain, max_candidates)
    if not proposed:
        raise ValueError("No substitutions were generated")
    if _platform_needs_plm(expected_width):
        _cache_vh_embeddings([vh] + [candidate["new_vh"] for candidate in proposed])

    baseline = score(_row(vh, vl), platform, expected_width)
    for candidate in proposed:
        predicted = score(_row(candidate["new_vh"], candidate["new_vl"]), platform, expected_width)
        for prop in PROPERTIES:
            candidate[f"d_{prop}"] = predicted[prop] - baseline[prop]
        candidate["combined_delta"] = mean(candidate[f"d_{prop}"] for prop in PROPERTIES)
    # exact ranking requirement: ascending mean delta across three served properties.
    return baseline, sorted(proposed, key=lambda item: item["combined_delta"])


def write_csv(rows: list[dict], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in CSV_COLUMNS} for row in rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vh", required=True, help="heavy-chain amino-acid sequence")
    parser.add_argument("--vl", default="", help="light-chain amino-acid sequence")
    parser.add_argument("--chain", choices=("vh", "vl", "both"), default="vh")
    parser.add_argument("--max-candidates", type=int, default=2500)
    parser.add_argument("--self-test", action="store_true", help="run a tiny built-in assertion check")
    parser.add_argument("--platform", default="model/saved/platform.pkl")
    parser.add_argument("--out", default="results/designs.csv")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return

    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")

    baseline, rows = design(args.vh, args.vl, args.platform, args.chain, args.max_candidates)
    write_csv(rows, args.out)
    print("Baseline oriented risk (higher = worse): " + ", ".join(
        f"{prop}={baseline[prop]:.4f}" for prop in PROPERTIES))
    print(f"\nTop 15 of {len(rows)} single-substitution proposals (negative delta = improvement):")
    print(f"  {'mutation':<13}{'aggregation':>13}{'polyreactivity':>18}{'hydrophobicity':>18}{'combined':>12}")
    for row in rows[:15]:
        print(f"  {row['mutation']:<13}{row['d_aggregation']:>+13.4f}{row['d_polyreactivity']:>+18.4f}"
              f"{row['d_hydrophobicity']:>+18.4f}{row['combined_delta']:>+12.4f}")
    print(f"\nwrote {args.out}")
    print("Predicted improvements — must be confirmed by assay before trusting.")


if __name__ == "__main__":
    main()
