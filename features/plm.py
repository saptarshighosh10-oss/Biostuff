"""
Protein-language-model embeddings (ESM-2 + AbLang2) with a disk cache.

Frozen, mean-pooled embeddings only — no fine-tuning. Heavy libraries
(torch/transformers/ablang2) are imported LAZILY inside the runner functions,
so this module imports and the cache layer is usable with only the standard
library present. Cache lives under data/cache/embeddings/<model_tag>/<sha256>.json.

Design decisions (see docs/superpowers/specs/2026-07-14-plm-scaleup-design.md §3.1, §6):
- General PLM: ESM-2 `facebook/esm2_t33_650M_UR50D` via HuggingFace `transformers`
  (clean AutoTokenizer/AutoModel path, maintained; `fair-esm` is an equivalent
  alternative — transformers chosen for the simpler frozen-embedding API).
- Antibody PLM: `ablang2`.
- Combined feature backbone for Head B = [ESM2_mean | AbLang2_mean].
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_TAG_ESM2 = "esm2_t33_650M"
MODEL_TAG_ABLANG2 = "ablang2"

ESM2_MODEL_ID = "facebook/esm2_t33_650M_UR50D"

# Repo-root-anchored cache dir. Tests override this module global with a tempdir.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = _REPO_ROOT / "data" / "cache" / "embeddings"


# --- disk cache (stdlib only) ------------------------------------------------

def _seq_key(sequence: str) -> str:
    """Deterministic cache key: sha256 of the sequence bytes."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def cache_path(model_tag: str, sequence: str) -> Path:
    """Path where the embedding for (model_tag, sequence) is/would be cached."""
    return CACHE_ROOT / model_tag / f"{_seq_key(sequence)}.json"


def load_cached(model_tag: str, sequence: str) -> list[float] | None:
    """Return the cached embedding, or None on cache miss."""
    p = cache_path(model_tag, sequence)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_cached(model_tag: str, sequence: str, vec: list[float]) -> None:
    """Write an embedding to the cache as a JSON list of floats."""
    p = cache_path(model_tag, sequence)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([float(x) for x in vec]))


# --- lazy heavy runners (import torch/transformers/ablang2 only on miss) ------

def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch  # imported lazily by caller's try-block context
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_esm2(sequences: list[str], device: str = "auto", batch_size: int = 8) -> list[list[float]]:
    """Mean-pooled ESM-2 embeddings. Lazily imports torch + transformers."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            f"ESM-2 embedding requires 'torch' and 'transformers' (missing: "
            f"{getattr(e, 'name', e)}). Install: pip install torch transformers"
        ) from e

    dev = _pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)
    model = AutoModel.from_pretrained(ESM2_MODEL_ID).to(dev).eval()

    out: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            toks = tokenizer(batch, return_tensors="pt", padding=True).to(dev)
            hidden = model(**toks).last_hidden_state  # (B, L, 1280)
            # Mean-pool over real tokens only (exclude padding via attention mask).
            mask = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.extend(pooled.cpu().tolist())
    return out


def _run_ablang2(sequences: list[str], device: str = "auto") -> list[list[float]]:
    """Mean-pooled AbLang2 embeddings. Lazily imports ablang2."""
    try:
        import ablang2
    except ImportError as e:
        raise RuntimeError(
            f"AbLang2 embedding requires 'ablang2' (missing: "
            f"{getattr(e, 'name', e)}). Install: pip install ablang2"
        ) from e

    # ponytail: AbLang2 expects paired [heavy, light] input; single chains go in
    # as [seq, '']. Exact call surface may need tuning on real hardware (the
    # calibration knob) — verify against the installed ablang2 version.
    model = ablang2.pretrained(device=_pick_device(device))
    model.freeze()
    pairs = [[s, ""] for s in sequences]
    vecs = model(pairs, mode="seqcoding")
    return [list(map(float, v)) for v in vecs]


# --- cache-first embedding API -----------------------------------------------

def _embed_with_cache(sequences, model_tag, runner, device):
    results: list[list[float] | None] = [None] * len(sequences)
    misses: list[str] = []
    miss_idx: list[int] = []
    for i, s in enumerate(sequences):
        cached = load_cached(model_tag, s)
        if cached is None:
            misses.append(s)
            miss_idx.append(i)
        else:
            results[i] = cached

    if misses:
        uniq = list(dict.fromkeys(misses))  # compute each distinct sequence once
        vmap = dict(zip(uniq, runner(uniq, device)))
        for s, v in vmap.items():
            save_cached(model_tag, s, v)
        for i, s in zip(miss_idx, misses):
            results[i] = vmap[s]
    return results  # type: ignore[return-value]


def embed_esm2(sequences: list[str], device: str = "auto") -> list[list[float]]:
    """Cache-first ESM-2 embeddings; runs the model only for cache misses."""
    return _embed_with_cache(sequences, MODEL_TAG_ESM2, _run_esm2, device)


def embed_ablang2(sequences: list[str], device: str = "auto") -> list[list[float]]:
    """Cache-first AbLang2 embeddings; runs the model only for cache misses."""
    return _embed_with_cache(sequences, MODEL_TAG_ABLANG2, _run_ablang2, device)


def embed_combined(sequence: str) -> list[float]:
    """
    Head-B feature backbone: [ESM2_mean | AbLang2_mean] read from cache.

    Both embeddings must already be cached (run --precompute first). Raises
    RuntimeError naming the missing model rather than silently loading weights.
    """
    esm = load_cached(MODEL_TAG_ESM2, sequence)
    ab = load_cached(MODEL_TAG_ABLANG2, sequence)
    missing = [tag for tag, v in ((MODEL_TAG_ESM2, esm), (MODEL_TAG_ABLANG2, ab)) if v is None]
    if missing:
        raise RuntimeError(
            f"embed_combined: no cached embedding for {missing}. "
            f"Run: python -m features.plm --precompute <sequences.txt> --model both"
        )
    return esm + ab


# --- CLI: batch-fill the cache -----------------------------------------------

def _precompute(seq_file: str, model: str) -> None:
    lines = Path(seq_file).read_text().splitlines()
    seqs = list(dict.fromkeys(s.strip() for s in lines if s.strip()))
    if model in ("esm2", "both"):
        embed_esm2(seqs)
    if model in ("ablang2", "both"):
        embed_ablang2(seqs)
    print(f"cached {len(seqs)} unique sequence(s) for model={model} under {CACHE_ROOT}")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="features.plm", description=__doc__)
    parser.add_argument("--precompute", metavar="SEQUENCES.TXT", required=True,
                        help="file with one sequence per line")
    parser.add_argument("--model", choices=("esm2", "ablang2", "both"), default="both")
    args = parser.parse_args(argv)
    _precompute(args.precompute, args.model)


if __name__ == "__main__":
    main()
