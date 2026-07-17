"""Strict, dependency-free reconstruction of substitution-only antibody variants."""

from __future__ import annotations

import re
from typing import Mapping

from data.contract import ContractError, canonical_sequence, sha256_hex


_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_MUTATION = re.compile(rf"([{_AMINO_ACIDS}])([1-9][0-9]*)([{_AMINO_ACIDS}])")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def parse_substitutions(value: object) -> tuple[tuple[int, str, str], ...]:
    """Parse one-based substitutions such as ``A23G;S45T``."""
    text = _text(value).upper()
    if not text:
        return ()
    tokens = [token for token in re.split(r"[;,\s]+", text) if token]
    mutations: list[tuple[int, str, str]] = []
    positions: set[int] = set()
    for token in tokens:
        match = _MUTATION.fullmatch(token)
        if match is None:
            raise ValueError(f"unsupported substitution: {token}")
        wild_type, raw_position, replacement = match.groups()
        position = int(raw_position)
        if position in positions:
            raise ValueError(f"duplicate mutation position: {position}")
        if wild_type == replacement:
            raise ValueError(f"substitution does not change residue: {token}")
        positions.add(position)
        mutations.append((position, wild_type, replacement))
    return tuple(sorted(mutations))


def _sequence_diff(parent: str, child: str) -> tuple[tuple[int, str, str], ...]:
    if len(parent) != len(child):
        raise ValueError("substitution reconstruction changed sequence length")
    return tuple(
        (position, before, after)
        for position, (before, after) in enumerate(zip(parent, child), 1)
        if before != after
    )


def apply_substitutions(parent: object, mutations: object) -> tuple[str, dict]:
    """Apply substitutions and independently round-trip the resulting diff."""
    try:
        parent_sequence = canonical_sequence(parent)
    except ContractError as exc:
        raise ValueError("parent is not a standard amino-acid sequence") from exc
    parsed = parse_substitutions(mutations)
    child = list(parent_sequence)
    for position, wild_type, replacement in parsed:
        if position > len(parent_sequence) or parent_sequence[position - 1] != wild_type:
            raise ValueError(
                f"mutation {wild_type}{position}{replacement} does not match parent"
            )
        child[position - 1] = replacement
    reconstructed = "".join(child)
    if _sequence_diff(parent_sequence, reconstructed) != parsed:
        raise ValueError("round-trip mutation check failed")
    return reconstructed, {
        "mutations": [f"{wild_type}{position}{replacement}"
                      for position, wild_type, replacement in parsed],
        "parent_residue_check": True,
        "round_trip_diff_check": True,
    }


def _verify_expected(
    chain: str,
    reconstructed: str,
    expected_sequence: object,
    expected_sha256: object,
) -> dict:
    raw_sequence = _text(expected_sequence)
    raw_digest = _text(expected_sha256).lower()
    if not raw_sequence and not raw_digest:
        raise ValueError(f"independent expected {chain} sequence or SHA-256 is required")

    methods: list[str] = []
    if raw_sequence:
        try:
            expected = canonical_sequence(raw_sequence)
        except ContractError as exc:
            raise ValueError(f"independent {chain} sequence is invalid") from exc
        if expected != reconstructed:
            raise ValueError(f"independent {chain} sequence does not match reconstruction")
        methods.append("full_sequence")
    if raw_digest:
        if _SHA256.fullmatch(raw_digest) is None:
            raise ValueError(f"independent {chain} SHA-256 is invalid")
        if raw_digest != sha256_hex(reconstructed):
            raise ValueError(f"independent {chain} SHA-256 does not match reconstruction")
        methods.append("sha256")
    return {
        "independent_sequence_check": "+".join(methods),
        "child_sha256": sha256_hex(reconstructed),
    }


def reconstruct_verified_pair(
    row: Mapping[str, object],
) -> tuple[str, str | None, dict]:
    """Reconstruct VH/VL only when mutation and independent evidence agree."""
    mutation_reference = _text(row.get("parent_mutation_reference"))
    verification_reference = _text(row.get("verification_reference"))
    if not mutation_reference or not verification_reference or mutation_reference == verification_reference:
        raise ValueError("two distinct provenance references are required")

    vh, vh_audit = apply_substitutions(row.get("parent_vh"), row.get("vh_mutations"))
    vh_audit.update(_verify_expected(
        "VH", vh, row.get("expected_vh"), row.get("expected_vh_sha256")
    ))

    parent_vl = _text(row.get("parent_vl"))
    if parent_vl:
        vl, vl_audit = apply_substitutions(parent_vl, row.get("vl_mutations"))
        vl_audit.update(_verify_expected(
            "VL", vl, row.get("expected_vl"), row.get("expected_vl_sha256")
        ))
    else:
        if any(_text(row.get(field)) for field in
               ("vl_mutations", "expected_vl", "expected_vl_sha256")):
            raise ValueError("VL reconstruction evidence was supplied without parent_vl")
        vl = None
        vl_audit = None

    if not parse_substitutions(row.get("vh_mutations")) and not parse_substitutions(row.get("vl_mutations")):
        raise ValueError("reconstruction requires at least one explicit substitution")

    chains = {"vh": vh_audit}
    if vl_audit is not None:
        chains["vl"] = vl_audit
    return vh, vl, {
        "parent_mutation_reference": mutation_reference,
        "verification_reference": verification_reference,
        "chains": chains,
    }
