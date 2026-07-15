"""Data contract and hashing helpers for Plan C provenance.

These helpers are intentionally lightweight and stdlib-only. They define the
normalized row schema, canonical JSON serialization, and deterministic hashing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
HASH_EXCLUDE_KEYS = {
    "timestamp",
    "created_at",
    "updated_at",
    "fetched_at",
    "fetch_timestamp_utc",
    "ingested_at",
}
SOURCE_ROW_HASH_EXCLUDE_KEYS = HASH_EXCLUDE_KEYS | {"source_row_hash"}

SOURCE_ROLES: dict[str, str] = {
    "flab": "supervised",
    "gdpa": "supervised",  # Ginkgo antibody developability benchmark (GDPa1/2); GDPa3 frozen holdout
    "proteingym": "auxiliary",
    "canya": "auxiliary",
    "figshare_a3d": "auxiliary",
    "figshare_agg": "auxiliary",
    "benchmark": "unlabeled",
    "abdev": "background_ood",
    "antiref": "background_ood",
    "sabdab": "background_ood",
    "oas": "background_ood",           # Observed Antibody Space — natural repertoires
    "thera_sabdab": "background_ood",  # Thera-SAbDab — clinical-stage therapeutic mAbs
    "anchors": "background_ood",
    "pdb_anchor": "background_ood",
    "unknown": "unlabeled",
    "legacy": "unlabeled",
    "input": "unlabeled",
}
_SOURCE_ALIASES = {
    "protein_gym": "proteingym",
    "protein-gym": "proteingym",
    "proteingym_dms": "proteingym",
    "figshareagg": "figshare_a3d",
    "figshare-a3d": "figshare_a3d",
    "figshare_a3": "figshare_a3d",
    "pdb-anchor": "pdb_anchor",
    "ab-dev": "abdev",
    "anti-ref": "antiref",
}

ENDPOINT_DIRECTIONS = {"higher_bad", "lower_bad"}


class ContractError(ValueError):
    """Raised when a row cannot satisfy the Plan C contract."""


def canonical_json_dumps(payload: Any, exclude_keys: set[str] | None = None) -> str:
    """
    Serialize payload with deterministic formatting and key order.
    """
    excluded = set(HASH_EXCLUDE_KEYS)
    if exclude_keys:
        excluded |= set(exclude_keys)
    cleaned = _strip_keys(payload, excluded)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(payload: Any, exclude_keys: set[str] | None = None) -> str:
    """
    Compute sha256 over canonical JSON bytes.
    """
    excluded = set(HASH_EXCLUDE_KEYS)
    if exclude_keys:
        excluded |= set(exclude_keys)
    text = canonical_json_dumps(payload, exclude_keys=excluded)
    return sha256_hex(text.encode("utf-8"))


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _strip_keys(payload: Any, exclude_keys: set[str]) -> Any:
    if isinstance(payload, dict):
        return {
            k: _strip_keys(v, exclude_keys)
            for k, v in payload.items()
            if k not in exclude_keys
        }
    if isinstance(payload, list):
        return [_strip_keys(v, exclude_keys) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_strip_keys(v, exclude_keys) for v in payload)
    return payload


def canonical_sequence(value: Any) -> str:
    """
    Normalize a sequence field to uppercase canonical amino-acid characters.
    """
    if not isinstance(value, str):
        raise ContractError("sequence must be a string")
    seq = "".join(ch for ch in value.strip().upper() if ch and ch != " ")
    if not seq:
        raise ContractError("sequence is empty")
    if not AA_RE.fullmatch(seq):
        raise ContractError(f"invalid amino-acid sequence: {value!r}")
    return seq


def source_supervision_role(source: str) -> str:
    canonical = _canonical_source_name(source)
    if canonical not in SOURCE_ROLES:
        raise ContractError(f"unknown source in supervision mapping: {source!r}")
    return SOURCE_ROLES[canonical]


def canonical_source_name(source: str | None) -> str:
    """
    Canonical source string that maps aliases and preserves explicit unknown.
    """
    return _canonical_source_name(source)


def _canonical_source_name(source: Any) -> str:
    if not isinstance(source, str):
        raise ContractError("source must be a non-empty string")
    source = source.strip().lower()
    if not source:
        raise ContractError("source must be a non-empty string")
    return _SOURCE_ALIASES.get(source, source)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_as_float):
        raise ContractError(f"non-finite numeric value is not allowed: {value!r}")
    return value_as_float


def _coalesce(value: Any, *rest: Any) -> Any:
    if value is not None:
        return value
    for item in rest:
        if item is not None:
            return item
    return None


def _coerce_binary_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value not in (0.0, 1.0):
            return None
        iv = int(value)
        if iv in (0, 1):
            return iv
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "confirmed_failure", "failure", "bad"}:
        return 1
    if text in {"0", "false", "no", "working", "good"}:
        return 0
    return None


def _infer_endpoint_direction(source: str, assay_metric: str | None, score_col: str | None) -> str:
    metric = (assay_metric or score_col or "").lower()
    if source == "flab":
        if any(token in metric for token in ("sec", "monomer", "solubil", "percent")):
            return "lower_bad"
    return "higher_bad"


def _resolve_endpoint_direction(
    row: dict[str, Any],
    *,
    source: str,
    source_role: str,
    assay_metric: str | None,
    score_col: str | None,
) -> str:
    explicit = row.get("endpoint_direction")
    if source_role == "supervised":
        if not isinstance(explicit, str):
            raise ContractError("supervised source requires explicit endpoint_direction")
        normalized = explicit.strip().lower()
        if not normalized:
            raise ContractError("supervised source requires explicit endpoint_direction")
        if normalized not in ENDPOINT_DIRECTIONS:
            raise ContractError(f"invalid endpoint_direction: {explicit!r}")
        return normalized

    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if not normalized:
            return _infer_endpoint_direction(
                source=source,
                assay_metric=assay_metric,
                score_col=str(score_col),
            )
        if normalized not in ENDPOINT_DIRECTIONS:
            raise ContractError(f"invalid endpoint_direction: {explicit!r}")
        return normalized
    if explicit is not None:
        raise ContractError(f"invalid endpoint_direction: {explicit!r}")

    # Non-supervised compatibility path: keep legacy inference for older metadata
    return _infer_endpoint_direction(
        source=source,
        assay_metric=assay_metric,
        score_col=str(score_col),
    )


def _candidate_sequence(row: dict[str, Any], key: str) -> str | None:
    if key in row and row[key]:
        return str(row[key]).strip() or None
    return None


def _normalize_chain_id(row: dict[str, Any]) -> str | None:
    if "chain_id" not in row or row.get("chain_id") is None:
        return None

    chain_id = str(row["chain_id"]).strip().lower()
    if not chain_id:
        return None
    if chain_id not in {"vh", "vl"}:
        raise ContractError(f"invalid chain_id: {row['chain_id']!r}")
    return chain_id


@dataclass(frozen=True)
class NormalizedRow:
    """Canonical normalized row used by data contracts."""

    record_id: str
    source: str
    study_id: str
    campaign_id: str
    molecule_id: str
    pair_id: str | None
    chain_id: str | None
    sequence: str
    vh_sequence: str | None
    vl_sequence: str | None
    assay_id: str
    assay_metric: str
    endpoint_direction: str
    endpoint_value: float | None
    endpoint_unit: str | None
    target_value: float | None
    supervision_status: str
    binary_label: int | None
    label_threshold: float | None
    group_study: str
    group_molecule: str
    group_campaign: str
    sequence_hash: str
    exclusions: list[str] = field(default_factory=list)
    source_url: str = ""
    source_row_hash: str = ""
    feature_flags: list[str] = field(default_factory=list)
    source_version: str = ""
    assay_conditions: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exclusions"] = list(self.exclusions)
        payload["feature_flags"] = list(self.feature_flags)
        return payload


def normalize_legacy_row(
    row: dict[str, Any],
    *,
    source_url: str | None = None,
) -> NormalizedRow:
    """
    Convert a legacy loader row into a normalized contract row.
    """
    if not isinstance(row, dict):
        raise ContractError("legacy row must be a dict")

    source = _canonical_source_name(row.get("source"))
    source_role = source_supervision_role(source)

    raw_seq = row.get("variant_sequence") or row.get("sequence")
    if raw_seq is None:
        raise ContractError("missing sequence field")
    sequence = canonical_sequence(raw_seq)
    sequence_hash = sha256_hex(sequence)

    vh_sequence = _candidate_sequence(row, "vh_sequence")
    vl_sequence = _candidate_sequence(row, "vl_sequence")
    if vh_sequence is not None:
        vh_sequence = canonical_sequence(vh_sequence)
    if vl_sequence is not None:
        vl_sequence = canonical_sequence(vl_sequence)

    study_id = (
        str(row.get("group_id") or row.get("study_id") or row.get("dataset") or source)
    )
    campaign_id = str(
        row.get("campaign_id") or row.get("dataset") or row.get("anchor_pdb") or source
    )
    molecule_id = str(row.get("molecule_id") or row.get("anchor_pdb") or study_id)

    assay_metric = str(
        row.get("assay_metric")
        or row.get("score_col")
        or row.get("metric")
        or row.get("dms_metric")
        or row.get("a3d_metric")
        or ""
    )
    assay_id = str(row.get("assay_id") or row.get("group_id") or study_id)
    if source_role == "supervised" and not assay_metric:
        raise ContractError("supervised source requires assay_metric")

    endpoint_direction = _resolve_endpoint_direction(
        row,
        source=source,
        source_role=source_role,
        assay_metric=assay_metric,
        score_col=str(row.get("score_col", "")),
    )
    if endpoint_direction not in ENDPOINT_DIRECTIONS:
        raise ContractError(f"invalid endpoint direction: {endpoint_direction!r}")

    endpoint_value = _coerce_float(
        _coalesce(
            row.get("endpoint_value"),
            row.get("score"),
            row.get("dms_score"),
            row.get("a3d_score"),
        )
    )
    endpoint_unit = row.get("endpoint_unit")
    if source_role == "supervised":
        if not isinstance(endpoint_unit, str):
            raise ContractError("supervised source requires endpoint_unit as a string")
        endpoint_unit = endpoint_unit.strip()
        if not endpoint_unit:
            raise ContractError("supervised source requires endpoint_unit")
    else:
        endpoint_unit = "" if endpoint_unit is None else str(endpoint_unit)
    if source_role == "supervised" and not endpoint_unit:
        raise ContractError("supervised source requires endpoint_unit")

    raw_label = row.get("label")
    raw_label_is_present = raw_label is not None and str(raw_label).strip() != ""
    binary_label = _coerce_binary_label(raw_label)
    label_threshold = _coerce_float(row.get("label_threshold"))

    if source_role == "supervised":
        if endpoint_value is None:
            raise ContractError("supervised source requires endpoint_value")
        if raw_label_is_present and binary_label is None:
            raise ContractError("supervised source requires valid binary label")
        if binary_label is not None and label_threshold is None:
            raise ContractError(
                "supervised row with binary label requires explicit label_threshold"
            )
        target_value = float(endpoint_value)
    else:
        target_value = None
        binary_label = None
        label_threshold = None
    if source_role == "supervised" and binary_label is None and label_threshold is not None:
        raise ContractError("label_threshold requires a binary label")

    pair_id = row.get("pair_id")
    if pair_id is not None and not str(pair_id).strip():
        pair_id = None
    if pair_id is not None and (vh_sequence is None or vl_sequence is None):
        raise ContractError("pair_id requires complete vh/vl pairing")
    pair_id = str(pair_id) if pair_id is not None else None

    chain_id = _normalize_chain_id(row)
    if pair_id is not None and chain_id in {"vh", "vl"}:
        raise ContractError("pair_id is not allowed on single-chain rows")

    feature_flags = sorted(set(str(v) for v in row.get("feature_flags", [])))
    if pair_id is None:
        feature_flags = sorted(set(feature_flags) | {"single_chain_only"})

    source_url_value = str(source_url or row.get("source_url") or "")
    source_version = str(row.get("source_version") or "")
    raw_conditions = row.get("assay_conditions")
    if isinstance(raw_conditions, dict):
        assay_conditions = dict(raw_conditions)
    else:
        condition_keys = (
            "concentration", "temperature", "buffer", "ph", "salt", "incubation_time",
            "replicate_count", "instrument", "formulation",
        )
        assay_conditions = {
            key: row[key] for key in condition_keys
            if key in row and row[key] not in (None, "")
        }
    source_row_hash = hash_payload(
        row,
        exclude_keys=SOURCE_ROW_HASH_EXCLUDE_KEYS,
    )

    rec_source = row.get("record_id")
    record_id = str(
        rec_source or f"{source}:{study_id}:{campaign_id}:{sequence_hash}"
    )

    return NormalizedRow(
        record_id=record_id,
        source=source,
        study_id=study_id,
        campaign_id=campaign_id,
        molecule_id=molecule_id,
        pair_id=pair_id,
        chain_id=chain_id,
        sequence=sequence,
        vh_sequence=vh_sequence,
        vl_sequence=vl_sequence,
        assay_id=assay_id,
        assay_metric=assay_metric,
        endpoint_direction=endpoint_direction,
        endpoint_value=endpoint_value,
        endpoint_unit=endpoint_unit,
        target_value=target_value,
        supervision_status=source_role,
        binary_label=binary_label,
        label_threshold=label_threshold,
        group_study=study_id,
        group_molecule=molecule_id,
        group_campaign=campaign_id,
        sequence_hash=sequence_hash,
        exclusions=list(sorted(set(str(v) for v in row.get("exclusions", [])))),
        source_url=source_url_value,
        source_row_hash=source_row_hash,
        feature_flags=feature_flags,
        source_version=source_version,
        assay_conditions=assay_conditions,
    )
