"""Source registry and deterministic fetch plan definitions for Plan C.

The registry is intentionally metadata-only: source URLs, schema declarations,
and stable plan materialization. Actual network fetch and row extraction live in
`data.*` source modules.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from data.contract import (
    ContractError,
    canonical_source_name,
    canonical_json_dumps,
    hash_payload,
    source_supervision_role,
)

REGISTRY_VERSION = "1"

_FIELD_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must be a non-empty string")
    return stripped


def _validate_http_urls(urls: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for url in urls:
        trimmed = _require_non_empty_string(url, "source URL")
        parsed = urlparse(trimmed)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid source URL: {url!r}")
        normalized.append(trimmed)
    return tuple(sorted(dict.fromkeys(normalized)))


def _validate_headers(
    headers: Any,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(headers, tuple):
        raise ValueError("request_headers must be a tuple of string pairs")
    normalized: dict[str, str] = {}
    forbidden = {"authorization", "x-api-key", "api-key"}
    for item in headers:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("request_headers must be a tuple of string pairs")
        key, value = item
        normalized_key = _require_non_empty_string(key, "request header key").strip()
        if normalized_key.lower() in forbidden:
            raise ValueError("request headers must not contain credentials")
        normalized_value = _require_non_empty_string(value, "request header value").strip()
        normalized[normalized_key] = normalized_value
    return tuple((k, normalized[k]) for k in sorted(normalized))


def _validate_field_names(
    values: Any,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be a tuple")
    normalized: list[str] = []
    for item in values:
        normalized_item = _require_non_empty_string(item, f"{label} entry")
        if not _FIELD_NAME_RE.fullmatch(normalized_item):
            raise ValueError(f"{label} entry is invalid: {item!r}")
        normalized.append(normalized_item)
    return tuple(sorted(dict.fromkeys(normalized)))


def _validate_aliases(values: Any) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError("source_aliases must be a tuple")
    normalized: list[str] = []
    for item in values:
        normalized_item = _require_non_empty_string(item, "source_aliases entry").lower()
        normalized.append(normalized_item)
    return tuple(sorted(dict.fromkeys(normalized)))


@dataclass(frozen=True)
class SourceDescriptor:
    source: str
    supervision_status: str
    source_urls: tuple[str, ...]
    request_headers: tuple[tuple[str, str], ...]
    required_row_fields: tuple[str, ...]
    optional_row_fields: tuple[str, ...]
    row_parser_hint: str
    schema_version: str = "1"
    source_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_source = _require_non_empty_string(self.source, "source")
        try:
            normalized_source = canonical_source_name(normalized_source)
        except ContractError as exc:
            raise ValueError(f"invalid source name: {self.source!r}") from exc

        if not isinstance(self.source_urls, tuple):
            raise ValueError("source_urls must be a tuple")

        normalized_supervision_status = _require_non_empty_string(
            self.supervision_status, "supervision_status"
        ).lower()
        expected_role = source_supervision_role(normalized_source)
        if normalized_supervision_status != expected_role:
            raise ValueError(
                f"supervision_status must match contract role for source {normalized_source!r}"
            )

        normalized_source_urls = _validate_http_urls(self.source_urls)
        normalized_headers = _validate_headers(self.request_headers)
        normalized_required_fields = _validate_field_names(
            self.required_row_fields, "required_row_fields"
        )
        normalized_optional_fields = _validate_field_names(
            self.optional_row_fields, "optional_row_fields"
        )
        overlap = set(normalized_required_fields).intersection(normalized_optional_fields)
        if overlap:
            formatted = ", ".join(sorted(overlap))
            raise ValueError(f"required and optional fields overlap: {formatted}")

        normalized_row_parser_hint = _require_non_empty_string(self.row_parser_hint, "row_parser_hint")
        normalized_schema_version = _require_non_empty_string(self.schema_version, "schema_version")
        normalized_aliases = _validate_aliases(self.source_aliases)
        if normalized_source in normalized_aliases:
            raise ValueError("source_aliases cannot include canonical source name")

        object.__setattr__(self, "source", normalized_source)
        object.__setattr__(self, "supervision_status", normalized_supervision_status)
        object.__setattr__(self, "source_urls", normalized_source_urls)
        object.__setattr__(self, "request_headers", normalized_headers)
        object.__setattr__(self, "required_row_fields", normalized_required_fields)
        object.__setattr__(self, "optional_row_fields", normalized_optional_fields)
        object.__setattr__(self, "row_parser_hint", normalized_row_parser_hint)
        object.__setattr__(self, "schema_version", normalized_schema_version)
        object.__setattr__(self, "source_aliases", normalized_aliases)

    def as_plan_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "supervision_status": self.supervision_status,
            "schema_version": self.schema_version,
            "source_urls": list(self.source_urls),
            "request_headers": [list(item) for item in self.request_headers],
            "required_row_fields": list(self.required_row_fields),
            "optional_row_fields": list(self.optional_row_fields),
            "row_parser_hint": self.row_parser_hint,
        }

    def row_schema_hash(self) -> str:
        payload = {
            "source": self.source,
            "supervision_status": self.supervision_status,
            "schema_version": self.schema_version,
            "required_row_fields": list(self.required_row_fields),
            "optional_row_fields": list(self.optional_row_fields),
        }
        return hash_payload(payload)

    def descriptor_hash(self) -> str:
        payload = asdict(self)
        payload["source_urls"] = list(self.source_urls)
        payload["request_headers"] = list(self.request_headers)
        payload["required_row_fields"] = list(self.required_row_fields)
        payload["optional_row_fields"] = list(self.optional_row_fields)
        payload["source_aliases"] = list(self.source_aliases)
        return hash_payload(payload)


def _source_descriptor(**kwargs: Any) -> SourceDescriptor:
    source_urls = kwargs["source_urls"]
    request_headers = kwargs.get("request_headers", ())
    required_row_fields = kwargs["required_row_fields"]
    optional_row_fields = kwargs["optional_row_fields"]
    source_aliases = kwargs.get("source_aliases", ())

    if not isinstance(source_urls, (list, tuple)):
        raise ValueError("source_urls must be a tuple or list")
    if not isinstance(request_headers, tuple):
        raise ValueError("request_headers must be a tuple")
    if not isinstance(required_row_fields, (list, tuple)):
        raise ValueError("required_row_fields must be a tuple")
    if not isinstance(optional_row_fields, (list, tuple)):
        raise ValueError("optional_row_fields must be a tuple")
    if not isinstance(source_aliases, tuple):
        raise ValueError("source_aliases must be a tuple")

    return SourceDescriptor(
        source=str(kwargs["source"]),
        supervision_status=str(kwargs["supervision_status"]),
        source_urls=tuple(source_urls),
        request_headers=tuple(request_headers),
        required_row_fields=tuple(required_row_fields),
        optional_row_fields=tuple(optional_row_fields),
        row_parser_hint=str(kwargs["row_parser_hint"]),
        schema_version=str(kwargs.get("schema_version", "1")),
        source_aliases=tuple(source_aliases),
    )


def _build_alias_index(catalog: dict[str, SourceDescriptor]) -> dict[str, str]:
    alias_index: dict[str, str] = {}
    for canonical, descriptor in catalog.items():
        for alias in descriptor.source_aliases:
            if alias in catalog:
                raise ValueError(
                    f"alias {alias!r} for {canonical!r} collides with canonical source"
                )
            existing = alias_index.get(alias)
            if existing is not None:
                raise ValueError(
                    f"alias {alias!r} maps to multiple sources: {existing!r}, {canonical!r}"
                )
            alias_index[alias] = canonical
    return alias_index


_SOURCE_CATALOG: dict[str, SourceDescriptor] = {
    "flab": _source_descriptor(
        source="flab",
        supervision_status="supervised",
        source_urls=(
            "https://api.github.com/repos/Graylab/FLAb/contents/data/aggregation",
            "https://raw.githubusercontent.com/Graylab/FLAb/main/data/aggregation",
        ),
        request_headers=(("Accept", "application/vnd.github.v3+json"),),
        required_row_fields=(
            "source",
            "variant_sequence",
            "study_id",
            "campaign_id",
            "molecule_id",
            "assay_id",
            "assay_metric",
            "endpoint_direction",
            "endpoint_value",
            "endpoint_unit",
            "target_value",
        ),
        optional_row_fields=(
            "supervision_status",
            "pair_id",
            "chain_id",
            "binary_label",
            "label_threshold",
            "source_url",
            "source_row_hash",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit supervised continuous-normalized rows with assay endpoint/value/unit and "
            "study/campaign/molecule provenance fields."
        ),
    ),
    "proteingym": _source_descriptor(
        source="proteingym",
        supervision_status="auxiliary",
        source_urls=(
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv",
            "https://huggingface.co/datasets/OATML-Markslab/ProteinGym/resolve/main/DMS_ProteinGym_substitutions",
            "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/DMS_ProteinGym_substitutions",
        ),
        request_headers=(),
        required_row_fields=(
            "source",
            "variant_sequence",
            "study_id",
            "campaign_id",
            "molecule_id",
            "assay_id",
            "assay_metric",
            "endpoint_direction",
            "endpoint_value",
            "endpoint_unit",
        ),
        optional_row_fields=(
            "target_value",
            "pair_id",
            "chain_id",
            "binary_label",
            "label_threshold",
            "source_url",
            "source_row_hash",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit auxiliary continuous rows from ProteinGym DMS endpoints with explicit "
            "assay context and direction; avoid supervised labels."
        ),
        source_aliases=("proteingym_dms",),
    ),
    "canya": _source_descriptor(
        source="canya",
        supervision_status="auxiliary",
        source_urls=(
            "https://api.github.com/repos/orozco-lab/CANYA/contents",
            "https://raw.githubusercontent.com/orozco-lab/CANYA/main",
            "https://zenodo.org/api/records?q=CANYA+nucleation+aggregation&sort=mostrecent&size=3",
        ),
        request_headers=(("Accept", "application/vnd.github.v3+json"),),
        required_row_fields=(
            "source",
            "variant_sequence",
            "study_id",
            "campaign_id",
            "molecule_id",
            "assay_id",
            "assay_metric",
            "endpoint_direction",
            "endpoint_value",
            "endpoint_unit",
        ),
        optional_row_fields=(
            "target_value",
            "pair_id",
            "chain_id",
            "binary_label",
            "label_threshold",
            "source_url",
            "source_row_hash",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit auxiliary A3D/nucleation peptide rows with explicit assay identity and "
            "continuous endpoint values; no supervised target synthesis."
        ),
    ),
    "figshare_agg": _source_descriptor(
        source="figshare_agg",
        supervision_status="auxiliary",
        source_urls=("https://api.figshare.com/v2/articles/22492606/files",),
        request_headers=(),
        required_row_fields=(
            "source",
            "variant_sequence",
            "study_id",
            "campaign_id",
            "molecule_id",
            "assay_id",
            "assay_metric",
            "endpoint_direction",
            "endpoint_value",
            "endpoint_unit",
        ),
        optional_row_fields=(
            "target_value",
            "pair_id",
            "chain_id",
            "binary_label",
            "label_threshold",
            "source_url",
            "source_row_hash",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit auxiliary A3D summary rows with explicit assay/context metadata and "
            "continuous aux endpoint only."
        ),
        source_aliases=("figshare_a3d", "figshareagg", "figshare-a3d", "figshare_a3"),
    ),
    "abdev": _source_descriptor(
        source="abdev",
        supervision_status="background_ood",
        source_urls=(
            "https://api.github.com/repos/Lailabcode/AbDev/contents",
            "https://raw.githubusercontent.com/Lailabcode/AbDev/main",
        ),
        request_headers=(("Accept", "application/vnd.github.v3+json"),),
        required_row_fields=("source", "variant_sequence"),
        optional_row_fields=(
            "assay_metric",
            "study_id",
            "campaign_id",
            "molecule_id",
            "source_url",
            "source_row_hash",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit unlabeled background references from AbDev as non-supervision OOD rows."
        ),
        source_aliases=("ab-dev",),
    ),
    "antiref": _source_descriptor(
        source="antiref",
        supervision_status="background_ood",
        source_urls=(
            "https://api.github.com/repos/brineylab/antiref/contents",
            "https://raw.githubusercontent.com/brineylab/antiref/main",
        ),
        request_headers=(("Accept", "application/vnd.github.v3+json"),),
        required_row_fields=("source", "variant_sequence"),
        optional_row_fields=(
            "assay_metric",
            "study_id",
            "campaign_id",
            "molecule_id",
            "source_url",
            "source_row_hash",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit unlabeled germline reference sequences as background OOD rows only."
        ),
    ),
    "sabdab": _source_descriptor(
        source="sabdab",
        supervision_status="background_ood",
        source_urls=(
            "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/?format=csv&CDRdef=chothia&rfactor=&resolution=3.0&otype=All&otype=&method=X-RAY+DIFFRACTION",
            "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/sequence/{pdb_id}/?chain_type={chain_type}&fasta=1",
        ),
        request_headers=(),
        required_row_fields=("source", "variant_sequence"),
        optional_row_fields=(
            "assay_metric",
            "study_id",
            "campaign_id",
            "molecule_id",
            "source_url",
            "source_row_hash",
            "anchor_pdb",
            "chain_type",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit unlabeled structural OOD rows from SAbDab; do not infer working labels."
        ),
    ),
    "pdb_anchor": _source_descriptor(
        source="pdb_anchor",
        supervision_status="background_ood",
        source_urls=(),
        request_headers=(),
        required_row_fields=("source", "variant_sequence"),
        optional_row_fields=(
            "anchor_pdb",
            "chain_type",
            "group_id",
            "assay_metric",
            "study_id",
            "campaign_id",
            "molecule_id",
            "source_url",
            "source_row_hash",
            "record_id",
            "sequence_hash",
        ),
        row_parser_hint=(
            "Emit phase-1 chain anchor references from local results; no network and no labels."
        ),
        source_aliases=("anchors", "anchor"),
    ),
}


_ALIAS_INDEX: dict[str, str] = _build_alias_index(_SOURCE_CATALOG)


def _normalize_source_name(source: str) -> str:
    try:
        canonical = canonical_source_name(source)
    except ContractError as exc:
        raise ValueError(f"invalid source name: {source!r}") from exc
    return _ALIAS_INDEX.get(canonical, canonical)


def list_sources() -> tuple[str, ...]:
    """Return all canonical source identifiers in stable order."""
    return tuple(sorted(_SOURCE_CATALOG))


def get_source_descriptor(source: str) -> SourceDescriptor:
    """Resolve and return the registered descriptor for one source."""
    normalized = _normalize_source_name(source)
    try:
        return _SOURCE_CATALOG[normalized]
    except KeyError as exc:
        message = ", ".join(list_sources())
        raise ValueError(
            f"unknown source {source!r}; known sources are: {message}"
        ) from exc


def _normalize_source_filter(
    source_filter: str | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if source_filter is None:
        return list_sources()
    if isinstance(source_filter, str):
        stripped = source_filter.strip()
        if not stripped:
            raise ValueError("source_filter cannot be empty")
        if stripped.lower() in {"all", "*"}:
            return list_sources()
        source_items = [stripped]
    else:
        if not isinstance(source_filter, (list, tuple)):
            raise TypeError("source_filter must be a string or sequence of strings")
        source_items = list(source_filter)
        if not source_items:
            raise ValueError("source_filter cannot be empty")
    if any(not isinstance(value, str) for value in source_items):
        raise TypeError("source names must be strings")
    normalized = [_normalize_source_name(value) for value in source_items]
    if len(normalized) != len(set(normalized)):
        normalized = list(dict.fromkeys(normalized))

    return tuple(sorted(get_source_descriptor(value).source for value in normalized))


def build_fetch_plan(source_filter: str | list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    """
    Build a deterministic, manifest-only source fetch plan.
    """
    normalized_sources = _normalize_source_filter(source_filter)
    source_plans = [get_source_descriptor(source).as_plan_payload() for source in normalized_sources]
    payload: dict[str, Any] = {
        "registry_version": REGISTRY_VERSION,
        "schema_version": "1",
        "sources": source_plans,
        "source_count": len(source_plans),
    }
    payload["fetch_plan_hash"] = hash_payload(payload)
    return payload


def fetch_plan_to_json(plan: dict[str, Any]) -> str:
    """Serialize a fetch plan using deterministic canonical JSON."""
    return canonical_json_dumps(plan)
