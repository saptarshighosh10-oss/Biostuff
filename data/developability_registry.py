"""Streaming provenance registry for the developability program."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data.contract import ContractError, canonical_sequence, hash_payload, sha256_hex
from data.run_manifest import file_sha256
from data.sequence_reconstruction import reconstruct_verified_pair


SCHEMA_VERSION = "2"
CDR_FIELDS = ("cdr_h1", "cdr_h2", "cdr_h3", "cdr_l1", "cdr_l2", "cdr_l3")
LIBRARY_SOURCES = {"emi", "igg", "iso", "tessier_lib1", "tessier_lib2"}
NARROW_AUXILIARY_SOURCES = {"harvey2022_vhh"}
UNIT_BY_LABEL_TYPE = {
    "hic_retention_min": "min",
    "hic_relative_retention": "relative_retention",
    "num_react": "reagent_count",
    "ova_binary": "binary_call",
    "polyreactivity_highlow": "binary_call",
    "psr_binary_vhh": "binary_call",
    "yn_binary": "binary_call",
    "psr_continuous": "reported_psr_value",
    "psr_smp_continuous": "reported_psr_value",
}
PROPERTY_BY_FAMILY = {
    "aggregation": "aggregation",
    "self_association": "aggregation",
    "colloidal_stability": "aggregation",
    "hydrophobicity": "hydrophobicity",
    "polyreactivity": "polyreactivity",
}


@dataclass(frozen=True)
class RegistryInput:
    path: Path
    property_name: str
    table_format: str


def default_inputs() -> list[RegistryInput]:
    root = Path.home() / "Downloads" / "antibody_sources"
    inputs = [
        RegistryInput(root / "polyreactivity_dataset.csv", "polyreactivity", "sequence_table"),
        RegistryInput(root / "hydrophobicity_dataset.csv", "hydrophobicity", "sequence_table"),
        RegistryInput(root / "polyreactivity_cdr_dataset.csv", "polyreactivity", "cdr_table"),
    ]
    for property_name in ("polyreactivity", "aggregation", "hydrophobicity"):
        path = root / f"reconstructed_{property_name}_dataset.csv"
        if path.exists():
            inputs.append(RegistryInput(path, property_name, "reconstructed_sequence_table"))
    return inputs


def default_model_rows() -> list[dict]:
    """Load existing served-property rows without touching GDPa3."""
    from model.head_b_acsins_unified import _acsins_rows
    from model.head_developability_honest import _load_rows

    rows = [
        row
        for row in _load_rows()
        if str(row.get("assay_family")) in PROPERTY_BY_FAMILY
    ]
    rows.extend(_acsins_rows(True))
    return rows


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sequence(value: object, *, allow_empty: bool = False, min_length: int = 1) -> str | None:
    text = "" if value is None else str(value).strip()
    if not text and allow_empty:
        return None
    try:
        sequence = canonical_sequence(text)
    except ContractError as exc:
        raise ValueError("invalid amino-acid sequence") from exc
    if len(sequence) < min_length:
        raise ValueError("amino-acid sequence is too short")
    return sequence


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("label is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError("label is not finite")
    return number


def _source_class(source: str) -> str:
    return "LIBRARY" if source in LIBRARY_SOURCES else "DIVERSE"


def _row_payload(spec: RegistryInput, row: dict[str, str], row_number: int) -> dict:
    source = (row.get("source") or "").strip()
    if not source:
        raise ValueError("source is missing")
    label_type = (row.get("label_type") or "unlabeled").strip()
    if spec.table_format != "unlabeled_table" and label_type not in UNIT_BY_LABEL_TYPE:
        raise ValueError(f"unsupported label_type: {label_type or '<empty>'}")
    endpoint_value = None if spec.table_format == "unlabeled_table" else _number(row.get("label"))

    vh = vl = None
    verification: dict = {}
    cdrs: dict[str, str | None] = {field: None for field in CDR_FIELDS}
    if spec.table_format == "cdr_table":
        for field in CDR_FIELDS:
            cdrs[field] = _sequence(row.get(field), min_length=1)
        supervision_tier = "cdr_supervised"
        molecule_key = "cdr:" + sha256_hex("\0".join(cdrs[field] or "" for field in CDR_FIELDS))
        identity_component_id = molecule_key
        evidence_tier = "direct"
        training_eligible = holdout_eligible = 1
    elif spec.table_format in {"sequence_table", "unlabeled_table"}:
        vh = _sequence(row.get("vh") or row.get("vh_sequence"), min_length=20)
        vl = _sequence(row.get("vl") or row.get("vl_sequence"), allow_empty=True, min_length=20)
        supervised = spec.table_format != "unlabeled_table"
        if vl:
            supervision_tier = "paired_fv_supervised" if supervised else "unlabeled_antibody"
            molecule_key = "pair:" + sha256_hex(f"{vh}\0{vl}")
        else:
            supervision_tier = "single_chain_supervised" if supervised else "unlabeled_antibody"
            molecule_key = "vh:" + sha256_hex(vh or "")
        identity_component_id = "vh:" + sha256_hex(vh or "")
        evidence_tier = "direct" if supervised else "unlabeled"
        training_eligible = holdout_eligible = int(supervised)
    elif spec.table_format == "reconstructed_sequence_table":
        vh, vl, verification = reconstruct_verified_pair(row)
        vh = _sequence(vh, min_length=20)
        vl = _sequence(vl, allow_empty=True, min_length=20)
        if vl:
            supervision_tier = "paired_fv_supervised"
            molecule_key = "pair:" + sha256_hex(f"{vh}\0{vl}")
        else:
            supervision_tier = "single_chain_supervised"
            molecule_key = "vh:" + sha256_hex(vh or "")
        identity_component_id = "vh:" + sha256_hex(vh or "")
        evidence_tier = "reconstructed_verified"
        training_eligible = 1
        holdout_eligible = 0
    else:
        raise ValueError(f"unsupported table format: {spec.table_format}")

    source_row_hash = hash_payload(row)
    record_id = hash_payload(
        {
            "property": spec.property_name,
            "source": source,
            "row_number": row_number,
            "source_row_hash": source_row_hash,
        }
    )
    split_group = (
        f"source:{source}"
        if source in LIBRARY_SOURCES | NARROW_AUXILIARY_SOURCES
        else identity_component_id
    )
    return {
        "record_id": record_id,
        "source": source,
        "property": spec.property_name,
        "assay_metric": label_type,
        "label_type": label_type,
        "endpoint_value": endpoint_value,
        "endpoint_unit": UNIT_BY_LABEL_TYPE.get(label_type),
        "endpoint_direction": "higher_bad",
        "supervision_tier": supervision_tier,
        "source_class": _source_class(source),
        "vh": vh,
        "vl": vl,
        **cdrs,
        "vh_hash": sha256_hex(vh) if vh else None,
        "vl_hash": sha256_hex(vl) if vl else None,
        "pair_hash": sha256_hex(f"{vh}\0{vl}") if vh and vl else None,
        "molecule_key": molecule_key,
        "identity_component_id": identity_component_id,
        "split_group": split_group,
        "source_row_hash": source_row_hash,
        "source_file": str(spec.path.expanduser().resolve()),
        "source_row_number": row_number,
        "assay_context_json": "{}",
        "evidence_tier": evidence_tier,
        "training_eligible": training_eligible,
        "holdout_eligible": holdout_eligible,
        "verification_json": json.dumps(verification, sort_keys=True, separators=(",", ":")),
        "provenance_status": (
            "deterministic_reconstruction_verified"
            if evidence_tier == "reconstructed_verified"
            else "local_consolidated_artifact"
        ),
        "status": "accepted",
        "exclusion_reason": None,
    }


def _model_row_payload(row: dict, row_number: int) -> dict:
    family = str(row.get("assay_family") or "").strip()
    property_name = PROPERTY_BY_FAMILY.get(family)
    if property_name is None:
        raise ValueError(f"unsupported assay_family: {family or '<empty>'}")
    source = str(row.get("study_id") or row.get("source") or "").strip()
    if not source:
        raise ValueError("source is missing")
    metric = str(row.get("assay_metric") or "").strip()
    if not metric:
        raise ValueError("assay_metric is missing")
    assay_id = str(row.get("assay_id") or metric).strip()
    endpoint_value = _number(row.get("endpoint_value"))
    direction = str(row.get("endpoint_direction") or "").strip()
    if direction not in {"higher_bad", "lower_bad"}:
        raise ValueError("endpoint_direction is invalid")

    raw_vh = row.get("vh_sequence")
    raw_vl = row.get("vl_sequence")
    if not raw_vh:
        raw_vh = row.get("variant_sequence") or row.get("sequence")
    vh = _sequence(raw_vh, min_length=20)
    vl = _sequence(raw_vl, allow_empty=True, min_length=20)
    if vl:
        supervision_tier = "paired_fv_supervised"
        sequence_molecule_key = "pair:" + sha256_hex(f"{vh}\0{vl}")
    else:
        supervision_tier = "single_chain_supervised"
        sequence_molecule_key = "vh:" + sha256_hex(vh or "")
    # Some project loaders represent combinatorial constructs with only one
    # chain sequence but retain a stable molecule ID (for example GDPa4.1
    # VH-by-VL constructs). Treat those constructs as distinct measurement
    # subjects without inventing the missing chain. Sequence identity remains
    # the leakage group below.
    molecule_id = str(row.get("molecule_id") or "").strip()
    molecule_key = (
        "entity:" + sha256_hex(f"{source}\0{molecule_id}")
        if molecule_id
        else sequence_molecule_key
    )
    identity_component_id = "vh:" + sha256_hex(vh or "")
    label_type = f"{assay_id}:{metric}"
    context_keys = (
        "concentration", "temperature", "buffer", "ph", "salt", "incubation_time",
        "formulation", "instrument", "antibody_format", "isotype", "molecule_id",
        "molecule_group_id", "group_id", "campaign_id", "construct_type",
    )
    context = {key: row[key] for key in context_keys if row.get(key) not in (None, "")}
    source_row_hash = hash_payload(row)
    record_id = hash_payload(
        {
            "loader": "served_project_rows",
            "source": source,
            "row_number": row_number,
            "source_row_hash": source_row_hash,
        }
    )
    return {
        "record_id": record_id,
        "source": source,
        "property": property_name,
        "assay_metric": label_type,
        "label_type": label_type,
        "endpoint_value": endpoint_value,
        "endpoint_unit": str(row.get("endpoint_unit") or "reported_assay_value"),
        "endpoint_direction": direction,
        "supervision_tier": supervision_tier,
        "source_class": "DIVERSE",
        "vh": vh,
        "vl": vl,
        **{field: None for field in CDR_FIELDS},
        "vh_hash": sha256_hex(vh or ""),
        "vl_hash": sha256_hex(vl) if vl else None,
        "pair_hash": sha256_hex(f"{vh}\0{vl}") if vl else None,
        "molecule_key": molecule_key,
        "identity_component_id": identity_component_id,
        "split_group": identity_component_id,
        "source_row_hash": source_row_hash,
        "source_file": "project_loader:served_project_rows",
        "source_row_number": row_number,
        "assay_context_json": json.dumps(context, sort_keys=True, separators=(",", ":")),
        "evidence_tier": "direct",
        "training_eligible": 1,
        "holdout_eligible": 1,
        "verification_json": "{}",
        "provenance_status": "project_loader_with_source_metadata",
        "status": "accepted",
        "exclusion_reason": None,
    }


_COLUMNS = (
    "record_id", "source", "property", "assay_metric", "label_type", "endpoint_value",
    "endpoint_unit", "endpoint_direction", "supervision_tier", "source_class", "vh", "vl",
    *CDR_FIELDS, "vh_hash", "vl_hash", "pair_hash", "molecule_key", "identity_component_id",
    "split_group", "source_row_hash", "source_file", "source_row_number", "assay_context_json",
    "evidence_tier", "training_eligible", "holdout_eligible", "verification_json",
    "provenance_status", "status", "exclusion_reason",
)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            property TEXT NOT NULL,
            assay_metric TEXT NOT NULL,
            label_type TEXT NOT NULL,
            endpoint_value REAL,
            endpoint_unit TEXT,
            endpoint_direction TEXT NOT NULL,
            supervision_tier TEXT NOT NULL,
            source_class TEXT NOT NULL,
            vh TEXT,
            vl TEXT,
            cdr_h1 TEXT, cdr_h2 TEXT, cdr_h3 TEXT,
            cdr_l1 TEXT, cdr_l2 TEXT, cdr_l3 TEXT,
            vh_hash TEXT,
            vl_hash TEXT,
            pair_hash TEXT,
            molecule_key TEXT NOT NULL,
            identity_component_id TEXT NOT NULL,
            split_group TEXT NOT NULL,
            source_row_hash TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            assay_context_json TEXT NOT NULL,
            evidence_tier TEXT NOT NULL,
            training_eligible INTEGER NOT NULL CHECK(training_eligible IN (0, 1)),
            holdout_eligible INTEGER NOT NULL CHECK(holdout_eligible IN (0, 1)),
            verification_json TEXT NOT NULL,
            provenance_status TEXT NOT NULL,
            status TEXT NOT NULL,
            exclusion_reason TEXT
        );
        CREATE TABLE rejections (
            source_file TEXT NOT NULL,
            source_row_number INTEGER NOT NULL,
            source TEXT,
            source_row_hash TEXT NOT NULL,
            reason TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );
        CREATE TABLE input_files (
            path TEXT PRIMARY KEY,
            property TEXT NOT NULL,
            table_format TEXT NOT NULL,
            sha256 TEXT NOT NULL
        );
        CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX records_source ON records(source);
        CREATE INDEX records_property ON records(property);
        CREATE INDEX records_tier ON records(supervision_tier);
        CREATE INDEX records_vh_hash ON records(vh_hash);
        CREATE INDEX records_vl_hash ON records(vl_hash);
        CREATE INDEX records_pair_hash ON records(pair_hash);
        CREATE INDEX records_split_group ON records(split_group);
        CREATE INDEX records_measurement
            ON records(property, assay_metric, molecule_key, endpoint_value, status);
        """
    )


def _insert_records(connection: sqlite3.Connection, rows: list[dict]) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in _COLUMNS)
    connection.executemany(
        f"INSERT INTO records ({','.join(_COLUMNS)}) VALUES ({placeholders})",
        [[row[column] for column in _COLUMNS] for row in rows],
    )


def _mark_conflicts_and_duplicates(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE conflict_keys AS
        SELECT property, assay_metric, molecule_key
        FROM records
        GROUP BY property, assay_metric, molecule_key
        HAVING MIN(endpoint_value) <> MAX(endpoint_value)
        """
    )
    connection.execute(
        "CREATE INDEX conflict_keys_lookup ON conflict_keys(property, assay_metric, molecule_key)"
    )
    connection.execute(
        """
        UPDATE records
        SET status = 'quarantined_conflict',
            exclusion_reason = 'conflicting labels for the same molecule and endpoint'
        WHERE EXISTS (
            SELECT 1 FROM conflict_keys AS conflict
            WHERE conflict.property = records.property
              AND conflict.assay_metric = records.assay_metric
              AND conflict.molecule_key = records.molecule_key
        )
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE duplicate_keep AS
        SELECT record_id
        FROM (
            SELECT record_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY property, assay_metric, molecule_key, endpoint_value
                       ORDER BY CASE evidence_tier
                           WHEN 'direct' THEN 0
                           WHEN 'reconstructed_verified' THEN 1
                           ELSE 2
                       END, record_id
                   ) AS priority
            FROM records
            WHERE status = 'accepted'
        )
        WHERE priority = 1
        """
    )
    connection.execute("CREATE UNIQUE INDEX duplicate_keep_record ON duplicate_keep(record_id)")
    connection.execute(
        """
        UPDATE records
        SET status = 'duplicate', exclusion_reason = 'exact duplicate measurement'
        WHERE status = 'accepted'
          AND NOT EXISTS (SELECT 1 FROM duplicate_keep WHERE duplicate_keep.record_id = records.record_id)
        """
    )


def _counts(connection: sqlite3.Connection) -> dict:
    scalar = lambda query: int(connection.execute(query).fetchone()[0])
    accepted_where = "WHERE status = 'accepted'"
    report = {
        "accepted_rows": scalar(f"SELECT COUNT(*) FROM records {accepted_where}"),
        "quarantined_conflict_rows": scalar("SELECT COUNT(*) FROM records WHERE status = 'quarantined_conflict'"),
        "duplicate_rows": scalar("SELECT COUNT(*) FROM records WHERE status = 'duplicate'"),
        "rejected_rows": scalar("SELECT COUNT(*) FROM rejections"),
        "unique_molecules": scalar(f"SELECT COUNT(DISTINCT molecule_key) FROM records {accepted_where}"),
        "unique_vh": scalar(f"SELECT COUNT(DISTINCT vh_hash) FROM records {accepted_where} AND vh_hash IS NOT NULL"),
        "unique_pairs": scalar(f"SELECT COUNT(DISTINCT pair_hash) FROM records {accepted_where} AND pair_hash IS NOT NULL"),
    }
    for key, column in (("source_counts", "source"), ("tier_counts", "supervision_tier"), ("property_counts", "property")):
        report[key] = {
            name: int(count)
            for name, count in connection.execute(
                f"SELECT {column}, COUNT(*) FROM records {accepted_where} GROUP BY {column} ORDER BY {column}"
            )
        }
    report["evidence_counts"] = {
        name: int(count)
        for name, count in connection.execute(
            f"SELECT evidence_tier, COUNT(*) FROM records {accepted_where} "
            "GROUP BY evidence_tier ORDER BY evidence_tier"
        )
    }
    report["training_eligible_rows"] = scalar(
        "SELECT COUNT(*) FROM records WHERE status = 'accepted' AND training_eligible = 1"
    )
    report["holdout_eligible_rows"] = scalar(
        "SELECT COUNT(*) FROM records WHERE status = 'accepted' AND holdout_eligible = 1"
    )
    report["cross_source_exact_vh_groups"] = scalar(
        "SELECT COUNT(*) FROM (SELECT vh_hash FROM records WHERE vh_hash IS NOT NULL "
        "GROUP BY vh_hash HAVING COUNT(DISTINCT source) > 1)"
    )
    report["cross_source_exact_vl_groups"] = scalar(
        "SELECT COUNT(*) FROM (SELECT vl_hash FROM records WHERE vl_hash IS NOT NULL "
        "GROUP BY vl_hash HAVING COUNT(DISTINCT source) > 1)"
    )
    return report


def build_registry(
    database: str | Path,
    inputs: Iterable[RegistryInput],
    *,
    model_rows: Iterable[dict] = (),
) -> dict:
    target = Path(database)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary_files = [temporary, Path(str(temporary) + "-journal"), Path(str(temporary) + "-wal"), Path(str(temporary) + "-shm")]
    for path in temporary_files:
        path.unlink(missing_ok=True)
    input_specs = list(inputs)
    input_manifest = []
    input_rows = 0
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            _create_schema(connection)
            batch: list[dict] = []
            for spec in input_specs:
                path = spec.path.expanduser().resolve()
                digest = file_sha256(path)
                if digest is None:
                    raise FileNotFoundError(path)
                connection.execute(
                    "INSERT INTO input_files VALUES (?, ?, ?, ?)",
                    (str(path), spec.property_name, spec.table_format, digest),
                )
                input_manifest.append(
                    {"path": str(path), "property": spec.property_name, "table_format": spec.table_format, "sha256": digest}
                )
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    for row_number, row in enumerate(csv.DictReader(stream), 2):
                        input_rows += 1
                        try:
                            batch.append(_row_payload(spec, row, row_number))
                        except ValueError as exc:
                            connection.execute(
                                "INSERT INTO rejections VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    str(path), row_number, (row.get("source") or "").strip() or None,
                                    hash_payload(row), str(exc), json.dumps(row, sort_keys=True),
                                ),
                            )
                        if len(batch) >= 2000:
                            _insert_records(connection, batch)
                            batch.clear()
                _insert_records(connection, batch)
                batch.clear()
            model_row_hashes = []
            for row_number, row in enumerate(model_rows, 1):
                input_rows += 1
                model_row_hashes.append(hash_payload(row))
                try:
                    batch.append(_model_row_payload(row, row_number))
                except ValueError as exc:
                    connection.execute(
                        "INSERT INTO rejections VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            "project_loader:served_project_rows", row_number,
                            str(row.get("study_id") or row.get("source") or "") or None,
                            hash_payload(row), str(exc), json.dumps(row, sort_keys=True, default=str),
                        ),
                    )
                if len(batch) >= 2000:
                    _insert_records(connection, batch)
                    batch.clear()
            _insert_records(connection, batch)
            if model_row_hashes:
                digest = hash_payload(model_row_hashes)
                connection.execute(
                    "INSERT INTO input_files VALUES (?, ?, ?, ?)",
                    ("project_loader:served_project_rows", "served_properties", "model_rows", digest),
                )
                input_manifest.append(
                    {
                        "path": "project_loader:served_project_rows",
                        "property": "served_properties",
                        "table_format": "model_rows",
                        "sha256": digest,
                    }
                )
            _mark_conflicts_and_duplicates(connection)
            report = {"schema_version": SCHEMA_VERSION, "input_rows": input_rows, **_counts(connection)}
            report["registry_hash"] = hash_payload({"inputs": input_manifest, "counts": report})
            report["input_files"] = input_manifest
            connection.executemany(
                "INSERT INTO registry_meta VALUES (?, ?)",
                [
                    ("schema_version", SCHEMA_VERSION),
                    ("generated_at", _utc_now()),
                    ("registry_hash", report["registry_hash"]),
                ],
            )
            connection.commit()
        os.replace(temporary, target)
        report["database"] = str(target.resolve())
        return report
    finally:
        for path in temporary_files:
            path.unlink(missing_ok=True)
