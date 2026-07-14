"""Cache and manifest helpers for deterministic frozen source snapshots."""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from data.contract import (
    HASH_EXCLUDE_KEYS,
    canonical_source_name,
    hash_payload,
    sha256_hex,
)


DEFAULT_CACHE_ROOT = Path("data/cache")
MANIFEST_DIR = "manifests"
RAW_DIR = "raw"
SOURCE_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
DATASET_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
MANIFEST_VERSION = "1"
ROW_HASH_EXCLUDE_KEYS = HASH_EXCLUDE_KEYS | {"source_row_hash"}


def _cache_root(cache_root: Path | str | None) -> Path:
    return Path(cache_root or DEFAULT_CACHE_ROOT)


def _assert_cache_path(
    cache_root: Path,
    path: Path,
    *,
    boundary: str,
) -> Path:
    root = _cache_root(cache_root)
    resolved_root = root.resolve()
    resolved_path = path.resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"{boundary} escapes resolved cache root") from exc

    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        try:
            relative_parts = resolved_path.relative_to(resolved_root).parts
        except ValueError as exc:
            raise RuntimeError(f"{boundary} escapes cache root") from exc

    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{boundary} includes symlink segment: {current}")

    return path


def _validate_source_name(source: Any) -> str:
    if not isinstance(source, str):
        raise ValueError(f"source must be a non-empty lowercase identifier")
    source_text = source.strip()
    if not source_text:
        raise ValueError(f"source must be a non-empty lowercase identifier")
    if source_text.lower() != source_text:
        raise ValueError(f"source must be lowercase: {source_text!r}")
    canonical = canonical_source_name(source_text)
    if not SOURCE_NAME_RE.fullmatch(canonical):
        raise ValueError(f"invalid source name: {source_text!r}")
    return canonical


def _validate_dataset_sha(dataset_sha: Any, *, field_name: str = "dataset_sha") -> str:
    if not isinstance(dataset_sha, str):
        raise ValueError(f"{field_name} must be a 64-char lowercase hex string")
    if not DATASET_SHA_RE.fullmatch(dataset_sha):
        raise ValueError(f"invalid {field_name}: {dataset_sha!r}")
    return dataset_sha


def _validate_writer_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError("rows must be a list")
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"rows[{row_index}] must be a dict")
        for key in row.keys():
            if not isinstance(key, str):
                raise RuntimeError(
                    f"rows[{row_index}] keys must be strings: {key!r} is not a string"
                )
    return list(rows)


def _validate_writer_source_urls(source_urls: Any) -> list[str]:
    if not isinstance(source_urls, list):
        raise RuntimeError("source_urls must be a list")
    normalized: list[str] = []
    for source_url_index, source_url in enumerate(source_urls):
        if not isinstance(source_url, str):
            raise RuntimeError(
                f"source_urls[{source_url_index}] must be a non-empty string"
            )
        stripped_source_url = source_url.strip()
        if not stripped_source_url:
            raise RuntimeError(
                f"source_urls[{source_url_index}] must be a non-empty string after trimming"
            )
        normalized.append(stripped_source_url)
    return normalized


def _manifest_path(cache_root: Path, source: str) -> Path:
    source = _validate_source_name(source)
    return _assert_cache_path(
        cache_root,
        cache_root / MANIFEST_DIR / f"{source}.json",
        boundary="manifest path",
    )


def _dataset_dir(cache_root: Path, source: str, dataset_sha: str) -> Path:
    source = _validate_source_name(source)
    dataset_sha = _validate_dataset_sha(dataset_sha)
    return _assert_cache_path(
        cache_root,
        cache_root / RAW_DIR / source / dataset_sha,
        boundary="dataset directory path",
    )


def _expected_dataset_dir(cache_root: Path, source: str, dataset_sha: str) -> str:
    dataset_dir = _dataset_dir(cache_root, source, dataset_sha)
    return str(dataset_dir.relative_to(cache_root))


def _expected_rows_file(cache_root: Path, source: str, dataset_sha: str) -> str:
    dataset_dir = _dataset_dir(cache_root, source, dataset_sha)
    return str(_rows_path(cache_root, dataset_dir).relative_to(cache_root))


def _rows_path(cache_root: Path, dataset_dir: Path) -> Path:
    return _assert_cache_path(
        cache_root,
        dataset_dir / "rows.jsonl.gz",
        boundary="rows file path",
    )


def _dataset_metadata_path(cache_root: Path, dataset_dir: Path) -> Path:
    return _assert_cache_path(
        cache_root,
        dataset_dir / "dataset.json",
        boundary="metadata file path",
    )


def _schema_hash(rows: list[dict[str, Any]]) -> str:
    all_keys: list[str] = []
    for row in rows:
        all_keys.extend(row.keys())
    return hash_payload({"keys": sorted(set(all_keys)), "count": len(rows)})


def _row_hash(row: dict[str, Any]) -> str:
    return hash_payload(row, exclude_keys=ROW_HASH_EXCLUDE_KEYS)


def _rows_payload(rows: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in _deterministic_rows(rows)
    ]
    return "\n".join(lines).encode("utf-8")


def _deterministic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (row.get("sequence_hash", ""), row.get("record_id", "")),
    )


def _manifest_hash(payload: dict[str, Any]) -> str:
    return hash_payload(payload, exclude_keys={"fetch_timestamp_utc", "manifest_content_hash"})


def _metadata_hash(payload: dict[str, Any]) -> str:
    return hash_payload(payload, exclude_keys={"metadata_hash"})


def _temp_path(path: Path) -> Path:
    token = uuid.uuid4().hex
    return path.with_name(f".{path.name}.{token}.tmp")


def _fsync_parent(path: Path) -> None:
    parent = path if path.is_dir() else path.parent
    fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_gzip_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    try:
        compressed = gzip.compress(content, compresslevel=9, mtime=0)
        with open(temp_path, "wb") as f:
            f.write(compressed)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        _fsync_parent(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def manifest_content_hash(payload: dict[str, Any]) -> str:
    """Public helper used by tests/clients to recompute manifest identity."""
    return _manifest_hash(payload)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"failed to read {label}: {path}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to decode {label} JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} payload must be a JSON object")
    return payload


def _read_rows_payload(path: Path, *, source: str, dataset_sha: str) -> bytes:
    try:
        with gzip.open(path, "rb") as f:
            return f.read()
    except (OSError, gzip.BadGzipFile) as exc:
        raise RuntimeError(
            f"failed to read frozen rows payload for {source}:{dataset_sha}"
        ) from exc


def _decode_rows_payload(rows_payload: bytes, *, source: str, dataset_sha: str) -> list[dict[str, Any]]:
    try:
        rows_text = rows_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"rows payload is not utf-8 for {source}:{dataset_sha}"
        ) from exc

    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(rows_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"malformed jsonl row at line {lineno} for {source}:{dataset_sha}"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(
                f"non-dict row at line {lineno} for {source}:{dataset_sha}"
            )
        rows.append(row)
    return rows


def _require_type(
    payload: dict[str, Any],
    key: str,
    expected_type: type,
    *,
    label: str,
) -> Any:
    value = payload.get(key)
    if not isinstance(value, expected_type):
        raise RuntimeError(
            f"{label} field {key!r} must be of type {expected_type.__name__}"
        )
    return value


def _require_non_empty_string(
    payload: dict[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    value = _require_type(payload, key, str, label=label)
    if not value:
        raise RuntimeError(f"{label} field {key!r} must be non-empty")
    return value


def _require_non_negative_int(payload: dict[str, Any], key: str, *, label: str) -> int:
    value = _require_type(payload, key, int, label=label)
    if isinstance(value, bool):
        raise RuntimeError(f"{label} field {key!r} must be a non-negative integer")
    if value < 0:
        raise RuntimeError(f"{label} field {key!r} must be non-negative")
    return value


def _require_list_of_str(payload: dict[str, Any], key: str, *, label: str) -> list[str]:
    values = _require_type(payload, key, list, label=label)
    if not all(isinstance(item, str) for item in values):
        raise RuntimeError(f"{label} field {key!r} must be a list of strings")
    return list(values)


def _validate_identity_payload(
    payload: dict[str, Any],
    *,
    label: str,
    expected_source: str,
    expected_dataset_sha: str,
    expected_dataset_dir: str,
    expected_rows_file: str,
    require_dataset_format: bool = False,
    require_dataset_dir: bool = True,
    require_manifest_version: bool = False,
    require_manifest_content_hash: bool = False,
) -> None:
    source = payload.get("source")
    try:
        source = _validate_source_name(source)
    except ValueError as exc:
        raise RuntimeError(f"{label} source invalid: {source!r}") from exc

    try:
        dataset_sha = _validate_dataset_sha(payload.get("dataset_sha"))
    except ValueError as exc:
        raise RuntimeError(f"{label} dataset_sha invalid") from exc
    rows_file = _require_non_empty_string(payload, "rows_file", label=label)
    if require_dataset_dir:
        dataset_dir = _require_non_empty_string(payload, "dataset_dir", label=label)
        if dataset_dir != expected_dataset_dir:
            raise RuntimeError(
                f"{label} dataset_dir mismatch for {expected_source}:{expected_dataset_sha}"
            )
    _require_non_negative_int(payload, "row_count", label=label)
    _require_list_of_str(payload, "row_hashes", label=label)
    _require_non_empty_string(payload, "row_schema_hash", label=label)
    _require_non_empty_string(payload, "rows_payload_hash", label=label)
    _require_non_empty_string(payload, "metadata_hash", label=label)
    _require_list_of_str(payload, "source_urls", label=label)

    if source != expected_source:
        raise RuntimeError(f"{label} source mismatch for {expected_source}:{expected_dataset_sha}")
    if dataset_sha != expected_dataset_sha:
        raise RuntimeError(f"{label} dataset_sha mismatch for {expected_source}:{expected_dataset_sha}")
    if rows_file != expected_rows_file:
        raise RuntimeError(f"{label} rows_file mismatch for {expected_source}:{expected_dataset_sha}")

    if require_manifest_version:
        manifest_version = _require_non_empty_string(
            payload, "manifest_version", label=label
        )
        if manifest_version != MANIFEST_VERSION:
            raise RuntimeError(
                f"{label} manifest_version must be {MANIFEST_VERSION!r} for "
                f"{expected_source}:{expected_dataset_sha}"
            )

    if require_dataset_format:
        dataset_format = _require_non_empty_string(
            payload, "dataset_format", label=label
        )
        if dataset_format != "jsonl.gz":
            raise RuntimeError(f"{label} dataset_format must be jsonl.gz")

    if require_manifest_content_hash:
        manifest_content_hash = _require_non_empty_string(
            payload, "manifest_content_hash", label=label
        )
        if manifest_content_hash != _manifest_hash(payload):
            raise RuntimeError(f"{label} manifest hash mismatch for {source}:{dataset_sha}")


def write_frozen_source_bundle(
    source: str,
    rows: list[dict[str, Any]],
    *,
    source_urls: list[str],
    cache_root: Path | str | None = None,
) -> dict[str, Any]:
    """
    Persist deterministic rows + manifests for a source snapshot.
    """
    cache_root = _cache_root(cache_root)
    source = _validate_source_name(source)

    rows = [dict(r) for r in _validate_writer_rows(rows)]
    source_urls = _validate_writer_source_urls(source_urls)
    rows = _deterministic_rows(rows)
    rows_bytes = _rows_payload(rows)
    dataset_sha = sha256_hex(rows_bytes)
    dataset_sha = _validate_dataset_sha(dataset_sha, field_name="writer-derived dataset_sha")
    dataset_dir = _dataset_dir(cache_root, source, dataset_sha)
    rows_path = _rows_path(cache_root, dataset_dir)
    metadata_path = _dataset_metadata_path(cache_root, dataset_dir)
    manifest_path = _manifest_path(cache_root, source)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows_file = str(rows_path.relative_to(cache_root))

    row_hashes = sorted(set(_row_hash(row) for row in rows))
    row_schema_hash = _schema_hash(rows)

    rows_payload_hash = sha256_hex(rows_bytes)
    metadata = {
        "source": source,
        "dataset_sha": dataset_sha,
        "dataset_dir": str(dataset_dir.relative_to(cache_root)),
        "row_count": len(rows),
        "row_hashes": row_hashes,
        "row_schema_hash": row_schema_hash,
        "rows_payload_hash": rows_payload_hash,
        "source_urls": sorted(set(source_urls)),
        "dataset_format": "jsonl.gz",
        "rows_file": str(rows_path.relative_to(cache_root)),
        "rows_file_hash": rows_payload_hash,
    }
    metadata["metadata_hash"] = _metadata_hash(metadata)

    manifest = {
        "source": source,
        "dataset_sha": dataset_sha,
        "dataset_dir": str(dataset_dir.relative_to(cache_root)),
        "rows_file": rows_file,
        "row_count": len(rows),
        "source_urls": sorted(set(source_urls)),
        "manifest_version": "1",
        "dataset_format": "jsonl.gz",
        "row_schema_hash": row_schema_hash,
        "row_hashes": row_hashes,
        "rows_payload_hash": rows_payload_hash,
        "metadata_hash": metadata["metadata_hash"],
        "fetch_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    manifest["manifest_content_hash"] = _manifest_hash(manifest)

    _write_gzip_atomic(rows_path, rows_bytes)
    _write_text_atomic(metadata_path, json.dumps(metadata, sort_keys=True, indent=2))
    _write_text_atomic(manifest_path, json.dumps(manifest, sort_keys=True, separators=(",", ":")))

    return {
        "manifest_path": str(manifest_path),
        "dataset_dir": str(dataset_dir),
        "dataset_sha": dataset_sha,
    }


def load_frozen_source_manifest(source: str, *, cache_root: Path | str | None = None) -> dict[str, Any]:
    cache_root = _cache_root(cache_root)
    try:
        source = _validate_source_name(source)
    except ValueError as exc:
        raise RuntimeError(f"invalid source for frozen source manifest: {source!r}") from exc
    manifest_path = _manifest_path(cache_root, source)
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest for source {source}")
    payload = _load_json_object(manifest_path, label="manifest")
    try:
        manifest_source = _validate_source_name(payload.get("source"))
    except ValueError as exc:
        raise RuntimeError(f"manifest source invalid for {source}") from exc
    if manifest_source != source:
        raise RuntimeError(f"manifest source mismatch for {source}")
    try:
        manifest_dataset_sha = _validate_dataset_sha(payload.get("dataset_sha"))
    except ValueError as exc:
        raise RuntimeError(f"manifest dataset_sha invalid for {source}") from exc

    expected_dataset_dir = _expected_dataset_dir(cache_root, source, manifest_dataset_sha)
    expected_rows_file = _expected_rows_file(cache_root, source, manifest_dataset_sha)
    _validate_identity_payload(
        payload,
        label="manifest",
        expected_source=manifest_source,
        expected_dataset_sha=manifest_dataset_sha,
        expected_dataset_dir=expected_dataset_dir,
        expected_rows_file=expected_rows_file,
        require_dataset_format=True,
        require_dataset_dir=True,
        require_manifest_version=True,
        require_manifest_content_hash=True,
    )
    return payload


def load_frozen_source_rows(
    source: str,
    dataset_sha: str | None = None,
    *,
    cache_root: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_root = _cache_root(cache_root)
    try:
        source = _validate_source_name(source)
    except ValueError as exc:
        raise RuntimeError(f"invalid source for frozen source rows: {source!r}") from exc
    manifest = load_frozen_source_manifest(source, cache_root=cache_root)
    if not isinstance(manifest, dict):
        raise RuntimeError("manifest payload must be a JSON object")
    requested_sha = None
    if dataset_sha is not None:
        try:
            requested_sha = _validate_dataset_sha(dataset_sha, field_name="requested dataset_sha")
        except ValueError as exc:
            raise RuntimeError(f"requested dataset_sha invalid for {source}") from exc
    manifest_sha = manifest.get("dataset_sha")

    resolved_sha = requested_sha or manifest_sha
    expected_dataset_dir = _expected_dataset_dir(cache_root, source, resolved_sha)
    expected_rows_file = _expected_rows_file(cache_root, source, resolved_sha)

    _validate_identity_payload(
        manifest,
        label="manifest",
        expected_source=source,
        expected_dataset_sha=resolved_sha,
        expected_dataset_dir=expected_dataset_dir,
        expected_rows_file=expected_rows_file,
        require_dataset_format=False,
    )

    dataset_dir = _dataset_dir(cache_root, source, resolved_sha)
    metadata_path = _dataset_metadata_path(cache_root, dataset_dir)
    rows_path = _rows_path(cache_root, dataset_dir)
    if not dataset_dir.exists() or not rows_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"incomplete frozen cache for {source}:{resolved_sha}")

    metadata = _load_json_object(metadata_path, label="metadata")
    _validate_identity_payload(
        metadata,
        label="metadata",
        expected_source=source,
        expected_dataset_sha=resolved_sha,
        expected_dataset_dir=expected_dataset_dir,
        expected_rows_file=expected_rows_file,
        require_dataset_format=True,
        require_dataset_dir=True,
    )

    expected_metadata_hash = _metadata_hash(metadata)
    if metadata.get("metadata_hash") != expected_metadata_hash:
        raise RuntimeError(f"metadata hash mismatch for {source}:{resolved_sha}")

    rows_payload = _read_rows_payload(
        rows_path,
        source=source,
        dataset_sha=resolved_sha,
    )
    payload_hash = sha256_hex(rows_payload)
    if manifest.get("rows_payload_hash") != payload_hash:
        raise RuntimeError(f"rows payload mismatch for {source}:{resolved_sha}")
    if metadata.get("rows_payload_hash") != payload_hash:
        raise RuntimeError(f"metadata rows payload hash mismatch for {source}:{resolved_sha}")

    rows = _decode_rows_payload(
        rows_payload,
        source=source,
        dataset_sha=resolved_sha,
    )
    observed_row_hashes = sorted(set(_row_hash(row) for row in rows))

    if metadata.get("row_count") != len(rows):
        raise RuntimeError("row count changed for frozen manifest")
    if manifest.get("row_count") != len(rows):
        raise RuntimeError("row count changed for frozen manifest")
    if manifest.get("row_hashes") != observed_row_hashes:
        raise RuntimeError("row hash set changed for frozen manifest")
    if metadata.get("row_hashes") != observed_row_hashes:
        raise RuntimeError("metadata row hash set changed for frozen manifest")
    if metadata.get("source") != source:
        raise RuntimeError(f"metadata source mismatch for {source}:{resolved_sha}")
    if metadata.get("dataset_sha") != resolved_sha:
        raise RuntimeError(f"metadata dataset sha mismatch for {source}:{resolved_sha}")

    observed_schema_hash = _schema_hash(_deterministic_rows(rows))
    if manifest.get("row_schema_hash") != observed_schema_hash:
        raise RuntimeError("row schema hash mismatch for frozen manifest")
    if metadata.get("row_schema_hash") != observed_schema_hash:
        raise RuntimeError("metadata row schema hash mismatch for frozen manifest")

    return rows, metadata


def assert_frozen_cache_match(
    source: str,
    dataset_sha: str,
    *,
    cache_root: Path | str | None = None,
) -> None:
    """
    Enforce strict equality between requested and cached source manifest/rows.
    """
    _, _ = load_frozen_source_rows(
        source,
        dataset_sha=dataset_sha,
        cache_root=cache_root,
    )
