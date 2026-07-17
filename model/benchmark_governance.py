"""Traceable benchmark policy, evidence snapshots, and holdout-use records."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

from data.run_manifest import file_sha256


SCHEMA_VERSION = "1"
PROPERTIES = ("polyreactivity", "aggregation", "hydrophobicity")
ROLES = ("development", "locked_validation", "final")


class PolicyError(ValueError):
    """Raised when benchmark governance would be ambiguous or unsafe."""


def load_policy(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyError(f"policy file not found: {candidate}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"policy is not valid JSON: {candidate}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PolicyError("policy root must be a JSON object")
    return payload


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise PolicyError(f"unsupported schema_version: {policy.get('schema_version')!r}")
    if set(policy.get("properties", {})) != set(PROPERTIES):
        raise PolicyError(f"properties must be exactly: {', '.join(PROPERTIES)}")
    if not isinstance(policy.get("metric_definitions"), dict):
        raise PolicyError("metric_definitions must be an object")
    if not isinstance(policy.get("frozen_artifacts"), list):
        raise PolicyError("frozen_artifacts must be a list")

    for property_name, property_policy in policy["properties"].items():
        roles = property_policy.get("source_roles", {})
        if set(roles) != set(ROLES):
            raise PolicyError(f"{property_name}: source_roles must be exactly {ROLES}")
        seen: set[str] = set()
        for role in ROLES:
            entries = roles[role]
            if not isinstance(entries, list):
                raise PolicyError(f"{property_name}.{role} must be a list")
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    raise PolicyError(f"{property_name}.{role} has an invalid source entry")
                if not isinstance(entry.get("prior_use"), bool):
                    raise PolicyError(f"{property_name}.{role}.{entry.get('name')} needs boolean prior_use")
                source = entry["name"]
                if source in seen:
                    raise PolicyError(f"{property_name}: {source} appears in multiple roles")
                seen.add(source)
                if role == "final" and entry["prior_use"]:
                    raise PolicyError(f"{property_name}: final source {source} must remain unspent")
        if not isinstance(property_policy.get("baselines"), list):
            raise PolicyError(f"{property_name}.baselines must be a list")
        if not isinstance(property_policy.get("count_evidence"), list):
            raise PolicyError(f"{property_name}.count_evidence must be a list")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(repo_root: str | Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    return candidate if candidate.is_absolute() else Path(repo_root) / candidate


def read_json_pointer(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise PolicyError(f"invalid JSON pointer: {pointer!r}")
    current = payload
    try:
        for raw_part in pointer[1:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            current = current[int(part)] if isinstance(current, list) else current[part]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PolicyError(f"JSON pointer not found: {pointer}") from exc
    return current


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyError(f"evidence file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"evidence is not valid JSON: {path}: {exc}") from exc


def _csv_group_count(path: Path, column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise PolicyError(f"CSV {path} is missing column {column!r}")
            for row in reader:
                value = (row.get(column) or "").strip()
                if value:
                    counts[value] = counts.get(value, 0) + 1
    except FileNotFoundError as exc:
        raise PolicyError(f"count evidence file not found: {path}") from exc
    return dict(sorted(counts.items()))


def _inspect_count_spec(spec: dict[str, Any], repo_root: Path) -> Any:
    path = resolve_path(repo_root, spec["path"])
    kind = spec["kind"]
    if kind == "csv_group_count":
        return _csv_group_count(path, spec["column"])
    payload = _load_json(path)
    value = read_json_pointer(payload, spec.get("json_pointer", ""))
    if kind == "json_value":
        return value
    if kind == "object_field_map":
        if not isinstance(value, dict):
            raise PolicyError(f"{spec['name']}: pointer must resolve to an object")
        field = spec["field"]
        try:
            return {name: record[field] for name, record in value.items()}
        except (KeyError, TypeError) as exc:
            raise PolicyError(f"{spec['name']}: object records must contain {field!r}") from exc
    if kind == "json_records":
        if not isinstance(value, list):
            raise PolicyError(f"{spec['name']}: pointer must resolve to a list")
        fields = spec["fields"]
        try:
            return [{field: record[field] for field in fields} for record in value]
        except (KeyError, TypeError) as exc:
            raise PolicyError(f"{spec['name']}: list records are missing requested fields") from exc
    raise PolicyError(f"unsupported count evidence kind: {kind!r}")


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    return [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_status(
    policy: dict[str, Any],
    *,
    repo_root: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    validate_policy(policy)
    root = Path(repo_root).resolve()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "metric_definitions": policy["metric_definitions"],
        "properties": {},
        "frozen_artifacts": [],
        "holdout_uses": read_ledger(ledger_path),
    }
    for property_name, property_policy in policy["properties"].items():
        property_report = {
            "source_roles": property_policy["source_roles"],
            "baselines": [],
            "counts": [],
        }
        for baseline in property_policy["baselines"]:
            item = dict(baseline)
            path = resolve_path(root, baseline["path"])
            item["evidence_sha256"] = file_sha256(path)
            try:
                item["value"] = read_json_pointer(_load_json(path), baseline["json_pointer"])
                item["status"] = "ok"
            except (KeyError, PolicyError) as exc:
                item.update({"value": None, "status": "missing", "error": str(exc)})
            property_report["baselines"].append(item)
        for count_spec in property_policy["count_evidence"]:
            item = {"name": count_spec["name"], "kind": count_spec["kind"]}
            try:
                item.update({"value": _inspect_count_spec(count_spec, root), "status": "ok"})
            except (KeyError, PolicyError) as exc:
                item.update({"value": None, "status": "missing", "error": str(exc)})
            property_report["counts"].append(item)
        report["properties"][property_name] = property_report
    for artifact in policy["frozen_artifacts"]:
        path = resolve_path(root, artifact["path"])
        report["frozen_artifacts"].append(
            {
                **artifact,
                "resolved_path": str(path),
                "present": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": file_sha256(path),
            }
        )
    return report
