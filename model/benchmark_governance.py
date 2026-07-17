"""Traceable benchmark policy, evidence snapshots, and holdout-use records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
