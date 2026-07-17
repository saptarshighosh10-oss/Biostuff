"""Print benchmark status or append a traceable holdout-use record."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from model.benchmark_governance import (
    PROPERTIES,
    PolicyError,
    build_status,
    load_policy,
    record_holdout_use,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "tools/benchmark_policy.json"
DEFAULT_LEDGER = ROOT / "tools/holdout_use_ledger.jsonl"
DEFAULT_OUT = ROOT / "results/benchmark_status.json"


def _format_value(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_status(report: dict[str, Any]) -> str:
    lines = [
        "Antibody benchmark governance status",
        f"generated: {report['generated_at']}",
        f"primary metric: {report['metric_definitions']['primary']}",
        "required metrics: " + ", ".join(report["metric_definitions"]["required"]),
    ]
    for property_name, prop in report["properties"].items():
        lines.extend(["", property_name.upper()])
        for role in ("development", "locked_validation", "final"):
            entries = prop["source_roles"][role]
            label = role.replace("_", " ")
            if not entries:
                lines.append(f"  {label}: NONE ASSIGNED")
                continue
            states = {
                "previously used" if entry.get("used", entry["prior_use"]) else "UNSPENT"
                for entry in entries
            }
            if len(states) == 1:
                names = ", ".join(entry["name"] for entry in entries)
                lines.append(f"  {label} ({states.pop()}): {names}")
            else:
                rendered = ", ".join(
                    f"{entry['name']} "
                    f"({'previously used' if entry.get('used', entry['prior_use']) else 'UNSPENT'})"
                    for entry in entries
                )
                lines.append(f"  {label}: {rendered}")
        lines.append("  traceable baselines:")
        for baseline in prop["baselines"]:
            if baseline["status"] == "ok":
                lines.append(f"    {baseline['name']}: {_format_value(baseline['value'])}")
            else:
                lines.append(f"    {baseline['name']}: MISSING ({baseline['error']})")
        lines.append("  independent counts:")
        for count in prop["counts"]:
            value = count["value"] if count["status"] == "ok" else f"MISSING ({count['error']})"
            lines.append(f"    {count['name']}: {value}")
            if count["name"] == "hard_transfer_test_rows" and isinstance(value, dict):
                small = {source: n for source, n in value.items() if n < 100}
                if small:
                    lines.append(f"    WARNING: small holdout(s), unstable AUC: {small}")
    missing = [item["path"] for item in report["frozen_artifacts"] if not item["present"]]
    lines.extend(
        [
            "",
            f"frozen artifacts: {len(report['frozen_artifacts']) - len(missing)}/{len(report['frozen_artifacts'])} present",
            f"future holdout-use records: {len(report['holdout_uses'])}",
        ]
    )
    if missing:
        lines.append("missing frozen files: " + ", ".join(missing))
    return "\n".join(lines)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    commands = parser.add_subparsers(dest="command")

    status = commands.add_parser("status", help="print status without training")
    status.add_argument("--out", type=Path, default=DEFAULT_OUT)

    record = commands.add_parser("record-use", help="append a traceable holdout-use record")
    record.add_argument("--property", choices=PROPERTIES, required=True)
    record.add_argument("--source", required=True)
    record.add_argument("--experiment-id", required=True)
    record.add_argument("--purpose", required=True)
    record.add_argument("--evidence", type=Path, required=True)
    record.add_argument("--confirm-locked-use", action="store_true")
    record.add_argument("--confirm-final-use", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.command == "record-use":
            record = record_holdout_use(
                policy,
                ledger_path=args.ledger,
                property_name=args.property,
                source=args.source,
                experiment_id=args.experiment_id,
                purpose=args.purpose,
                evidence_path=args.evidence,
                confirm_locked_use=args.confirm_locked_use,
                confirm_final_use=args.confirm_final_use,
            )
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0

        output = args.out if args.command == "status" else DEFAULT_OUT
        report = build_status(policy, repo_root=ROOT, ledger_path=args.ledger)
        _atomic_json_write(output, report)
        print(render_status(report))
        print(f"\nreport: {output}")
        return 0
    except PolicyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
