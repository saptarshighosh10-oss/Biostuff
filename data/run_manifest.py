"""Deterministic, append-safe ledger for end-to-end pipeline runs."""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from data.contract import hash_payload, sha256_hex


SCHEMA_VERSION = "1"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = __import__("hashlib").sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_files(paths: Iterable[str | Path]) -> dict[str, str | None]:
    return {str(path): file_sha256(path) for path in paths}


class RunLedger:
    """Persist stage state so interrupted runs remain explainable and reusable."""

    def __init__(self, path: str | Path, config: dict[str, Any]):
        self.path = Path(path)
        self.config = config
        self.run_id = hash_payload({"schema_version": SCHEMA_VERSION, "config": config})
        self.payload = self._load_or_create()
        self._write()

    def _load_or_create(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, dict) and existing.get("run_id") == self.run_id:
                return existing
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": _utc_now(),
            "config": self.config,
            "stages": {},
        }

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def start(self, stage: str, *, inputs: Iterable[str | Path] = (), outputs: Iterable[str | Path] = ()) -> None:
        self.payload["stages"][stage] = {
            "status": "running",
            "started_at": _utc_now(),
            "inputs": _hash_files(inputs),
            "outputs": list(map(str, outputs)),
        }
        self._write()

    def complete(self, stage: str, *, outputs: Iterable[str | Path] = ()) -> None:
        state = self.payload["stages"].setdefault(stage, {})
        state.update({
            "status": "complete",
            "completed_at": _utc_now(),
            "output_hashes": _hash_files(outputs),
        })
        self._write()

    def skip(self, stage: str, reason: str) -> None:
        self.payload["stages"][stage] = {
            "status": "skipped",
            "reason": reason,
            "recorded_at": _utc_now(),
        }
        self._write()

    def fail(self, stage: str, error: BaseException) -> None:
        state = self.payload["stages"].setdefault(stage, {})
        state.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "failed_at": _utc_now()})
        self._write()

