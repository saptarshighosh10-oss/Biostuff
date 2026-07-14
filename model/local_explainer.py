"""Bounded, resumable local-model wording/checking for prediction artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib import request


def _generate(model: str, prompt: str, timeout: int = 180) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
    }).encode()
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read()).get("response", "").strip()


def _tagged_lines(response: str) -> dict[str, str]:
    values = {}
    for line in response.splitlines():
        line = line.strip().strip("`")
        match = re.match(r"^(r\d+)\s*[|:\t]\s*(.+)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _description_batch(rows: list[dict], model: str) -> dict[str, str]:
    evidence = "\n".join(
        f"r{i}: {json.dumps(row.get('failure_explanation', {}), sort_keys=True)}"
        for i, row in enumerate(rows)
    )
    response = _generate(
        model,
        "Rewrite each evidence record as one short sentence. Do not add facts. "
        "Preserve uncertainty. Return exactly one line per ID as ID|sentence.\n" + evidence,
    )
    return _tagged_lines(response)


def _check_batch(rows: list[dict], model: str) -> dict[str, str]:
    evidence = "\n".join(
        f"r{i}: evidence={json.dumps(row.get('failure_explanation', {}), sort_keys=True)} "
        f"description={row.get('local_description', '')}"
        for i, row in enumerate(rows)
    )
    response = _generate(
        model,
        "Check each description against its evidence. Return exactly one line per ID as "
        "ID|PASS or ID|FAIL. FAIL means it adds an unsupported claim.\n" + evidence,
    )
    return _tagged_lines(response)


def _single_description(row: dict, model: str) -> str:
    return _generate(
        model,
        "Write one short sentence from this verified evidence only. Do not add facts. "
        + json.dumps(row.get("failure_explanation", {}), sort_keys=True),
    )


def _single_check(row: dict, model: str) -> str:
    response = _generate(
        model,
        "Reply only PASS or FAIL. FAIL means this description adds an unsupported claim. "
        + json.dumps({
            "evidence": row.get("failure_explanation", {}),
            "description": row.get("local_description", ""),
        }, sort_keys=True),
    ).upper()
    match = re.search(r"\b(PASS|FAIL)\b", response)
    return match.group(1) if match else "UNPARSED"


def _write_output(output_file: str, input_file: str, results: list[dict]) -> None:
    payload = {"schema_version": "1", "source": input_file, "results": results}
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def explain_selected(input_file: str, output_file: str, top_n: int = 1000,
                     model: str = "qwen3:1.7b", checker: str = "llama3.2:3b",
                     all_rows: bool = False, checkpoint_every: int = 50,
                     batch_size: int = 32) -> dict:
    """Add synthetic wording only; deterministic evidence remains authoritative."""
    payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
    results = payload.get("results", payload if isinstance(payload, list) else [])
    existing = Path(output_file)
    if existing.exists():
        saved = json.loads(existing.read_text(encoding="utf-8"))
        if len(saved.get("results", [])) == len(results):
            results = saved["results"]

    selected = results if all_rows else [
        row for row in results
        if row.get("failure_probability", 0.0) >= 0.7
        or row.get("confidence") == "low"
    ][:top_n]

    batches = 0
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        if all(row.get("local_description") and row.get("local_description_check") for row in batch):
            continue
        descriptions = _description_batch(batch, model)
        for i, row in enumerate(batch):
            row["local_description"] = descriptions.get(f"r{i}") or _single_description(row, model)
            row["local_description_source"] = "local_ollama_synthetic"
        checks = _check_batch(batch, checker)
        for i, row in enumerate(batch):
            row["local_description_check"] = checks.get(f"r{i}") or _single_check(row, checker)
        batches += 1
        if batches % checkpoint_every == 0:
            _write_output(output_file, input_file, results)

    _write_output(output_file, input_file, results)
    return {"rows": len(results), "selected": len(selected), "batches": batches, "output": output_file}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add bounded local-model wording to predictions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--checker", default="llama3.2:3b")
    parser.add_argument("--all", action="store_true", help="Generate/check wording for every row")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(explain_selected(args.input, args.output, args.top, args.model, args.checker,
                           args.all, args.checkpoint_every, args.batch_size))
