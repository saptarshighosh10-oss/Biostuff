"""Optional local Ollama wording/checking for selected prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


def _generate(model: str, prompt: str, timeout: int = 120) -> str:
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


def explain_selected(input_file: str, output_file: str, top_n: int = 1000,
                     model: str = "qwen3:1.7b", checker: str = "llama3.2:3b",
                     all_rows: bool = False, checkpoint_every: int = 50) -> dict:
    """Rewrite selected rows, or every row, while preserving deterministic evidence."""
    payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
    results = payload.get("results", payload if isinstance(payload, list) else [])
    selected = (results if all_rows else [
        row for row in results
        if row.get("failure_probability", 0.0) >= 0.7
        or row.get("confidence") == "low"
    ][:top_n])

    for index, row in enumerate(selected, 1):
        if row.get("local_description") and row.get("local_description_check"):
            continue
        evidence = row.get("failure_explanation", {})
        prompt = (
            "Rewrite this evidence as two concise sentences. Do not add facts. "
            "Say when the cause is uncertain. Return plain text only.\n\n"
            + json.dumps(evidence, sort_keys=True)
        )
        row["local_description"] = _generate(model, prompt)
        check_prompt = (
            "Check whether this description adds unsupported biological claims. "
            "Reply only PASS or FAIL.\n\n"
            + json.dumps({"evidence": evidence, "description": row["local_description"]})
        )
        row["local_description_check"] = _generate(checker, check_prompt).upper()
        row["local_description_source"] = "local_ollama_synthetic"
        if index % checkpoint_every == 0:
            _write_output(output_file, input_file, results)

    _write_output(output_file, input_file, results)
    return {"rows": len(results), "selected": len(selected), "output": output_file}


def _write_output(output_file: str, input_file: str, results: list[dict]) -> None:
    output = {"schema_version": "1", "source": input_file, "results": results}
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add bounded local-model wording to predictions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--checker", default="llama3.2:3b")
    parser.add_argument("--all", action="store_true", help="Generate/check wording for every row")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    print(explain_selected(args.input, args.output, args.top, args.model, args.checker,
                           args.all, args.checkpoint_every))
