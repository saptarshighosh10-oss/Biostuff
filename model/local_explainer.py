"""Optional local Ollama wording/checking for selected prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


def _generate(model: str, prompt: str, timeout: int = 120) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read()).get("response", "").strip()


def explain_selected(input_file: str, output_file: str, top_n: int = 1000,
                     model: str = "qwen3:1.7b", checker: str = "llama3.2:3b") -> dict:
    """Rewrite only high-risk/uncertain rows; preserve deterministic evidence."""
    payload = json.loads(Path(input_file).read_text(encoding="utf-8"))
    results = payload.get("results", payload if isinstance(payload, list) else [])
    selected = [
        row for row in results
        if row.get("failure_probability", 0.0) >= 0.7
        or row.get("confidence") == "low"
    ][:top_n]

    for row in selected:
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

    output = {"schema_version": "1", "source": input_file, "results": results}
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return {"rows": len(results), "selected": len(selected), "output": output_file}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add bounded local-model wording to predictions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--checker", default="llama3.2:3b")
    args = parser.parse_args()
    print(explain_selected(args.input, args.output, args.top, args.model, args.checker))
