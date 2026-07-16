"""GPU Colab job: fine-tune ESM-2 on library PSR, evaluate frozen diverse holdouts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LIBRARIES = {"emi", "igg", "iso", "tessier_lib1", "tessier_lib2"}
HOLDOUTS = ("aims_mouse", "aims_flu", "shehata2019")


def _number(row: dict) -> float | None:
    try:
        return float(row["label"])
    except (KeyError, TypeError, ValueError):
        return None


def prepare_rows(csv_path: Path, max_per_class: int, seed: int) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """Prepare a strict, VH-deduplicated library training set and untouched holdouts."""
    rows = [row for row in csv.DictReader(csv_path.open(encoding="utf-8")) if (row.get("vh") or "").strip()]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["vh"] = row["vh"].strip().upper()
        by_source[row["source"]].append(row)

    labels: dict[int, int | None] = {}
    for source, source_rows in by_source.items():
        values = [_number(row) for row in source_rows]
        threshold = None
        if source == "shehata2019":
            numeric = sorted(value for value in values if value is not None)
            threshold = float(np.quantile(numeric, 0.85)) if numeric else None
        elif source == "aims_flu":
            numeric = sorted(value for value in values if value is not None)
            threshold = float(np.median(numeric)) if numeric else None
        for row, value in zip(source_rows, values):
            labels[id(row)] = int(value > threshold) if threshold is not None and value is not None else (
                int(value) if value in (0.0, 1.0) else None)

    holdout_rows = {source: [row for row in by_source.get(source, []) if labels[id(row)] is not None]
                    for source in HOLDOUTS}
    held_vh = {row["vh"] for rows in holdout_rows.values() for row in rows}
    raw_train = [row for source in LIBRARIES for row in by_source.get(source, []) if labels[id(row)] is not None]
    strict_train = [row for row in raw_train if row["vh"] not in held_vh]
    by_vh: dict[str, list[dict]] = defaultdict(list)
    for row in strict_train:
        by_vh[row["vh"]].append(row)
    conflicts = sum(len(group) for group in by_vh.values() if len({labels[id(row)] for row in group}) > 1)
    unique_train = [group[0] for group in by_vh.values() if len({labels[id(row)] for row in group}) == 1]

    rng = random.Random(seed)
    rng.shuffle(unique_train)
    positives = [row for row in unique_train if labels[id(row)] == 1][:max_per_class]
    negatives = [row for row in unique_train if labels[id(row)] == 0][:max_per_class]
    train = positives + negatives
    rng.shuffle(train)
    for row in train:
        row["target"] = labels[id(row)]
    for rows in holdout_rows.values():
        for row in rows:
            row["target"] = labels[id(row)]

    counts = lambda rows: {source: dict(Counter(row["target"] for row in rows if row["source"] == source))
                           for source in sorted(LIBRARIES)}
    report = {"raw_library_rows": len(raw_train), "shared_holdout_rows_removed": len(raw_train) - len(strict_train),
              "conflicting_label_rows_removed": conflicts, "unique_train_before_cap": len(unique_train),
              "train_rows": len(train), "train_by_source_class": counts(train),
              "holdout_rows": {source: len(rows) for source, rows in holdout_rows.items()}}
    return train, holdout_rows, report


def run(args: argparse.Namespace) -> dict:
    train, holdouts, report = prepare_rows(Path(args.csv), args.max_per_class, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.preflight_only:
        return report

    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    class SequenceDataset(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            encoded = tokenizer(row["vh"][:args.max_length], truncation=True, max_length=args.max_length)
            encoded["labels"] = row["target"]
            return encoded

    training = TrainingArguments("plm_finetune_out", per_device_train_batch_size=args.batch_size,
                                 num_train_epochs=args.epochs, learning_rate=args.learning_rate,
                                 fp16=torch.cuda.is_available(), logging_steps=200,
                                 report_to="none", save_strategy="no", seed=args.seed)
    Trainer(model, training, train_dataset=SequenceDataset(train), data_collator=DataCollatorWithPadding(tokenizer)).train()
    model.eval()
    aucs = {}
    for source, rows in holdouts.items():
        scores = []
        for start in range(0, len(rows), args.eval_batch_size):
            batch = rows[start:start + args.eval_batch_size]
            encoded = tokenizer([row["vh"][:args.max_length] for row in batch], padding=True, truncation=True,
                                max_length=args.max_length, return_tensors="pt").to(model.device)
            with torch.no_grad():
                scores.extend(torch.softmax(model(**encoded).logits, -1)[:, 1].cpu().tolist())
        aucs[source] = roc_auc_score([row["target"] for row in rows], scores)
        print(f"{source}: n={len(rows)} AUC={aucs[source]:.3f}")
    report["heldout_auc"] = aucs
    report["mean_diverse_auc"] = sum(aucs.values()) / len(aucs)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="polyreactivity_dataset.csv")
    parser.add_argument("--out", default="plm_finetune_report.json")
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--max-per-class", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments)
    Path(arguments.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
