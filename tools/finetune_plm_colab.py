"""Manifest-driven Colab PLM adaptation with fixed diverse holdouts.

The runner compares library-only, diverse-only, or mixed training without ever
training on AIMS mouse, AIMS flu, or Shehata. It supports frozen encoders,
LoRA adapters, and full fine-tuning, exports every held-out prediction, and can
run seeds 42/43/44 from one command. ``--preflight-only`` is safe on a Mac.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from data.contract import hash_payload
from data.run_manifest import file_sha256


LIBRARIES = {"emi", "igg", "iso", "tessier_lib1", "tessier_lib2"}
HOLDOUTS = ("aims_mouse", "aims_flu", "shehata2019")
ARMS = ("library_only", "diverse_only", "mixed")
MODES = ("frozen", "adapter", "full")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _number(row: dict) -> float | None:
    try:
        value = float(row["label"])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def _pair_key(row: dict) -> tuple[str, str]:
    return str(row.get("vh") or ""), str(row.get("vl") or "")


def _valid_sequence(sequence: str, *, allow_empty: bool = False) -> bool:
    return (allow_empty and not sequence) or (len(sequence) >= 20 and set(sequence) <= STANDARD_AA)


def _framework_cluster(row: dict) -> str:
    """Dependency-free scaffold proxy used only for sampling control."""
    vh, vl = _pair_key(row)
    return hash_payload({
        "vh_framework_proxy": vh[:30] + vh[45:75] + vh[-20:],
        "vl_framework_proxy": vl[:25] + vl[40:70] + vl[-15:] if vl else "missing_vl",
    })[:20]


def _assign_targets(by_source: dict[str, list[dict]]) -> None:
    """Apply source-native binary labels or an exact top-15% continuous tail."""
    for source, rows in by_source.items():
        numeric = [(row, _number(row)) for row in rows]
        valid = [(row, value) for row, value in numeric if value is not None]
        values = {float(value) for _, value in valid}
        if source in {"aims_flu"} or {row.get("label_type") for row, _ in valid} == {"num_react"}:
            for row, value in valid:
                row["target"] = int(float(value) >= 2.0)
        elif values and values <= {0.0, 1.0}:
            for row, value in valid:
                row["target"] = int(float(value))
        else:
            count = max(1, math.ceil(len(valid) * 0.15))
            ordered = sorted(valid, key=lambda item: (float(item[1]), _pair_key(item[0])), reverse=True)
            bad = {id(row) for row, _ in ordered[:count]}
            for row, _ in valid:
                row["target"] = int(id(row) in bad)


def _training_sources(all_sources: set[str], arm: str) -> set[str]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    development_diverse = all_sources - LIBRARIES - set(HOLDOUTS)
    if arm == "library_only":
        return all_sources & LIBRARIES
    if arm == "diverse_only":
        return development_diverse
    return (all_sources & LIBRARIES) | development_diverse


def _balanced_sample(
    rows: list[dict], *, max_per_source_class: int, max_per_cluster_class: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    by_cluster: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cluster[(row["source"], row["target"], _framework_cluster(row))].append(row)
    cluster_capped = []
    for group in by_cluster.values():
        group = sorted(group, key=_pair_key)
        cluster_capped.extend(
            rng.sample(group, max_per_cluster_class)
            if len(group) > max_per_cluster_class else group
        )
    by_source_class: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in cluster_capped:
        by_source_class[(row["source"], row["target"])].append(row)
    selected = []
    for group in by_source_class.values():
        group = sorted(group, key=_pair_key)
        selected.extend(
            rng.sample(group, max_per_source_class)
            if len(group) > max_per_source_class else group
        )
    rng.shuffle(selected)
    return selected


def prepare_rows(
    csv_path: Path,
    max_per_class: int,
    seed: int,
    *,
    arm: str = "library_only",
    max_per_cluster_class: int = 250,
) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """Prepare source/cluster-balanced training and untouched diverse holdouts."""
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    by_source: dict[str, list[dict]] = defaultdict(list)
    rejected = 0
    for raw in raw_rows:
        row = dict(raw)
        row["vh"] = str(row.get("vh") or "").strip().upper()
        row["vl"] = str(row.get("vl") or "").strip().upper()
        if (not _valid_sequence(row["vh"])
                or not _valid_sequence(row["vl"], allow_empty=True)
                or _number(row) is None):
            rejected += 1
            continue
        by_source[str(row.get("source") or "unknown")].append(row)
    _assign_targets(by_source)

    holdouts = {source: list(by_source.get(source, [])) for source in HOLDOUTS}
    held_vh = {row["vh"] for rows in holdouts.values() for row in rows}
    held_vl = {row["vl"] for rows in holdouts.values() for row in rows if row["vl"]}
    allowed_sources = _training_sources(set(by_source), arm)
    raw_train = [row for source in allowed_sources for row in by_source[source]]
    strict_train = [row for row in raw_train
                    if row["vh"] not in held_vh and (not row["vl"] or row["vl"] not in held_vl)]

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in strict_train:
        by_pair[_pair_key(row)].append(row)
    conflicts = sum(len(group) for group in by_pair.values()
                    if len({row["target"] for row in group}) > 1)
    unique_train = [group[0] for group in by_pair.values()
                    if len({row["target"] for row in group}) == 1]
    train = _balanced_sample(
        unique_train,
        max_per_source_class=max_per_class,
        max_per_cluster_class=max_per_cluster_class,
        seed=seed,
    )

    source_class_counts = {
        source: dict(sorted(Counter(row["target"] for row in train if row["source"] == source).items()))
        for source in sorted({row["source"] for row in train})
    }
    report = {
        "input_rows": len(raw_rows),
        "rejected_rows": rejected,
        "arm": arm,
        "training_sources": sorted(allowed_sources),
        "raw_training_rows": len(raw_train),
        "shared_holdout_rows_removed": len(raw_train) - len(strict_train),
        "conflicting_label_rows_removed": conflicts,
        "unique_train_before_caps": len(unique_train),
        "train_rows": len(train),
        "train_by_source_class": source_class_counts,
        "independent_framework_proxy_clusters": len({_framework_cluster(row) for row in train}),
        "holdout_rows": {source: len(rows) for source, rows in holdouts.items()},
        "holdout_positives": {source: sum(row["target"] for row in rows) for source, rows in holdouts.items()},
        "aims_flu_is_tiny": len(holdouts["aims_flu"]) == 97,
        "chain_boundary": "tokenizer text_pair: VH [model separator] VL; missing VL remains single-chain",
    }
    return train, holdouts, report


def _configure_model(model, mode: str):
    if mode == "frozen":
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
        return model
    if mode == "adapter":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "adapter mode requires PEFT. In Colab run: pip install peft"
            ) from exc
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )
        return get_peft_model(model, config)
    if mode == "full":
        return model
    raise ValueError(f"mode must be one of {MODES}")


def _tokenize(tokenizer, row: dict, max_length: int, **kwargs):
    light = row.get("vl") or None
    return tokenizer(
        row["vh"],
        text_pair=light,
        truncation=True,
        max_length=max_length,
        **kwargs,
    )


def run_one(args: argparse.Namespace, seed: int) -> dict:
    train, holdouts, report = prepare_rows(
        Path(args.csv),
        args.max_per_class,
        seed,
        arm=args.arm,
        max_per_cluster_class=args.max_per_cluster_class,
    )
    report.update({"seed": seed, "mode": args.mode, "model": args.model})
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.preflight_only:
        return report
    if not train:
        raise RuntimeError(f"training arm {args.arm!r} produced zero usable rows")

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model = _configure_model(model, args.mode)

    class SequenceDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            encoded = _tokenize(tokenizer, row, args.max_length)
            encoded["labels"] = row["target"]
            return encoded

    run_dir = Path(args.work_dir) / f"{args.arm}-{args.mode}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model,
        training,
        train_dataset=SequenceDataset(train),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    resume = args.resume_from_checkpoint or (True if args.resume else None)
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(run_dir / "model")
    tokenizer.save_pretrained(run_dir / "model")
    model.eval()

    prediction_path = run_dir / "heldout_predictions.csv"
    aucs: dict[str, float] = {}
    prs: dict[str, float] = {}
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "vh", "vl", "label", "score"))
        writer.writeheader()
        for source, rows in holdouts.items():
            labels = [row["target"] for row in rows]
            if len(set(labels)) != 2:
                report.setdefault("skipped_holdouts", {})[source] = "only one label class"
                continue
            scores = []
            for start in range(0, len(rows), args.eval_batch_size):
                batch = rows[start:start + args.eval_batch_size]
                encoded = tokenizer(
                    [row["vh"] for row in batch],
                    text_pair=[row["vl"] or None for row in batch],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    scores.extend(torch.softmax(model(**encoded).logits, -1)[:, 1].cpu().tolist())
            aucs[source] = float(roc_auc_score(labels, scores))
            prs[source] = float(average_precision_score(labels, scores))
            for row, score in zip(rows, scores):
                writer.writerow({"source": source, "vh": row["vh"], "vl": row["vl"],
                                 "label": row["target"], "score": score})
            print(f"{source}: n={len(rows)} ROC_AUC={aucs[source]:.3f} PR_AUC={prs[source]:.3f}")
    report.update({
        "heldout_auc": aucs,
        "heldout_pr_auc": prs,
        "mean_diverse_auc": fmean(aucs.values()) if aucs else None,
        "predictions": str(prediction_path),
        "predictions_sha256": file_sha256(prediction_path),
        "artifact_dir": str(run_dir),
    })
    (run_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _resolved_manifest(args: argparse.Namespace, seeds: list[int]) -> dict:
    return {
        "schema_version": "1",
        "data": str(Path(args.csv).resolve()),
        "data_sha256": file_sha256(args.csv),
        "model": args.model,
        "arm": args.arm,
        "mode": args.mode,
        "seeds": seeds,
        "holdouts": list(HOLDOUTS),
        "max_per_source_class": args.max_per_class,
        "max_per_framework_proxy_cluster_class": args.max_per_cluster_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "chain_boundary": "tokenizer text_pair",
    }


def run(args: argparse.Namespace) -> dict:
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        seeds = [args.seed]
    manifest = _resolved_manifest(args, seeds)
    if manifest["data_sha256"] is None:
        raise FileNotFoundError(args.csv)
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runs = [run_one(args, seed) for seed in seeds]
    source_seed_auc: dict[str, list[float]] = defaultdict(list)
    for result in runs:
        for source, auc in result.get("heldout_auc", {}).items():
            source_seed_auc[source].append(float(auc))
    return {
        "schema_version": "1",
        "manifest": manifest,
        "manifest_sha256": file_sha256(manifest_path),
        "runs": runs,
        "mean_auc_by_source": {source: fmean(values) for source, values in source_seed_auc.items()},
        "three_seed_macro_auc": (
            fmean(fmean(values) for values in source_seed_auc.values()) if source_seed_auc else None
        ),
        "development_only": True,
        "gdpa3_touched": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="polyreactivity_dataset.csv")
    parser.add_argument("--out", default="plm_finetune_report.json")
    parser.add_argument("--manifest-out", default="plm_finetune_manifest.json")
    parser.add_argument("--work-dir", default="plm_finetune_work")
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--arm", choices=ARMS, default="library_only")
    parser.add_argument("--mode", choices=MODES, default="frozen")
    parser.add_argument("--max-per-class", type=int, default=40_000)
    parser.add_argument("--max-per-cluster-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42, help="legacy single-seed fallback")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


if __name__ == "__main__":
    # Colab's kernel launcher injects ``-f <kernel.json>`` into pasted scripts.
    arguments, _unknown = build_parser().parse_known_args()
    result = run(arguments)
    Path(arguments.out).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
