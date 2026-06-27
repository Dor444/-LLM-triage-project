"""Fine-tune DistilBERT for multi-label risk-factor classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_LENGTH,
    DEFAULT_RISK_THRESHOLD,
    RANDOM_SEED,
    RISK_FACTOR_LABELS,
    SPLIT_FILENAMES,
)
from data_utils import load_labeled_jsonl, risk_multihot, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing train/validation/test JSONL files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save model and metrics.")
    parser.add_argument("--model-name", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_RISK_THRESHOLD)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_split(data_dir: str | Path, split: str) -> list[dict]:
    return load_labeled_jsonl(Path(data_dir) / SPLIT_FILENAMES[split])


def make_dataset(records: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "text": record["message"],
                "labels": risk_multihot(record.get("risk_factors", [])),
            }
            for record in records
        ]
    )


def build_training_args(args: argparse.Namespace):
    from transformers import TrainingArguments

    common = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "micro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
    }
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def make_compute_metrics(threshold: float):
    def compute_metrics(eval_pred) -> dict[str, float]:
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score

        logits, labels = eval_pred
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        predictions = (probabilities >= threshold).astype(int)
        labels = labels.astype(int)
        return {
            "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "micro_precision": float(precision_score(labels, predictions, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(labels, predictions, average="micro", zero_division=0)),
            "exact_match": float((predictions == labels).all(axis=1).mean()),
        }

    return compute_metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

    label2id = {label: idx for idx, label in enumerate(RISK_FACTOR_LABELS)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(RISK_FACTOR_LABELS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
        local_files_only=args.local_files_only,
    )

    train_ds = make_dataset(load_split(args.data_dir, "train"))
    validation_ds = make_dataset(load_split(args.data_dir, "validation"))
    test_ds = make_dataset(load_split(args.data_dir, "test"))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    validation_ds = validation_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=make_compute_metrics(args.threshold),
    )
    trainer.train()
    validation_metrics = trainer.evaluate(validation_ds, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(
        {
            "task": "risk_factor_multilabel_classification",
            "model_name": args.model_name,
            "threshold": args.threshold,
            "risk_factor_labels": RISK_FACTOR_LABELS,
            "validation": validation_metrics,
            "test": test_metrics,
        },
        Path(args.output_dir) / "metrics.json",
    )
    print({"validation": validation_metrics, "test": test_metrics})


if __name__ == "__main__":
    main()
