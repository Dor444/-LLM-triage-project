"""Tune the probability threshold for the multi-label risk-factor model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import DEFAULT_MAX_LENGTH, RISK_FACTOR_LABELS
from data_utils import load_labeled_jsonl, risk_multihot, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Fine-tuned risk model directory.")
    parser.add_argument("--validation-file", required=True, help="Validation JSONL file.")
    parser.add_argument("--output", required=True, help="Path for threshold tuning JSON.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70],
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def predict_probabilities(args: argparse.Namespace, messages: list[str]):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batches = []
    for start in range(0, len(messages), args.batch_size):
        batch_messages = messages[start : start + args.batch_size]
        encoded = tokenizer(
            batch_messages,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu().numpy()
        batches.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(batches, axis=0)


def compute_metrics(probabilities, labels, threshold: float) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    predictions = (probabilities >= threshold).astype(int)
    labels = labels.astype(int)
    return {
        "threshold": threshold,
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(labels, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(labels, predictions, average="micro", zero_division=0)),
        "exact_match": float((predictions == labels).all(axis=1).mean()),
    }


def main() -> None:
    args = parse_args()
    records = load_labeled_jsonl(args.validation_file)
    messages = [record["message"] for record in records]
    labels = np.array([risk_multihot(record.get("risk_factors", [])) for record in records])
    probabilities = predict_probabilities(args, messages)

    results = [compute_metrics(probabilities, labels, threshold) for threshold in args.thresholds]
    best = max(results, key=lambda item: (item["micro_f1"], item["macro_f1"]))
    payload = {
        "model_dir": str(args.model_dir),
        "validation_file": str(args.validation_file),
        "risk_factor_labels": RISK_FACTOR_LABELS,
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "all_results": results,
    }
    write_json(payload, Path(args.output))
    print(payload["best_metrics"])


if __name__ == "__main__":
    main()
