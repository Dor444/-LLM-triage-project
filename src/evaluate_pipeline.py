"""Evaluate unified routing pipeline predictions against labeled JSONL data.

This script reads saved pipeline predictions. It does not rerun model inference,
so the reported metrics remain tied to the checked-in JSONL result artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_utils import load_labeled_jsonl, normalize_bool, normalize_risk_factors, normalize_urgency, read_jsonl, write_json
from metrics_utils import accuracy_score_simple, exact_match_rate, macro_f1_score_simple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def predicted_urgency(record: dict[str, Any]) -> str | None:
    value = record.get("pred_urgency", record.get("urgency"))
    if isinstance(value, dict):
        value = value.get("label")
    return normalize_urgency(value)


def predicted_risks(record: dict[str, Any]) -> list[str]:
    value = record.get("pred_risk_factors", record.get("risk_factors", []))
    if isinstance(value, list) and value and isinstance(value[0], dict):
        value = [item.get("label") or item.get("name") for item in value]
    return normalize_risk_factors(value)


def predicted_insufficient(record: dict[str, Any]) -> bool | None:
    value = record.get(
        "pred_insufficient",
        record.get("insufficient_info", record.get("insufficient_information")),
    )
    if isinstance(value, dict):
        value = value.get("value")
    return normalize_bool(value)


def align_records(labels: list[dict], predictions: list[dict]) -> list[tuple[dict, dict]]:
    prediction_by_id = {
        prediction.get("id"): prediction
        for prediction in predictions
        if prediction.get("id")
    }
    if prediction_by_id and all(label.get("id") in prediction_by_id for label in labels):
        return [(label, prediction_by_id[label["id"]]) for label in labels]
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions have different lengths and cannot be aligned by id.")
    return list(zip(labels, predictions))


def main() -> None:
    args = parse_args()
    labels = load_labeled_jsonl(args.labels_file)
    predictions = read_jsonl(args.predictions_file)
    pairs = align_records(labels, predictions)

    gold_urgency = [label["urgency"] for label, _ in pairs]
    pred_urgency = [predicted_urgency(prediction) or "Green" for _, prediction in pairs]

    gold_risks = [set(label.get("risk_factors", [])) for label, _ in pairs]
    pred_risks = [set(predicted_risks(prediction)) for _, prediction in pairs]

    gold_insufficient = [bool(label["insufficient_info"]) for label, _ in pairs]
    pred_insufficient = [
        bool(predicted_insufficient(prediction))
        for _, prediction in pairs
    ]

    human_review_flags = [
        bool(prediction.get("needs_human_review", prediction.get("human_review_required")))
        for _, prediction in pairs
    ]

    metrics = {
        "num_examples": len(pairs),
        "urgency_accuracy": float(accuracy_score_simple(gold_urgency, pred_urgency)),
        "urgency_macro_f1": float(macro_f1_score_simple(gold_urgency, pred_urgency)),
        "risk_exact_match_accuracy": float(exact_match_rate(gold_risks, pred_risks)),
        "insufficient_accuracy": float(accuracy_score_simple(gold_insufficient, pred_insufficient)),
        "human_review_rate": float(sum(human_review_flags) / max(len(human_review_flags), 1)),
    }
    write_json(metrics, Path(args.output))
    print(metrics)


if __name__ == "__main__":
    main()
