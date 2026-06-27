"""Unified JSON routing pipeline for patient portal messages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_RISK_THRESHOLD,
    ID_TO_URGENCY,
    PIPELINE_DISCLAIMER,
    RISK_FACTOR_LABELS,
)
from data_utils import clean_records, read_jsonl, write_jsonl


@dataclass
class SequenceModel:
    tokenizer: Any
    model: Any
    device: Any
    max_length: int

    @classmethod
    def load(cls, model_dir: str | Path, max_length: int, local_files_only: bool) -> "SequenceModel":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device, max_length=max_length)

    def predict_logits(self, text: str):
        encoded = self.tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            return self.model(**encoded).logits.detach().cpu().numpy()[0]


def softmax(logits):
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))


class IntakeRoutingPipeline:
    def __init__(
        self,
        urgency_model: SequenceModel,
        risk_model: SequenceModel,
        insufficient_model: SequenceModel,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        insufficient_threshold: float = 0.50,
        low_confidence_threshold: float = 0.55,
    ) -> None:
        self.urgency_model = urgency_model
        self.risk_model = risk_model
        self.insufficient_model = insufficient_model
        self.risk_threshold = risk_threshold
        self.insufficient_threshold = insufficient_threshold
        self.low_confidence_threshold = low_confidence_threshold

    def route(self, message: str, message_id: str = "") -> dict[str, Any]:
        urgency_logits = self.urgency_model.predict_logits(message)
        urgency_probabilities = softmax(urgency_logits)
        urgency_index = int(urgency_probabilities.argmax())
        urgency = ID_TO_URGENCY.get(urgency_index, str(urgency_index))
        urgency_confidence = float(urgency_probabilities[urgency_index])

        risk_probabilities = sigmoid(self.risk_model.predict_logits(message))
        risk_scores = {
            label: float(risk_probabilities[index])
            for index, label in enumerate(RISK_FACTOR_LABELS)
        }
        risk_factors = [
            label
            for label, score in risk_scores.items()
            if score >= self.risk_threshold
        ]
        non_none_risks = [label for label in risk_factors if label != "None"]
        if non_none_risks:
            risk_factors = non_none_risks
        elif not risk_factors:
            risk_factors = ["None"]

        insufficient_logits = self.insufficient_model.predict_logits(message)
        insufficient_probabilities = softmax(insufficient_logits)
        insufficient_probability = float(insufficient_probabilities[1])
        insufficient_info = insufficient_probability >= self.insufficient_threshold
        insufficient_confidence = float(max(insufficient_probabilities))

        review_reasons = []
        if urgency in {"Yellow", "Red"}:
            review_reasons.append(f"urgency_{urgency.lower()}")
        if any(label != "None" for label in risk_factors):
            review_reasons.append("risk_factor_present")
        if insufficient_info:
            review_reasons.append("insufficient_information")
        if urgency_confidence < self.low_confidence_threshold:
            review_reasons.append("low_urgency_confidence")
        if insufficient_confidence < self.low_confidence_threshold:
            review_reasons.append("low_insufficient_info_confidence")

        if urgency == "Red":
            target_action = "Immediate human review"
        elif urgency == "Yellow":
            target_action = "Same-day human review"
        elif insufficient_info:
            target_action = "Human review required due to insufficient information"
        elif any(label != "None" for label in risk_factors):
            target_action = "Human review required due to detected risk signal"
        else:
            target_action = "Routine queue"

        human_review_required = target_action != "Routine queue" or bool(review_reasons)

        return {
            "id": message_id,
            "message": message,
            "input_message": message,
            "urgency": urgency,
            "pred_urgency": urgency,
            "urgency_confidence": urgency_confidence,
            "risk_factors": risk_factors,
            "pred_risk_factors": risk_factors,
            "risk_factor_scores": risk_scores,
            "risk_threshold": self.risk_threshold,
            "insufficient_info": bool(insufficient_info),
            "pred_insufficient": bool(insufficient_info),
            "insufficient_info_probability": insufficient_probability,
            "human_review_required": bool(human_review_required),
            "needs_human_review": bool(human_review_required),
            "target_action": target_action,
            "review_reasons": review_reasons,
            "analysis": {
                "urgency": urgency,
                "urgency_confidence": urgency_confidence,
                "risk_factors": risk_factors,
                "insufficient": "Insufficient" if insufficient_info else "Sufficient",
                "insufficient_confidence": insufficient_confidence,
            },
            "routing": {
                "needs_human_review": bool(human_review_required),
                "target_action": target_action,
            },
            "debug_probabilities": {
                "urgency_probabilities": {
                    ID_TO_URGENCY.get(index, str(index)): float(probability)
                    for index, probability in enumerate(urgency_probabilities)
                },
                "risk_factor_probabilities": risk_scores,
                "insufficient_probabilities": {
                    "Sufficient": float(insufficient_probabilities[0]),
                    "Insufficient": float(insufficient_probabilities[1]),
                },
            },
            "disclaimer": PIPELINE_DISCLAIMER,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--message", help="Single portal message to route.")
    input_group.add_argument("--input-file", help="JSONL file containing messages.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--urgency-model-dir", required=True)
    parser.add_argument("--risk-model-dir", required=True)
    parser.add_argument("--insufficient-model-dir", required=True)
    parser.add_argument("--risk-threshold", type=float, default=DEFAULT_RISK_THRESHOLD)
    parser.add_argument("--insufficient-threshold", type=float, default=0.50)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_input_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.message:
        return [{"id": "cli-message", "message": args.message}]
    records, failed = clean_records(read_jsonl(args.input_file), require_labels=False)
    if failed:
        raise ValueError(f"Input file contains {len(failed)} invalid records; first error: {failed[0]}")
    return records


def main() -> None:
    args = parse_args()
    pipeline = IntakeRoutingPipeline(
        urgency_model=SequenceModel.load(args.urgency_model_dir, args.max_length, args.local_files_only),
        risk_model=SequenceModel.load(args.risk_model_dir, args.max_length, args.local_files_only),
        insufficient_model=SequenceModel.load(args.insufficient_model_dir, args.max_length, args.local_files_only),
        risk_threshold=args.risk_threshold,
        insufficient_threshold=args.insufficient_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
    )
    records = load_input_records(args)
    outputs = [pipeline.route(record["message"], record.get("id", "")) for record in records]
    write_jsonl(outputs, args.output)
    print({"routed_messages": len(outputs), "output": args.output})


if __name__ == "__main__":
    main()
