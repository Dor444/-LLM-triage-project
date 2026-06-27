"""Run zero-shot and few-shot Qwen urgency prompting baselines.

These prompting baselines classify urgency only. They are included for
comparison with the fine-tuned DistilBERT urgency model and do not affect the
saved unified-pipeline metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import DEFAULT_QWEN_MODEL, URGENCY_LABELS
from data_utils import load_labeled_jsonl, normalize_urgency, write_json, write_jsonl
from metrics_utils import accuracy_score_simple, macro_f1_score_simple


ZERO_SHOT_SYSTEM_PROMPT = """You are labeling patient portal messages for an educational administrative routing experiment.
Classify urgency only as Green, Yellow, or Red.
Do not provide diagnosis, treatment, or medical advice.
Return exactly one JSON object."""

FEW_SHOT_EXAMPLES = [
    (
        "Can you refill my allergy medicine? I feel okay but ran out yesterday.",
        {"urgency": "Green"},
    ),
    (
        "My surgical wound is more red and leaking a little fluid today.",
        {"urgency": "Yellow"},
    ),
    (
        "I have heavy chest pressure and trouble breathing right now.",
        {"urgency": "Red"},
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL, help="Local Qwen model name or path.")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--mode", choices=["zero-shot", "few-shot", "both"], default="both")
    parser.add_argument("--output", required=True, help="Path for metrics JSON.")
    parser.add_argument("--predictions-output", default=None, help="Optional path for per-example JSONL.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_text_generator(model_name: str, local_files_only: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer), tokenizer


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def build_prompt(message: str, mode: str) -> list[dict[str, str]]:
    if mode == "zero-shot":
        user_prompt = (
            "Classify this portal message urgency as Green, Yellow, or Red. "
            "Return JSON like {\"urgency\": \"Green\"}.\n\n"
            f"Message: {message}"
        )
    else:
        examples = "\n".join(
            f"Message: {text}\nAnswer: {json.dumps(label)}" for text, label in FEW_SHOT_EXAMPLES
        )
        user_prompt = (
            f"{examples}\n\n"
            "Now classify this portal message urgency as Green, Yellow, or Red. "
            "Return JSON like {\"urgency\": \"Green\"}.\n\n"
            f"Message: {message}"
        )
    return [
        {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_urgency(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = normalize_urgency(payload.get("urgency"))
            if label:
                return label
        except json.JSONDecodeError:
            pass
    for label in URGENCY_LABELS:
        if re.search(rf"\b{label}\b", text, flags=re.IGNORECASE):
            return label
    return None


def predict_one(generator: Any, tokenizer: Any, message: str, mode: str, args: argparse.Namespace) -> tuple[str | None, str]:
    prompt = render_chat_prompt(tokenizer, build_prompt(message, mode))
    output = generator(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        return_full_text=False,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    generated_text = output[0]["generated_text"]
    return extract_urgency(generated_text), generated_text


def evaluate(labels: list[str], predictions: list[str | None]) -> dict[str, float]:
    fallback_predictions = [prediction or "Green" for prediction in predictions]
    return {
        "accuracy": float(accuracy_score_simple(labels, fallback_predictions)),
        "macro_f1": float(macro_f1_score_simple(labels, fallback_predictions, URGENCY_LABELS)),
        "parse_failure_rate": float(sum(prediction is None for prediction in predictions) / max(len(predictions), 1)),
    }


def progress(iterable, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def run_mode(generator: Any, tokenizer: Any, records: list[dict], mode: str, args: argparse.Namespace):
    predictions = []
    for record in progress(records, desc=mode):
        label, raw_output = predict_one(generator, tokenizer, record["message"], mode, args)
        predictions.append(
            {
                "id": record["id"],
                "mode": mode,
                "gold_urgency": record["urgency"],
                "predicted_urgency": label,
                "raw_output": raw_output,
            }
        )
    metrics = evaluate(
        [record["urgency"] for record in records],
        [prediction["predicted_urgency"] for prediction in predictions],
    )
    return metrics, predictions


def main() -> None:
    args = parse_args()
    records = load_labeled_jsonl(args.test_file)
    if args.limit is not None:
        records = records[: args.limit]

    generator, tokenizer = load_text_generator(args.model, args.local_files_only)
    modes = ["zero-shot", "few-shot"] if args.mode == "both" else [args.mode]

    metrics: dict[str, dict[str, float]] = {}
    all_predictions = []
    for mode in modes:
        mode_metrics, mode_predictions = run_mode(generator, tokenizer, records, mode, args)
        metrics[mode] = mode_metrics
        all_predictions.extend(mode_predictions)

    payload = {
        "model": args.model,
        "test_file": str(args.test_file),
        "num_examples": len(records),
        "metrics": metrics,
    }
    write_json(payload, Path(args.output))
    if args.predictions_output:
        write_jsonl(all_predictions, args.predictions_output)
    print(payload)


if __name__ == "__main__":
    main()
