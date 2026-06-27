"""Generate synthetic patient portal messages with a local LLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import DEFAULT_QWEN_MODEL, RANDOM_SEED, RISK_FACTOR_LABELS, URGENCY_LABELS
from data_utils import clean_record, seed_everything, write_json, write_jsonl


GENERATION_SYSTEM_PROMPT = """You create synthetic patient portal messages for an educational administrative routing project.
The output is not medical advice and must not include diagnosis or treatment instructions.
Return exactly one JSON object and no extra text."""

GENERATION_USER_TEMPLATE = """Create one noisy but realistic patient portal message.

Allowed urgency labels: {urgency_labels}
Allowed risk factors: {risk_labels}

Return JSON with exactly these keys:
- id: short unique string
- message: patient-written portal message, 1 to 5 sentences, may include typos or missing context
- urgency: one of Green, Yellow, Red
- risk_factors: list of zero or more allowed risk factors
- insufficient_info: boolean

Vary the scenario, age context, writing style, and completeness.
Example number: {index}
"""

DRY_RUN_EXAMPLES = [
    {
        "id": "dry-green-refill",
        "message": "Hi, can you refill my thyroid medicine? I feel fine but only have two tablets left.",
        "urgency": "Green",
        "risk_factors": ["None"],
        "insufficient_info": False,
    },
    {
        "id": "dry-yellow-postop",
        "message": "My stitches area is red and warmer today after the procedure last week. No fever that I know of.",
        "urgency": "Yellow",
        "risk_factors": ["Severe_Infection"],
        "insufficient_info": False,
    },
    {
        "id": "dry-red-chest",
        "message": "I am having bad chest pressure and trouble breathing right now, please tell me what to do.",
        "urgency": "Red",
        "risk_factors": ["Chest_Pain_or_Pressure", "Respiratory_Distress"],
        "insufficient_info": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL, help="Local Hugging Face causal LM.")
    parser.add_argument("--num-tasks", type=int, default=3000)
    parser.add_argument("--output", required=True, help="Path for valid JSONL examples.")
    parser.add_argument("--failed-output", required=True, help="Path for failed generations.")
    parser.add_argument("--summary-output", default=None, help="Optional path for summary JSON.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create template examples without loading an LLM.")
    return parser.parse_args()


def build_prompt(index: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERATION_USER_TEMPLATE.format(
                urgency_labels=", ".join(URGENCY_LABELS),
                risk_labels=", ".join(RISK_FACTOR_LABELS),
                index=index,
            ),
        },
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON object: {exc}") from exc


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


def generate_one(generator: Any, tokenizer: Any, index: int, args: argparse.Namespace) -> dict[str, Any]:
    prompt = render_chat_prompt(tokenizer, build_prompt(index))
    output = generator(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        return_full_text=False,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    generated_text = output[0]["generated_text"]
    return extract_json_object(generated_text)


def progress(iterable, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    valid: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    if args.dry_run:
        for index in range(args.num_tasks):
            raw = dict(DRY_RUN_EXAMPLES[index % len(DRY_RUN_EXAMPLES)])
            raw["id"] = f"{raw['id']}-{index:04d}"
            valid.append(clean_record(raw, require_labels=True))
    else:
        generator, tokenizer = load_text_generator(args.model, args.local_files_only)
        for index in progress(range(args.num_tasks), desc="Generating"):
            try:
                raw = generate_one(generator, tokenizer, index, args)
                valid.append(clean_record(raw, require_labels=True))
            except Exception as exc:
                failed.append({"index": index, "error": str(exc)})

    write_jsonl(valid, args.output)
    write_jsonl(failed, args.failed_output)

    summary = {
        "num_tasks": args.num_tasks,
        "valid_examples": len(valid),
        "failed_examples": len(failed),
        "success_rate": round(len(valid) / max(args.num_tasks, 1), 4),
        "model": "dry-run" if args.dry_run else args.model,
    }
    summary_path = args.summary_output or str(Path(args.output).with_suffix(".summary.json"))
    write_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
