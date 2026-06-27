"""Split a labeled JSONL dataset into train, validation, and test files."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from config import RANDOM_SEED, SPLIT_FILENAMES
from data_utils import load_labeled_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to labeled JSONL data.")
    parser.add_argument("--output-dir", required=True, help="Directory for split JSONL files.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def label_counts(records: list[dict]) -> dict[str, int]:
    return dict(Counter(record["urgency"] for record in records))


def fallback_train_test_split(
    records: list[dict],
    test_size: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_size))
    test_count = min(test_count, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
    return shuffled[test_count:], shuffled[:test_count]


def split_once(
    records: list[dict],
    test_size: float,
    seed: int,
    labels: list[str],
) -> tuple[list[dict], list[dict]]:
    try:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            records,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except (ImportError, ValueError):
        return fallback_train_test_split(records, test_size=test_size, seed=seed)


def main() -> None:
    args = parse_args()
    total = args.train_size + args.validation_size + args.test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train-size + validation-size + test-size must equal 1.0")

    records = load_labeled_jsonl(args.input)
    labels = [record["urgency"] for record in records]

    train_records, temp_records = split_once(
        records,
        test_size=1.0 - args.train_size,
        seed=args.seed,
        labels=labels,
    )

    temp_labels = [record["urgency"] for record in temp_records]
    validation_fraction_of_temp = args.validation_size / (args.validation_size + args.test_size)
    validation_records, test_records = split_once(
        temp_records,
        test_size=1.0 - validation_fraction_of_temp,
        seed=args.seed,
        labels=temp_labels,
    )

    output_dir = Path(args.output_dir)
    write_jsonl(train_records, output_dir / SPLIT_FILENAMES["train"])
    write_jsonl(validation_records, output_dir / SPLIT_FILENAMES["validation"])
    write_jsonl(test_records, output_dir / SPLIT_FILENAMES["test"])

    metadata = {
        "input": str(args.input),
        "seed": args.seed,
        "counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "urgency_distribution": {
            "train": label_counts(train_records),
            "validation": label_counts(validation_records),
            "test": label_counts(test_records),
        },
    }
    write_json(metadata, output_dir / "split_metadata.json")
    print(metadata)


if __name__ == "__main__":
    main()
