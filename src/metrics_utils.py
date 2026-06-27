"""Small metric helpers shared by evaluation scripts.

The project only needs simple exact-match accuracy and macro-F1 for saved
evaluation artifacts. Keeping these helpers dependency-light makes the
evaluation scripts runnable without loading model-training libraries.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def accuracy_score_simple(labels: Sequence[Any], predictions: Sequence[Any]) -> float:
    """Return exact-match accuracy for two aligned label sequences."""
    return sum(label == prediction for label, prediction in zip(labels, predictions)) / max(len(labels), 1)


def macro_f1_score_simple(
    labels: Sequence[str],
    predictions: Sequence[str],
    label_order: Iterable[str] | None = None,
) -> float:
    """Return unweighted macro-F1 for string labels."""
    label_set = list(label_order) if label_order is not None else sorted(set(labels) | set(predictions))
    scores: list[float] = []

    for label in label_set:
        true_positive = sum(gold == label and pred == label for gold, pred in zip(labels, predictions))
        false_positive = sum(gold != label and pred == label for gold, pred in zip(labels, predictions))
        false_negative = sum(gold == label and pred != label for gold, pred in zip(labels, predictions))

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    return sum(scores) / max(len(scores), 1)


def exact_match_rate(labels: Sequence[set[Any]], predictions: Sequence[set[Any]]) -> float:
    """Return the share of examples whose predicted set exactly matches the true set."""
    return sum(gold == pred for gold, pred in zip(labels, predictions)) / max(len(labels), 1)
