"""Create project visualizations from saved evaluation outputs.

The script does not rerun model inference. It reads saved outputs under
``results/`` and writes PNG figures under ``visuals/``.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from config import (
    PIPELINE_OUTPUTS_PATH,
    RISK_FACTOR_LABELS,
    THRESHOLD_RESULTS_PATH,
    URGENCY_LABELS,
    VISUALS_DIR,
)
from data_utils import read_json, read_jsonl


def require_plotting_libraries():
    """Import plotting libraries lazily so compile checks do not require them."""
    os.environ.setdefault("MPLCONFIGDIR", str(VISUALS_DIR / ".mplconfig"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependency. Run `pip install -r requirements.txt` "
            "and then retry `python src/create_visuals.py`."
        ) from exc

    sns.set_theme(style="whitegrid")
    return plt, sns


def confusion_matrix_counts(rows: list[dict[str, Any]], true_key: str, pred_key: str, labels: list[Any]) -> list[list[int]]:
    """Return count matrix for true/predicted values in saved output rows."""
    matrix = [[0 for _ in labels] for _ in labels]
    label_to_index = {label: index for index, label in enumerate(labels)}

    for row in rows:
        true_value = row[true_key]
        pred_value = row[pred_key]
        if true_value in label_to_index and pred_value in label_to_index:
            matrix[label_to_index[true_value]][label_to_index[pred_value]] += 1

    return matrix


def save_heatmap(matrix, x_labels, y_labels, title: str, output_path: Path) -> None:
    """Save a labeled confusion-matrix heatmap."""
    plt, sns = require_plotting_libraries()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_urgency_confusion_matrix(rows: list[dict[str, Any]]) -> None:
    """Create urgency confusion matrix from saved true/predicted labels."""
    matrix = confusion_matrix_counts(rows, "true_urgency", "pred_urgency", URGENCY_LABELS)
    save_heatmap(
        matrix,
        URGENCY_LABELS,
        URGENCY_LABELS,
        "Urgency Confusion Matrix",
        VISUALS_DIR / "urgency_confusion_matrix.png",
    )


def plot_insufficient_confusion_matrix(rows: list[dict[str, Any]]) -> None:
    """Create insufficient-information confusion matrix."""
    labels = [False, True]
    label_names = ["Sufficient", "Insufficient"]
    matrix = confusion_matrix_counts(rows, "true_insufficient", "pred_insufficient", labels)
    save_heatmap(
        matrix,
        label_names,
        label_names,
        "Insufficient Information Confusion Matrix",
        VISUALS_DIR / "insufficient_confusion_matrix.png",
    )


def plot_routing_distribution(rows: list[dict[str, Any]]) -> None:
    """Create a horizontal bar chart of final routing actions."""
    plt, sns = require_plotting_libraries()
    counts = Counter(row.get("target_action", "Unknown") for row in rows)
    actions = [item[0] for item in counts.most_common()]
    values = [counts[action] for action in actions]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(x=values, y=actions, color="#4C78A8", ax=ax)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title("Routing Action Distribution")
    for index, value in enumerate(values):
        ax.text(value + 0.2, index, str(value), va="center")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "routing_distribution.png", dpi=180)
    plt.close(fig)


def count_risk_factors(rows: list[dict[str, Any]], key: str) -> Counter:
    """Count all risk-factor labels in a saved pipeline-output field."""
    counts: Counter = Counter()
    for row in rows:
        for label in row.get(key, []):
            counts[label] += 1
    return counts


def plot_risk_factor_true_vs_predicted_counts(rows: list[dict[str, Any]]) -> None:
    """Compare true and predicted risk-factor label counts."""
    plt, _ = require_plotting_libraries()
    true_counts = count_risk_factors(rows, "true_risk_factors")
    pred_counts = count_risk_factors(rows, "pred_risk_factors")
    labels = list(RISK_FACTOR_LABELS)

    for label in sorted(set(true_counts) | set(pred_counts)):
        if label not in labels:
            labels.append(label)

    y_positions = list(range(len(labels)))
    bar_height = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [pos - bar_height / 2 for pos in y_positions],
        [true_counts[label] for label in labels],
        height=bar_height,
        label="True",
    )
    ax.barh(
        [pos + bar_height / 2 for pos in y_positions],
        [pred_counts[label] for label in labels],
        height=bar_height,
        label="Predicted",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_title("Risk Factor True vs Predicted Counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "risk_factor_true_vs_predicted_counts.png", dpi=180)
    plt.close(fig)


def plot_risk_threshold_tuning_curve() -> None:
    """Create threshold-tuning plot from saved threshold results.

    If the saved file only contains the selected threshold summary, the figure
    shows the available selected-threshold point rather than inventing a curve.
    """
    plt, _ = require_plotting_libraries()
    payload = read_json(THRESHOLD_RESULTS_PATH)
    results = sorted(payload.get("all_results", []), key=lambda row: row.get("threshold", 0))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if results:
        thresholds = [row["threshold"] for row in results]
        ax.plot(thresholds, [row.get("micro_f1") for row in results], marker="o", label="Micro-F1")
        ax.plot(thresholds, [row.get("macro_f1") for row in results], marker="o", label="Macro-F1")
    else:
        best_threshold = payload.get("best_threshold", payload.get("notebook_pipeline_threshold"))
        best_metrics = payload.get("best_metrics", {})
        ax.scatter([best_threshold], [best_metrics.get("micro_f1")], label="Micro-F1", s=90)
        ax.scatter([best_threshold], [best_metrics.get("macro_f1")], label="Macro-F1", s=90)
        ax.text(
            best_threshold,
            max(best_metrics.get("micro_f1", 0), best_metrics.get("macro_f1", 0)) + 0.02,
            "Saved file contains selected-threshold summary",
            ha="center",
            fontsize=9,
        )

    ax.axvline(payload.get("best_threshold", 0.30), color="gray", linestyle="--", linewidth=1, label="Selected threshold")
    ax.set_xlabel("Risk-factor threshold")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Risk Threshold Tuning Summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "risk_threshold_tuning_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Generate all project visuals from existing saved results."""
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(PIPELINE_OUTPUTS_PATH)

    plot_urgency_confusion_matrix(rows)
    plot_insufficient_confusion_matrix(rows)
    plot_routing_distribution(rows)
    plot_risk_factor_true_vs_predicted_counts(rows)
    plot_risk_threshold_tuning_curve()

    print(f"Created visualizations in {VISUALS_DIR}")


if __name__ == "__main__":
    main()
