"""Create project visualizations from saved evaluation outputs.

The script does not rerun model inference. It reads saved outputs under
``results/`` and writes PNG figures under ``visuals/``.
"""

from __future__ import annotations

import os
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

from config import (
    PIPELINE_OUTPUTS_PATH,
    PIPELINE_SUMMARY_PATH,
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

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    return plt, sns


def save_figure(plt, fig, output_path: Path) -> None:
    """Save a compact, high-resolution figure suitable for GitHub display."""
    plt.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white",
    )
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=False,
        square=True,
        linewidths=0.7,
        linecolor="white",
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("True", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=11)
    save_figure(plt, fig, output_path)


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
    display_actions = [textwrap.fill(action, width=32) for action in actions]

    figure_height = max(4.6, 0.85 * len(actions) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, figure_height))
    sns.barplot(x=values, y=display_actions, color="#2D789C", ax=ax)
    ax.set_xlabel("Messages", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Routing Action Distribution", fontsize=16, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=11)
    max_value = max(values, default=1)
    ax.set_xlim(0, max_value * 1.18)
    for index, value in enumerate(values):
        ax.text(value + max_value * 0.02, index, str(value), va="center", fontsize=11, fontweight="bold")
    sns.despine(ax=ax, left=True, bottom=False)
    save_figure(plt, fig, VISUALS_DIR / "routing_distribution.png")


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

    display_labels = [textwrap.fill(label.replace("_", " "), width=28) for label in labels]
    y_positions = list(range(len(labels)))
    bar_height = 0.38
    figure_height = max(7.0, 0.58 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, figure_height))
    ax.barh(
        [pos - bar_height / 2 for pos in y_positions],
        [true_counts[label] for label in labels],
        height=bar_height,
        label="True",
        color="#2D789C",
    )
    ax.barh(
        [pos + bar_height / 2 for pos in y_positions],
        [pred_counts[label] for label in labels],
        height=bar_height,
        label="Predicted",
        color="#D9A11E",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Label count", fontsize=12)
    ax.set_title("Risk-Factor True vs Predicted Counts", fontsize=16, fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    save_figure(plt, fig, VISUALS_DIR / "risk_factor_true_vs_predicted_counts.png")


def plot_risk_threshold_tuning_curve() -> None:
    """Create threshold-tuning plot from saved threshold results.

    If the saved file only contains the selected threshold summary, the figure
    shows the available selected-threshold point rather than inventing a curve.
    """
    plt, _ = require_plotting_libraries()
    payload = read_json(THRESHOLD_RESULTS_PATH)
    results = sorted(payload.get("all_results", []), key=lambda row: row.get("threshold", 0))

    if results:
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        thresholds = [row["threshold"] for row in results]
        ax.plot(
            thresholds,
            [row.get("micro_f1") for row in results],
            color="#2D789C",
            linewidth=2.5,
            marker="o",
            markersize=7,
            label="Micro-F1",
        )
        ax.plot(
            thresholds,
            [row.get("macro_f1") for row in results],
            color="#D9A11E",
            linewidth=2.5,
            marker="o",
            markersize=7,
            label="Macro-F1",
        )
        ax.axvline(
            payload.get("best_threshold", 0.30),
            color="#5D6B78",
            linestyle="--",
            linewidth=1.5,
            label="Selected threshold",
        )
        ax.set_xlabel("Risk-factor threshold", fontsize=12)
        ax.set_ylabel("F1", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title("Risk-Threshold Tuning", fontsize=16, fontweight="bold", pad=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=11, frameon=False)
        save_figure(plt, fig, VISUALS_DIR / "risk_threshold_tuning_curve.png")
        return

    from matplotlib.patches import FancyBboxPatch

    best_threshold = float(payload.get("best_threshold", payload.get("notebook_pipeline_threshold", 0.30)))
    best_metrics = payload.get("best_metrics", {})
    micro_f1 = float(best_metrics.get("micro_f1", 0.0))
    macro_f1 = float(best_metrics.get("macro_f1", 0.0))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.91,
        "Risk-Factor Threshold Selection",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#1F2933",
    )
    card = FancyBboxPatch(
        (0.12, 0.15),
        0.76,
        0.62,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.5,
        edgecolor="#D2DCE5",
        facecolor="#F4F8FA",
    )
    ax.add_patch(card)
    ax.text(0.5, 0.67, "SELECTED THRESHOLD", ha="center", fontsize=11, fontweight="bold", color="#5D6B78")
    ax.text(0.5, 0.49, f"{best_threshold:.2f}", ha="center", fontsize=36, fontweight="bold", color="#2F9FA0")
    ax.text(
        0.5,
        0.31,
        f"Micro-F1  {micro_f1:.4f}     Macro-F1  {macro_f1:.4f}",
        ha="center",
        fontsize=12,
        color="#1F2933",
    )
    ax.text(
        0.5,
        0.20,
        "Saved results contain the selected-threshold summary.",
        ha="center",
        fontsize=9.5,
        color="#5D6B78",
    )
    save_figure(plt, fig, VISUALS_DIR / "risk_threshold_tuning_curve.png")


def plot_model_results_summary() -> None:
    """Create a compact overview from the saved unified-pipeline metrics."""
    plt, _ = require_plotting_libraries()
    summary = read_json(PIPELINE_SUMMARY_PATH)
    labels = [
        "Urgency\naccuracy",
        "Risk exact\nmatch",
        "Insufficient\naccuracy",
        "Human review\nrate",
    ]
    values = [
        summary["urgency_accuracy"],
        summary["risk_exact_match_accuracy"],
        summary["insufficient_accuracy"],
        summary["human_review_rate"],
    ]
    colors = ["#2D789C", "#2E9F62", "#2F9FA0", "#D9A11E"]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Unified Pipeline Results on 50 Saved Test Examples", fontsize=16, fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelsize=11, pad=8)
    ax.tick_params(axis="y", labelsize=11)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    save_figure(plt, fig, VISUALS_DIR / "model_results_summary.png")


def main() -> None:
    """Generate all project visuals from existing saved results."""
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(PIPELINE_OUTPUTS_PATH)

    plot_urgency_confusion_matrix(rows)
    plot_insufficient_confusion_matrix(rows)
    plot_routing_distribution(rows)
    plot_risk_factor_true_vs_predicted_counts(rows)
    plot_risk_threshold_tuning_curve()
    plot_model_results_summary()

    print(f"Created visualizations in {VISUALS_DIR}")


if __name__ == "__main__":
    main()
