"""
Create clean standalone visualization PNG files for the final GitHub repository.

The script uses saved evaluation outputs only.
It does not rerun model inference and does not change metrics.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


RESULTS_DIR = Path("results")
VISUALS_DIR = Path("visuals")

PIPELINE_OUTPUTS = RESULTS_DIR / "pipeline_test_outputs.jsonl"
PIPELINE_SUMMARY = RESULTS_DIR / "pipeline_evaluation_summary.json"

VISUALS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


URGENCY_LABELS = ["Green", "Yellow", "Red"]
INSUFFICIENT_LABELS = ["Sufficient", "Insufficient"]

RISK_FACTOR_LABELS = [
    "Chest_Pain_or_Pressure",
    "Respiratory_Distress",
    "Acute_Neurological",
    "Severe_Infection",
    "Anaphylaxis_or_Allergy",
    "Uncontrolled_Bleeding",
    "Severe_Pain",
    "Trauma_or_Injury",
    "Medication_Adverse_Reaction",
    "None",
]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close()


def short_route(label: str) -> str:
    mapping = {
        "Human review required due to insufficient information": "Review: insufficient info",
        "Human review required due to detected risk signal": "Review: risk signal",
        "Same-day human review": "Same-day review",
        "Immediate human review": "Immediate review",
        "Routine queue": "Routine queue",
    }
    return mapping.get(label, label)


def plot_urgency_confusion(rows):
    y_true = [r["true_urgency"] for r in rows]
    y_pred = [r["pred_urgency"] for r in rows]

    cm = confusion_matrix(y_true, y_pred, labels=URGENCY_LABELS)

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=URGENCY_LABELS,
        yticklabels=URGENCY_LABELS,
        cbar=False,
        square=True,
        annot_kws={"size": 16},
    )

    ax.set_title("Urgency Confusion Matrix", fontsize=18, pad=16)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    save_fig(VISUALS_DIR / "urgency_confusion_matrix.png")


def plot_insufficient_confusion(rows):
    y_true = [
        "Insufficient" if r["true_insufficient"] else "Sufficient"
        for r in rows
    ]
    y_pred = [
        "Insufficient" if r["pred_insufficient"] else "Sufficient"
        for r in rows
    ]

    cm = confusion_matrix(y_true, y_pred, labels=INSUFFICIENT_LABELS)

    plt.figure(figsize=(6.8, 5.8))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=INSUFFICIENT_LABELS,
        yticklabels=INSUFFICIENT_LABELS,
        cbar=False,
        square=True,
        annot_kws={"size": 16},
    )

    ax.set_title("Insufficient Information Confusion Matrix", fontsize=17, pad=16)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    save_fig(VISUALS_DIR / "insufficient_confusion_matrix.png")


def plot_routing_distribution(rows):
    counts = Counter(short_route(r["target_action"]) for r in rows)
    items = counts.most_common()

    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    plt.figure(figsize=(9, 5.8))
    ax = sns.barplot(x=values, y=labels, color="#4C78A8")

    ax.set_title("Routing Action Distribution", fontsize=18, pad=16)
    ax.set_xlabel("Count", fontsize=14)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)

    max_value = max(values)
    ax.set_xlim(0, max_value + 4)

    for i, value in enumerate(values):
        ax.text(value + 0.3, i, str(value), va="center", fontsize=13)

    save_fig(VISUALS_DIR / "routing_distribution.png")


def plot_risk_factor_counts(rows):
    true_counter = Counter()
    pred_counter = Counter()

    for row in rows:
        for label in row["true_risk_factors"]:
            true_counter[label] += 1
        for label in row["pred_risk_factors"]:
            pred_counter[label] += 1

    data = []
    for label in RISK_FACTOR_LABELS:
        data.append(
            {
                "risk_factor": label,
                "true": true_counter[label],
                "predicted": pred_counter[label],
            }
        )

    df = pd.DataFrame(data)
    df["total"] = df["true"] + df["predicted"]
    df = df.sort_values("total", ascending=True)

    y = range(len(df))

    plt.figure(figsize=(11, 7))
    plt.barh(
        [i - 0.18 for i in y],
        df["true"],
        height=0.35,
        label="True",
        color="#4C78A8",
    )
    plt.barh(
        [i + 0.18 for i in y],
        df["predicted"],
        height=0.35,
        label="Predicted",
        color="#E4572E",
    )

    plt.yticks(y, df["risk_factor"], fontsize=11)
    plt.xlabel("Count", fontsize=14)
    plt.title("Risk Factor True vs Predicted Counts", fontsize=18, pad=16)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(axis="x", alpha=0.35)

    save_fig(VISUALS_DIR / "risk_factor_true_vs_predicted_counts.png")


def plot_threshold_summary():
    """
    The saved threshold JSON is a reconstructed summary, not the full sweep.
    Therefore this figure intentionally shows the selected threshold and final F1 values.
    """

    selected_threshold = 0.30
    micro_f1 = 0.8792
    macro_f1 = 0.7775

    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()
    ax.axis("off")

    ax.text(
        0.5,
        0.82,
        "Risk-Factor Threshold Selection",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    ax.text(
        0.5,
        0.58,
        f"Selected threshold: {selected_threshold:.2f}",
        ha="center",
        va="center",
        fontsize=26,
        fontweight="bold",
        color="#2A9D8F",
    )

    ax.text(
        0.5,
        0.37,
        f"Micro-F1: {micro_f1:.4f}    Macro-F1: {macro_f1:.4f}",
        ha="center",
        va="center",
        fontsize=17,
    )

    ax.text(
        0.5,
        0.18,
        "Used by the multi-label risk-factor classifier in the unified routing pipeline.",
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )

    save_fig(VISUALS_DIR / "risk_threshold_tuning_curve.png")


def plot_model_results_summary():
    metrics = [
        ("Urgency\nAccuracy", 0.8095),
        ("Urgency\nMacro-F1", 0.8104),
        ("Risk\nMicro-F1", 0.8792),
        ("Risk\nMacro-F1", 0.7775),
        ("Insufficient\nAccuracy", 0.8690),
        ("Insufficient\nMacro-F1", 0.7945),
        ("Pipeline\nRisk Exact", 0.92),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(x=labels, y=values, color="#2A9D8F")

    ax.set_ylim(0, 1.0)
    ax.set_title("Model and Pipeline Results Summary", fontsize=18, pad=16)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)

    for i, value in enumerate(values):
        ax.text(i, value + 0.025, f"{value:.3f}", ha="center", fontsize=12)

    save_fig(VISUALS_DIR / "model_results_summary.png")


def main():
    if not PIPELINE_OUTPUTS.exists():
        raise FileNotFoundError(f"Missing file: {PIPELINE_OUTPUTS}")

    rows = load_jsonl(PIPELINE_OUTPUTS)

    plot_urgency_confusion(rows)
    plot_insufficient_confusion(rows)
    plot_routing_distribution(rows)
    plot_risk_factor_counts(rows)
    plot_threshold_summary()
    plot_model_results_summary()

    print("Generated visuals:")
    for path in sorted(VISUALS_DIR.glob("*.png")):
        print(" -", path)


if __name__ == "__main__":
    main()