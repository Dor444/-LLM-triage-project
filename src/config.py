"""Project-wide configuration for noisy medical intake routing."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALS_DIR = PROJECT_ROOT / "visuals"

RANDOM_SEED = 42

URGENCY_LABELS = ["Green", "Yellow", "Red"]
URGENCY_TO_ID = {label: idx for idx, label in enumerate(URGENCY_LABELS)}
ID_TO_URGENCY = {idx: label for label, idx in URGENCY_TO_ID.items()}

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

DEFAULT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_LENGTH = 160
DEFAULT_RISK_THRESHOLD = 0.30

PIPELINE_DISCLAIMER = (
    "Administrative routing signal only. Not a diagnosis, treatment "
    "recommendation, or substitute for human clinical review."
)

SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}

PIPELINE_OUTPUTS_PATH = RESULTS_DIR / "pipeline_test_outputs.jsonl"
THRESHOLD_RESULTS_PATH = RESULTS_DIR / "risk_factor_threshold_tuning_results.json"
PIPELINE_SUMMARY_PATH = RESULTS_DIR / "pipeline_evaluation_summary.json"
MODEL_RESULTS_MARKDOWN_PATH = RESULTS_DIR / "model_results_summary.md"
