"""Utilities for loading, validating, and writing project data."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from config import RISK_FACTOR_LABELS, URGENCY_LABELS


RISK_FACTOR_SET = set(RISK_FACTOR_LABELS)
RISK_FACTOR_LOOKUP = {label.lower(): label for label in RISK_FACTOR_LABELS}

URGENCY_ALIASES = {
    "green": "Green",
    "low": "Green",
    "routine": "Green",
    "nonurgent": "Green",
    "non-urgent": "Green",
    "yellow": "Yellow",
    "medium": "Yellow",
    "moderate": "Yellow",
    "soon": "Yellow",
    "red": "Red",
    "high": "Red",
    "urgent": "Red",
    "emergent": "Red",
    "emergency": "Red",
}

RISK_ALIASES = {
    "none": "None",
    "no_risk": "None",
    "no_risk_factors": "None",
    "chest_pain": "Chest_Pain_or_Pressure",
    "chest_pressure": "Chest_Pain_or_Pressure",
    "chest_pain_or_pressure": "Chest_Pain_or_Pressure",
    "shortness_of_breath": "Respiratory_Distress",
    "sob": "Respiratory_Distress",
    "difficulty_breathing": "Respiratory_Distress",
    "breathing": "Respiratory_Distress",
    "respiratory_distress": "Respiratory_Distress",
    "neuro": "Acute_Neurological",
    "neurologic_symptoms": "Acute_Neurological",
    "neurological": "Acute_Neurological",
    "stroke_symptoms": "Acute_Neurological",
    "infection": "Severe_Infection",
    "fever": "Severe_Infection",
    "fever_infection": "Severe_Infection",
    "severe_infection": "Severe_Infection",
    "allergy": "Anaphylaxis_or_Allergy",
    "anaphylaxis": "Anaphylaxis_or_Allergy",
    "anaphylaxis_or_allergy": "Anaphylaxis_or_Allergy",
    "bleeding": "Uncontrolled_Bleeding",
    "uncontrolled_bleeding": "Uncontrolled_Bleeding",
    "severe_pain": "Severe_Pain",
    "pain": "Severe_Pain",
    "trauma": "Trauma_or_Injury",
    "injury": "Trauma_or_Injury",
    "trauma_or_injury": "Trauma_or_Injury",
    "medication": "Medication_Adverse_Reaction",
    "medication_issue": "Medication_Adverse_Reaction",
    "medication_adverse_reaction": "Medication_Adverse_Reaction",
    "adverse_reaction": "Medication_Adverse_Reaction",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write dictionaries to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def normalize_message(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_urgency(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    return URGENCY_ALIASES.get(key)


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    key = str(value).strip().lower()
    if key in {"true", "yes", "y", "1", "insufficient"}:
        return True
    if key in {"false", "no", "n", "0", "sufficient"}:
        return False
    return None


def normalize_risk_factor(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"", "null"}:
        return None
    if key in RISK_FACTOR_LOOKUP:
        return RISK_FACTOR_LOOKUP[key]
    return RISK_ALIASES.get(key)


def normalize_risk_factors(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        values = re.split(r"[,;/|]", value)
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]

    normalized: list[str] = []
    for item in values:
        label = normalize_risk_factor(item)
        if label and label not in normalized:
            normalized.append(label)
    return normalized


def clean_record(record: dict[str, Any], require_labels: bool = True) -> dict[str, Any]:
    """Normalize one raw record into the canonical project schema."""
    labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
    message = normalize_message(
        first_present(record, "message", "text", "patient_message", "portal_message")
    )
    if not message:
        raise ValueError("Missing message text")

    cleaned: dict[str, Any] = {
        "id": str(first_present(record, "id", "message_id", "task_id", "example_id") or ""),
        "message": message,
    }

    urgency = normalize_urgency(
        first_present(record, "urgency", "urgency_label")
        or first_present(labels, "urgency_level", "urgency")
    )
    risk_factors = normalize_risk_factors(
        first_present(record, "risk_factors", "risk_factor_labels", "risks")
        or first_present(labels, "risk_factors", "risk_factor_labels", "risks")
    )
    raw_insufficient = first_present(
        record,
        "insufficient_info",
        "insufficient_information",
        "is_insufficient",
    )
    if raw_insufficient is None:
        raw_insufficient = first_present(
            labels,
            "insufficient_information",
            "insufficient_info",
            "is_insufficient",
        )
    insufficient_info = normalize_bool(raw_insufficient)

    if require_labels:
        missing = []
        if urgency is None:
            missing.append("urgency")
        if insufficient_info is None:
            missing.append("insufficient_info")
        if missing:
            raise ValueError(f"Missing or invalid labels: {', '.join(missing)}")

    if urgency is not None:
        cleaned["urgency"] = urgency
    cleaned["risk_factors"] = risk_factors
    if insufficient_info is not None:
        cleaned["insufficient_info"] = insufficient_info

    if not cleaned["id"]:
        cleaned["id"] = stable_record_id(cleaned["message"])
    return cleaned


def clean_records(
    records: Iterable[dict[str, Any]],
    require_labels: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return valid cleaned records and failed records with error messages."""
    valid: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            cleaned = clean_record(record, require_labels=require_labels)
            valid.append(cleaned)
        except ValueError as exc:
            failed.append({"index": index, "error": str(exc), "raw": record})
    return valid, failed


def stable_record_id(message: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", message.lower()).strip("-")
    return slug[:64] or "message"


def label_to_id(label: str) -> int:
    return URGENCY_LABELS.index(label)


def id_to_label(index: int) -> str:
    return URGENCY_LABELS[int(index)]


def risk_multihot(labels: Iterable[str]) -> list[float]:
    label_set = set(labels)
    return [1.0 if label in label_set else 0.0 for label in RISK_FACTOR_LABELS]


def load_labeled_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records, failed = clean_records(read_jsonl(path), require_labels=True)
    if failed:
        details = "; ".join(f"{item['index']}: {item['error']}" for item in failed[:5])
        raise ValueError(f"{len(failed)} invalid records in {path}. First errors: {details}")
    return records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
