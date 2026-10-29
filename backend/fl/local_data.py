from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np

FEATURE_COLUMNS = ["age", "gender", "fever", "cough", "oxygen_level", "blood_pressure"]
LABEL_COLUMN = "treatment"
POSITIVE_LABEL = "hospitalization"


def _parse_row(row: dict) -> List[float]:
    return [
        float(row["age"]),
        1.0 if row["gender"].strip().upper() == "F" else 0.0,
        float(row["fever"]),
        float(row["cough"]),
        float(row["oxygen_level"]),
        float(row["blood_pressure"]),
    ]


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    labels: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(_parse_row(row))
            labels.append(1.0 if row[LABEL_COLUMN].strip() == POSITIVE_LABEL else 0.0)
    return np.array(features, dtype=np.float64), np.array(labels, dtype=np.float64)


# Explicit disease_code -> filename mapping (not a fuzzy search), so a request
# for one disease can never silently load a different disease's file.
DISEASE_FILENAMES = {
    "COVID19": "covid_dataset.csv",
    "LUNG_CANCER": "lung_cancer_dataset.csv",
    "DIABETES": "diabetes_dataset.csv",
}


def find_hospital_csv(data_dir: str, hospital: str, disease_code: str) -> str:
    """CSV for ONE (hospital, disease) pair. Every hospital now holds data for
    every disease, so this can no longer just return "the first CSV". The
    disease_code is required, never inferred."""
    if disease_code not in DISEASE_FILENAMES:
        raise ValueError(f"Unknown disease_code: {disease_code}. Known: {list(DISEASE_FILENAMES)}")
    path = os.path.join(data_dir, hospital, DISEASE_FILENAMES[disease_code])
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {disease_code} dataset found for {hospital} at {path}")
    return path


def list_available_diseases(data_dir: str, hospital: str) -> List[str]:
    hosp_dir = os.path.join(data_dir, hospital)
    if not os.path.isdir(hosp_dir):
        return []
    present = set(os.listdir(hosp_dir))
    return [code for code, fname in DISEASE_FILENAMES.items() if fname in present]
