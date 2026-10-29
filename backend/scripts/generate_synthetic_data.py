"""Generates synthetic clinical CSVs for every (hospital, disease) pair --
3 hospitals x 3 diseases = 9 files, each hospital getting its own patient
population for each disease.

Why: the previous version gave each hospital exactly ONE, DIFFERENT disease,
which structurally made "eligible hospitals for a Diabetes project" resolve
to exactly one hospital regardless of how correct the eligibility query was.
That contradicts the actual FL premise: MULTIPLE hospitals holding DIFFERENT
PATIENTS for the SAME disease, training ONE shared model.

Each disease keeps a canonical feature-outcome relationship; each hospital
applies its own perturbation, preserving the non-IID heterogeneity that made
FedProx measurably better than FedAvg in earlier experiments.

Run: python scripts/generate_synthetic_data.py
"""
from __future__ import annotations

import csv
import hashlib
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N_PER_DATASET = 1000
HOSPITALS = ["hospital1", "hospital2", "hospital3"]

# [age, fever, cough, oxygen_deficit, bp_dev]
DISEASES = {
    "COVID19": {"filename": "covid_dataset.csv", "weights": np.array([0.022, 1.3, 1.0, 0.30, 0.008]), "bias": -3.0},
    "LUNG_CANCER": {"filename": "lung_cancer_dataset.csv", "weights": np.array([0.045, 0.35, 1.15, 0.10, 0.012]), "bias": -3.0},
    "DIABETES": {"filename": "diabetes_dataset.csv", "weights": np.array([0.045, 0.20, 0.12, 0.06, 0.032]), "bias": -3.2},
}

HOSPITAL_PROFILES = {
    "hospital1": {"jitter": 1.00, "bias_shift": 0.0, "noise_sd": 0.85},
    "hospital2": {"jitter": 1.15, "bias_shift": -0.15, "noise_sd": 0.85},
    "hospital3": {"jitter": 0.85, "bias_shift": 0.15, "noise_sd": 0.85},
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _stable_seed(hospital, disease):
    return int(hashlib.sha256(f"{hospital}:{disease}".encode()).hexdigest()[:8], 16)


def generate_dataset(hospital, disease_code):
    cfg = DISEASES[disease_code]
    prof = HOSPITAL_PROFILES[hospital]
    rng = np.random.default_rng(_stable_seed(hospital, disease_code))
    n = N_PER_DATASET

    age = np.clip(rng.normal(55, 15, n), 18, 90)
    gender = rng.choice(["M", "F"], size=n)
    fever = rng.binomial(1, 0.4, n).astype(float)
    cough = rng.binomial(1, 0.4, n).astype(float)
    oxygen_level = np.clip(rng.normal(95, 4, n), 80, 100)
    blood_pressure = np.clip(rng.normal(120, 15, n), 80, 180)

    oxygen_deficit = 95 - oxygen_level
    bp_dev = np.abs(blood_pressure - 120)
    X = np.stack([age - 50, fever, cough, oxygen_deficit, bp_dev], axis=1)
    logit = (cfg["bias"] + prof["bias_shift"]) + X @ (cfg["weights"] * prof["jitter"]) + rng.normal(0, prof["noise_sd"], n)
    treatment = rng.binomial(1, _sigmoid(logit), n)

    severity = 0.6 * oxygen_deficit + 0.5 * fever + 0.4 * cough + rng.normal(0, 2.5, n)
    diagnosis = np.where(severity > np.median(severity), "severe", "mild")

    return [{
        "patient_id": i, "age": round(float(age[i]), 1), "gender": gender[i],
        "fever": int(fever[i]), "cough": int(cough[i]),
        "oxygen_level": round(float(oxygen_level[i]), 2),
        "blood_pressure": round(float(blood_pressure[i]), 2),
        "diagnosis": diagnosis[i],
        "treatment": "hospitalization" if treatment[i] == 1 else "home_care",
    } for i in range(n)]


def main():
    for hospital in HOSPITALS:
        for code, cfg in DISEASES.items():
            rows = generate_dataset(hospital, code)
            d = os.path.join(OUT_DIR, hospital)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, cfg["filename"])
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            rate = sum(1 for r in rows if r["treatment"] == "hospitalization") / len(rows)
            print(f"{hospital}/{code}: {len(rows)} rows (hospitalization rate={rate:.3f})")


if __name__ == "__main__":
    main()
