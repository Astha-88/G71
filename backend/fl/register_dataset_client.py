"""Run from INSIDE a hospital's own collaborator container:

    docker-compose exec fl-hospital1 python3 -m fl.register_dataset_client

Registers EVERY dataset this hospital holds (one per disease -- see
scripts/generate_synthetic_data.py for why every hospital holds all of them)
with the backend, by POSTing the raw bytes to /datasets/register over the
docker-compose network. The backend stores them in MinIO having never touched
a local path itself; this container is the only place that ever had the files.

Why this matters beyond tidiness: "request from all eligible hospitals"
decides eligibility from the `datasets` TABLE, which is populated by THIS
script -- not by the CSV files existing. Registering only one disease per
hospital would leave the table showing one hospital per disease, so an
all-eligible request would still resolve to a single hospital no matter how
correct the query is.

Idempotent: re-running skips any (institution, disease) already registered
instead of creating duplicate dataset rows.
"""
from __future__ import annotations

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fl.local_data import find_hospital_csv, list_available_diseases  # noqa: E402

HOSPITAL_NAME = os.environ["HOSPITAL_NAME"]
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

# Matches the institution names scripts/seed_data.py creates ("Hospital1" etc.).
INSTITUTION_NAME = HOSPITAL_NAME.capitalize()

PRETTY_DISEASE = {"COVID19": "COVID-19", "LUNG_CANCER": "Lung Cancer", "DIABETES": "Diabetes"}


def _resolve_institution_id(institution_name: str) -> int:
    resp = requests.get(f"{BACKEND_URL}/institutions", timeout=10)
    resp.raise_for_status()
    for inst in resp.json():
        if inst["name"].lower() == institution_name.lower():
            return inst["id"]
    raise RuntimeError(
        f"No institution named '{institution_name}' found on the backend. "
        "Run scripts/seed_data.py first."
    )


def _already_registered(institution_id: int) -> set:
    resp = requests.get(f"{BACKEND_URL}/datasets", timeout=10)
    resp.raise_for_status()
    return {d["disease_code"] for d in resp.json() if d["institution_id"] == institution_id}


def main() -> None:
    diseases = list_available_diseases(DATA_DIR, HOSPITAL_NAME)
    if not diseases:
        raise RuntimeError(f"No datasets found for {HOSPITAL_NAME} under {DATA_DIR}")

    institution_id = _resolve_institution_id(INSTITUTION_NAME)
    existing = _already_registered(institution_id)

    registered = skipped = 0
    for disease_code in diseases:
        if disease_code in existing:
            print(f"[{HOSPITAL_NAME}] {disease_code}: already registered, skipping")
            skipped += 1
            continue

        csv_path = find_hospital_csv(DATA_DIR, HOSPITAL_NAME, disease_code)
        name = f"{PRETTY_DISEASE.get(disease_code, disease_code)} Patient Records - {INSTITUTION_NAME}"
        print(f"[{HOSPITAL_NAME}] {disease_code}: uploading {os.path.basename(csv_path)} as institution_id={institution_id}")

        with open(csv_path, "rb") as f:
            files = {"file": (os.path.basename(csv_path), f, "text/csv")}
            data = {
                "name": name,
                "disease_code": disease_code,
                "institution_id": str(institution_id),
                "format": "csv",
            }
            resp = requests.post(f"{BACKEND_URL}/datasets/register", data=data, files=files, timeout=30)

        if resp.status_code >= 400:
            raise RuntimeError(f"Registration of {disease_code} failed ({resp.status_code}): {resp.text}")
        print(f"[{HOSPITAL_NAME}] {disease_code}: registered dataset id={resp.json()['id']}")
        registered += 1

    print(f"[{HOSPITAL_NAME}] done: {registered} registered, {skipped} already present")


if __name__ == "__main__":
    main()
