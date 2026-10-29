import os
import subprocess
import sys
from pathlib import Path

# Ensure `backend/` is on sys.path regardless of working directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from db.database import init_db, session_scope
from db import models

DATA_DIR = ROOT / "data"


def _ensure_synthetic_data_exists() -> None:
    """Generates the hospital CSVs via scripts/generate_synthetic_data.py
    if they aren't already there. LOCAL DEV ONLY (see call site below) --
    in the docker-compose flow this needs to happen on the HOST before
    `docker-compose build`, not from inside a running container, because
    fl-hospitalN's Dockerfile COPYs its hospital's CSV in at BUILD time.
    By the time this script runs (via `docker-compose exec backend ...`,
    after `up`), those images are already built; generating data in here
    would be too late to affect them, and the backend container no longer
    has a volume mount to persist it to anyway (see docker-compose.yml).

    This used to be duplicated inline in this file with a SEPARATE, older
    generator (`generate_patient_dataset`, seeded identically for all three
    hospitals) that produced a trivially-deterministic label
    (`diagnosis = "severe" if oxygen_level < 92`, `treatment = diagnosis`)
    -- and every run of seed_data.py silently regenerated and overwrote
    the realistic, non-IID, noisy-label CSVs from
    scripts/generate_synthetic_data.py with that old trivial version. That
    duplicate generator has been removed; this is the one place dataset
    generation happens now.
    """
    hospital1_csv = DATA_DIR / "hospital1" / "covid_dataset.csv"
    if hospital1_csv.exists():
        return
    print("No hospital CSVs found -- generating synthetic data first...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py")],
        check=True,
    )


def main() -> None:
    init_db()
    if os.getenv("IN_DOCKER") != "1":
        _ensure_synthetic_data_exists()

    
    with session_scope() as session:
        # Create institutions only when they do not already exist.
        institution_specs = [
            ("Hospital1", "hospital", "City A"),
            ("Hospital2", "hospital", "City B"),
            ("Hospital3", "hospital", "City C"),
            ("ResearchInstitute1", "research", "City X"),
            ("ResearchInstitute2", "research", "City Y"),
        ]

        institutions = {}

        for name, institution_type, location in institution_specs:
            institution = (
                session.query(models.Institution)
                .filter_by(name=name)
                .one_or_none()
            )

            if institution is None:
                institution = models.Institution(
                    name=name,
                    type=institution_type,
                    location=location,
                )
                session.add(institution)

            institutions[name] = institution

        session.flush()

        # Create users only when they do not already exist.
        user_specs = [
            ("admin_h1@example.com", "Hospital1 Admin", "hospital_admin", "Hospital1"),
            ("admin_h2@example.com", "Hospital2 Admin", "hospital_admin", "Hospital2"),
            ("admin_h3@example.com", "Hospital3 Admin", "hospital_admin", "Hospital3"),
            ("researcher@example.com", "Lead Researcher", "researcher", "ResearchInstitute1"),
        ]

        for email, name, role, institution_name in user_specs:
            user = (
                session.query(models.User)
                .filter_by(email=email)
                .one_or_none()
            )

            if user is None:
                session.add(
                    models.User(
                        email=email,
                        name=name,
                        role=role,
                        institution_id=institutions[institution_name].id,
                    )
                )

        session.flush()


    print("Seeded institutions and users.")

    # --- Dataset registration -------------------------------------------
    # In Docker, the backend container no longer has (or needs) filesystem
    # access to backend/data -- each hospital's own fl-hospitalN container
    # is the only place that has its own CSV, because its image was built
    # with only that file COPYed in (see fl/Dockerfile). So registration
    # has to happen from inside each of THOSE containers, not from here.
    if os.getenv("IN_DOCKER") == "1":
        print(
            "\nSkipping dataset registration from inside the backend container "
            "(it intentionally no longer has filesystem access to hospital data "
            "beyond what's needed for local generation above).\n"
            "Register each hospital's dataset by running, from the host:\n"
            "  docker-compose exec fl-hospital1 python3 -m fl.register_dataset_client\n"
            "  docker-compose exec fl-hospital2 python3 -m fl.register_dataset_client\n"
            "  docker-compose exec fl-hospital3 python3 -m fl.register_dataset_client\n"
        )
    else:
        # Local, non-docker dev: one filesystem, one developer, nothing to
        # isolate -- register directly for convenience.
        from services.dataset_service import DatasetService

        with session_scope() as session:
            h1 = session.query(models.Institution).filter_by(name="Hospital1").one()
            h2 = session.query(models.Institution).filter_by(name="Hospital2").one()
            h3 = session.query(models.Institution).filter_by(name="Hospital3").one()

            service = DatasetService(session)

            # Every hospital holds every disease (see
            # scripts/generate_synthetic_data.py), so register all of them --
            # eligibility for "request from all hospitals" reads this table.
            from fl.local_data import DISEASE_FILENAMES
            pretty = {"COVID19": "COVID-19", "LUNG_CANCER": "Lung Cancer", "DIABETES": "Diabetes"}
            for inst in (h1, h2, h3):
                for disease_code, filename in DISEASE_FILENAMES.items():
                    path = DATA_DIR / inst.name.lower() / filename
                    if not path.exists():
                        continue
                    service.register_dataset_from_path(
                        name=f"{pretty[disease_code]} Patient Records - {inst.name}",
                        disease_code=disease_code,
                        institution_id=inst.id,
                        local_file_path=str(path),
                    )
        print("Registered datasets directly (local dev mode).")

    print("\nSeed data complete.")


if __name__ == "__main__":
    # If you're running this on your host machine, Postgres is typically reachable at localhost.
    # If you're running inside docker-compose, use service hostname `postgres`.
    if os.getenv("IN_DOCKER") == "1":
        os.environ["DATABASE_URL"] = os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/vienko"
        )
    else:
        os.environ["DATABASE_URL"] = os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vienko"
        )
    main()
