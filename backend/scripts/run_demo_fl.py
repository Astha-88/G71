import os
import sys
from pathlib import Path

# Ensure `backend/` is on sys.path regardless of working directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from db.database import init_db, session_scope
from db import models
from services.access_service import AccessService
from services.research_project_service import ResearchProjectService
from services.openfl_service import OpenFLService


def main() -> None:
    init_db()

    # --- Step 1: project + request + approve, in one transaction, then
    # commit -------------------------------------------------------------
    # (run_experiment_sync below opens its OWN db session, since it's
    # written to also work as a FastAPI BackgroundTask, which can't share a
    # session with the request that scheduled it. That means the
    # experiment row created here has to be committed -- not just
    # flushed -- before we call it, or the training job's session won't
    # see it yet under normal read-committed isolation.)
    with session_scope() as session:
        dataset = session.query(models.Dataset).order_by(models.Dataset.id.asc()).first()
        if not dataset:
            raise RuntimeError("No datasets found. Run scripts/seed_data.py first.")

        researcher = (
            session.query(models.User)
            .filter(models.User.email == "researcher@example.com")
            .one_or_none()
        )
        if not researcher:
            raise RuntimeError("Researcher user not found. Re-run scripts/seed_data.py.")

        owning_admin = (
            session.query(models.User)
            .filter(
                models.User.role == "hospital_admin",
                models.User.institution_id == dataset.institution_id,
            )
            .first()
        )
        if not owning_admin:
            raise RuntimeError(
                f"No hospital_admin found for institution {dataset.institution_id}. "
                "Re-run scripts/seed_data.py."
            )

        disease = session.query(models.DiseaseRegistry).get(dataset.disease_id)
        topic = disease.code if disease else dataset.name

        # Every request now belongs to a ResearchProject -- even this
        # single-hospital demo goes through ResearchProjectService rather
        # than calling AccessService.create_request directly, exercising
        # the same path the real product uses (one-active-project
        # enforcement, hospital-blocking checks, etc).
        proj_svc = ResearchProjectService(session)
        project = proj_svc.create_project(
            researcher_id=researcher.id,
            topic=topic,
            purpose="Demo study",
            hospital_scope="specific",
            hospital_ids=[dataset.institution_id],
            fl_rounds=10,
            fedprox_mu=0.05,
        )
        req = project.requests[0]
        print(f"Created research project {project.id} ({topic}) with request {req.id} for dataset {dataset.name}")

        access = AccessService(session)
        updated = access.approve(
            request_id=req.id, approver_id=owning_admin.id, decision="approved", comment="ok"
        )
        print(f"Request status after approval by {owning_admin.email}: {updated.status}")
        proj_svc.sync_project_status(project.id)

        experiment_id = None
        if updated.status == "approved":
            exp = models.FederatedExperiment(
                name=f"Demo FL on {dataset.name}",
                dataset_id=dataset.id,
                project_id=project.id,
                requester_id=researcher.id,
                fl_rounds=10,
                fedprox_mu=0.05,
                status="created",
            )
            session.add(exp)
            session.flush()
            experiment_id = exp.id
        # `with` block exits here -> commits, so experiment_id is now
        # visible to any other session, including the one below.

    if experiment_id is None:
        print("Request was not approved; no experiment created.")
        return

    # --- Step 2: run real multi-process FedAvg/FedProx training -----------
    # This spawns 3 real OS subprocesses (one per hospital), talks to them
    # over real HTTP on loopback, and persists the final model. Blocking
    # here is fine for a CLI demo script; the FastAPI endpoint that does
    # the same thing (POST /experiments/start/{id}) runs this as a
    # background task instead so it doesn't hold an HTTP connection open.
    print(f"\nStarting federated training for experiment {experiment_id} (this spawns 3 subprocesses)...")

    def on_round(round_idx: int, metrics: dict) -> None:
        print(
            f"  round {round_idx}: loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f}"
        )

    OpenFLService.run_experiment_sync(experiment_id, on_round_complete=on_round)

    # --- Step 3: read back the result --------------------------------------
    with session_scope() as session:
        experiment = session.query(models.FederatedExperiment).get(experiment_id)
        print(f"\nExperiment {experiment.id} finished with status={experiment.status}")
        if experiment.global_model_json:
            print("A trained global model was persisted and is servable via "
                  f"POST /experiments/{experiment.id}/predict")


if __name__ == "__main__":
    if os.getenv("IN_DOCKER") == "1":
        os.environ["DATABASE_URL"] = os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/vienko"
        )
    else:
        os.environ["DATABASE_URL"] = os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vienko"
        )
    main()
