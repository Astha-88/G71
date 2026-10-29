from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import numpy as np

from db import models
from db.database import session_scope
from fl.aggregator import FederatedTrainer

logger = logging.getLogger(__name__)

HOSPITAL_CLIENTS = ["hospital1", "hospital2", "hospital3"]


def _resolve_hospital_clients(project_id: Optional[int], session) -> list:
    """Which collaborator hostnames actually participate in this training
    run. When the experiment belongs to a ResearchProject, ONLY that
    project's currently-APPROVED hospitals participate -- this is the real
    enforcement behind "training must be authorized based on actual
    approved participants": a researcher with an approved request at
    Hospital1 must not end up training against Hospital2's or Hospital3's
    data too, just because they're also running collaborator containers.

    Falls back to the fixed 3-hospital demo list ONLY when there's no
    project at all (scripts/run_demo_fl.py, or an experiment predating
    projects). That fallback is a named, deliberate scope limit -- a real
    N-hospital deployment should have no such fallback, only
    project-derived participant lists; it exists here purely so the
    existing demo script (which doesn't go through the project system)
    keeps working unchanged.
    """
    if project_id is None:
        return list(HOSPITAL_CLIENTS)

    approved_requests = (
        session.query(models.DatasetAccessRequest)
        .filter(
            models.DatasetAccessRequest.project_id == project_id,
            models.DatasetAccessRequest.status == "approved",
        )
        .all()
    )
    hospitals = []
    seen = set()
    for req in approved_requests:
        dataset = session.query(models.Dataset).get(req.dataset_id)
        if not dataset:
            continue
        institution = session.query(models.Institution).get(dataset.institution_id)
        if not institution:
            continue
        # Matches the naming convention used everywhere else in the FL
        # layer: seed_data.py creates institutions named "Hospital1" etc,
        # and fl/collaborator_server.py's HOSPITAL_NAME env var / the
        # docker-compose service names are the lowercase form of exactly
        # that name.
        slug = institution.name.lower()
        if slug not in seen:
            seen.add(slug)
            hospitals.append(slug)
    return hospitals

def _resolve_disease_code(project_id: Optional[int], session) -> str:
    """Which disease's data collaborators train on. Derived from the CANONICAL
    DiseaseRegistry.code of the project's own datasets (not by re-parsing the
    free-text topic). Prefers an APPROVED request's dataset."""
    FALLBACK = "COVID19"  # legacy path only: experiments with no project
    if project_id is None:
        return FALLBACK
    req = (session.query(models.DatasetAccessRequest)
           .filter(models.DatasetAccessRequest.project_id == project_id,
                   models.DatasetAccessRequest.status == "approved").first())
    if not req:
        req = (session.query(models.DatasetAccessRequest)
               .filter(models.DatasetAccessRequest.project_id == project_id).first())
    if req and req.dataset_id:
        ds = session.query(models.Dataset).get(req.dataset_id)
        if ds:
            disease = session.query(models.DiseaseRegistry).get(ds.disease_id)
            if disease and disease.code:
                return disease.code
    return FALLBACK


# "subprocess" (default): spawn local OS processes -- what scripts/run_demo_fl.py
# and local `uvicorn main:app` without docker-compose use.
# "network": talk to already-running fl-hospital1/2/3 containers over the
# docker-compose network (see docker-compose.yml). Set by FL_DEPLOYMENT_MODE
# in the backend container's environment.
FL_DEPLOYMENT_MODE = os.getenv("FL_DEPLOYMENT_MODE", "subprocess")

# mTLS config, set by docker-compose.yml on the backend service. This was a
# real, silent gap during development: docker-compose mounted the certs and
# set these env vars, but nothing in this file ever READ them, so the
# aggregator would have connected over plain HTTP regardless of the
# FL_TLS_ENABLED=1 set in the container's environment -- the certs would
# have sat there unused. Caught by tracing the env var through to here
# rather than assuming "I set it in docker-compose" was the same as "the
# code uses it."
FL_TLS_ENABLED = os.getenv("FL_TLS_ENABLED", "0") == "1"
FL_TLS_CERT = os.getenv("FL_TLS_CERT")
FL_TLS_KEY = os.getenv("FL_TLS_KEY")
FL_TLS_CA = os.getenv("FL_TLS_CA")

# Fault-tolerance policy -- deliberately deployment-level ops config (env
# vars), not per-experiment API fields like fedprox_mu/dp_*. How long to
# wait for a straggler and what quorum to require is a reliability policy
# an operator sets for the whole deployment, not something a researcher
# should be tuning per training run.
FL_CLIENT_TIMEOUT_SECONDS = float(os.getenv("FL_CLIENT_TIMEOUT_SECONDS", "30.0"))
FL_MIN_QUORUM_FRACTION = float(os.getenv("FL_MIN_QUORUM_FRACTION", "0.5"))


class OpenFLService:
    """Thin synchronous entry point meant to run inside a background task
    (see api/experiments.py's use of FastAPI's BackgroundTasks). It owns its
    own DB session for the lifetime of the run -- request-scoped sessions
    can't safely cross into a background thread, so this deliberately
    doesn't take one as a constructor argument like the rest of the
    services do.
    """

    @staticmethod
    def run_experiment_sync(
        experiment_id: int,
        on_round_complete: Optional[Callable[[int, dict], None]] = None,
    ) -> None:
        with session_scope() as session:
            experiment = session.query(models.FederatedExperiment).get(experiment_id)
            if not experiment:
                logger.error("run_experiment_sync: experiment %s not found", experiment_id)
                return
            experiment.status = "running"
            session.flush()
            rounds = experiment.fl_rounds
            mu = experiment.fedprox_mu
            dp_enabled = experiment.dp_enabled
            dp_clip_norm = experiment.dp_clip_norm
            dp_noise_multiplier = experiment.dp_noise_multiplier
            dp_delta = experiment.dp_delta
            aggregation_strategy = experiment.aggregation_strategy
            secagg_enabled = experiment.secagg_enabled
            hospital_clients = _resolve_hospital_clients(experiment.project_id, session)
            disease_code = _resolve_disease_code(experiment.project_id, session)

        if not hospital_clients:
            # Enforces "training requires actual approved participants" --
            # a project with zero currently-approved hospitals must never
            # silently fall back to training against SOME hospital anyway.
            logger.error(
                "run_experiment_sync: experiment %s has no approved hospital participants "
                "(project_id=%s) -- refusing to start training",
                experiment_id, experiment.project_id if experiment else None,
            )
            with session_scope() as session:
                experiment = session.query(models.FederatedExperiment).get(experiment_id)
                experiment.status = "failed"
            return

        try:
            trainer = FederatedTrainer(
                hospitals=hospital_clients, disease_code=disease_code, local_epochs=5, lr=0.2, mu=mu, mode=FL_DEPLOYMENT_MODE,
                dp_enabled=dp_enabled, dp_clip_norm=dp_clip_norm,
                dp_noise_multiplier=dp_noise_multiplier, dp_delta=dp_delta,
                tls_enabled=FL_TLS_ENABLED, tls_cert=FL_TLS_CERT, tls_key=FL_TLS_KEY, tls_ca=FL_TLS_CA,
                aggregation_strategy=aggregation_strategy,
                secagg_enabled=secagg_enabled,
            )
        except ValueError:
            # e.g. secagg_enabled + aggregation_strategy="coordinate_median"
            # together -- this validation also happens earlier, at
            # experiment creation time (api/experiments.py), so reaching
            # this branch would mean an experiment was created before that
            # check existed or was bypassed some other way. Catching it
            # here too means a bad config fails visibly ("failed" status)
            # instead of leaving the experiment stuck in "running" forever
            # with a background task that silently died.
            logger.exception("Invalid FL configuration for experiment %s", experiment_id)
            with session_scope() as session:
                experiment = session.query(models.FederatedExperiment).get(experiment_id)
                experiment.status = "failed"
            return

        try:
            trainer.spawn()
        except Exception:
            logger.exception("Failed to spawn FL collaborator processes")
            with session_scope() as session:
                experiment = session.query(models.FederatedExperiment).get(experiment_id)
                experiment.status = "failed"
            return

        try:
            def _persist_round(round_idx: int, metrics: dict) -> None:
                with session_scope() as session:
                    tr = models.TrainingRound(
                        experiment_id=experiment_id,
                        round_number=round_idx,
                        status="completed",
                        metrics=metrics,
                    )
                    session.add(tr)
                if on_round_complete:
                    on_round_complete(round_idx, metrics)

            result = trainer.run(
                rounds=rounds, on_round_complete=_persist_round, seed=experiment_id or 0,
                client_timeout=FL_CLIENT_TIMEOUT_SECONDS, min_quorum_fraction=FL_MIN_QUORUM_FRACTION,
            )

            import json
            with session_scope() as session:
                experiment = session.query(models.FederatedExperiment).get(experiment_id)
                experiment.status = "completed"
                experiment.global_model_json = json.dumps(result["global_model"])

                if dp_enabled and result["history"]:
                    # Worst-case (max) epsilon across the 3 collaborators --
                    # each hospital's accountant tracks its OWN dataset's
                    # privacy spend independently; we report the largest so
                    # this single number is a valid upper bound for every
                    # participant, not an average that understates the
                    # weakest one.
                    last_round_clients = result["history"][-1]["per_client"]
                    epsilons = [
                        c["dp"]["epsilon_spent"] for c in last_round_clients
                        if c.get("dp") and c["dp"].get("epsilon_spent") is not None
                    ]
                    if epsilons:
                        experiment.dp_epsilon_spent = max(epsilons)
        except Exception:
            logger.exception("FL training run failed for experiment %s", experiment_id)
            with session_scope() as session:
                experiment = session.query(models.FederatedExperiment).get(experiment_id)
                experiment.status = "failed"
        finally:
            trainer.shutdown()

    @staticmethod
    def predict(experiment: models.FederatedExperiment, patient: dict) -> dict:
        """Real-time inference using the persisted global model from a
        completed run. `patient` is raw feature values; standardization
        uses the same federated mean/std the model was trained with.
        """
        import json

        if not experiment.global_model_json:
            raise ValueError("Experiment has no trained model yet (status must be 'completed')")

        model = json.loads(experiment.global_model_json)
        w = np.array(model["weights"])
        b = model["bias"]
        mean = np.array(model["mean"])
        std = np.array(model["std"])

        x = np.array([
            float(patient["age"]),
            1.0 if str(patient["gender"]).strip().upper() == "F" else 0.0,
            float(patient["fever"]),
            float(patient["cough"]),
            float(patient["oxygen_level"]),
            float(patient["blood_pressure"]),
        ])
        x_std = (x - mean) / std
        logit = float(x_std @ w + b)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return {
            "hospitalization_risk": round(float(prob), 4),
            "recommendation": "hospitalization" if prob >= 0.5 else "home_care",
        }
