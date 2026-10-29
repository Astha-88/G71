from __future__ import annotations

import asyncio
import json
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from db.database import get_session
from db import models
from services.openfl_service import OpenFLService


router = APIRouter()

# In-memory registry of live listeners per experiment, so a WebSocket
# client watching an in-progress run gets each round pushed to it as soon
# as the background job finishes it -- rather than polling GET /rounds on
# a timer, which is what the dashboard had to do before this existed.
_listeners: Dict[int, list] = {}


class ExperimentCreate(BaseModel):
    name: str
    dataset_id: int
    requester_id: int
    rounds: int = 5
    fedprox_mu: float = 0.05  # 0.0 = plain FedAvg; >0 penalizes client drift under non-IID data
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0  # larger = more private, more accuracy cost (see fl/dp_accountant.py)
    dp_delta: float = 1e-5
    aggregation_strategy: str = "fedavg"  # or "coordinate_median" for robustness against a malicious collaborator
    secagg_enabled: bool = False  # mutually exclusive with aggregation_strategy="coordinate_median"


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    status: str
    rounds: int


class PredictRequest(BaseModel):
    age: float
    gender: str
    fever: int
    cough: int
    oxygen_level: float
    blood_pressure: float


@router.get("/", response_model=list[ExperimentOut])
def list_experiments(requester_id: int, session: Session = Depends(get_session)) -> list[models.FederatedExperiment]:
    return (
        session.query(models.FederatedExperiment)
        .filter(models.FederatedExperiment.requester_id == requester_id)
        .order_by(models.FederatedExperiment.id.desc())
        .all()
    )


@router.post("/create", response_model=ExperimentOut)
def create_experiment(
    payload: ExperimentCreate, session: Session = Depends(get_session)
) -> models.FederatedExperiment:
    dataset = session.query(models.Dataset).get(payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if payload.secagg_enabled and payload.aggregation_strategy == "coordinate_median":
        raise HTTPException(
            status_code=400,
            detail="secagg_enabled and aggregation_strategy='coordinate_median' are mutually "
                   "exclusive: SecAgg hides individual updates from the aggregator; "
                   "coordinate-median aggregation requires seeing them.",
        )

    experiment = models.FederatedExperiment(
        name=payload.name,
        dataset_id=payload.dataset_id,
        requester_id=payload.requester_id,
        fl_rounds=payload.rounds,
        fedprox_mu=payload.fedprox_mu,
        dp_enabled=payload.dp_enabled,
        dp_clip_norm=payload.dp_clip_norm,
        dp_noise_multiplier=payload.dp_noise_multiplier,
        dp_delta=payload.dp_delta,
        aggregation_strategy=payload.aggregation_strategy,
        secagg_enabled=payload.secagg_enabled,
        status="created",
    )
    session.add(experiment)
    session.flush()
    return ExperimentOut(
        id=experiment.id,
        name=experiment.name,
        status=experiment.status,
        rounds=experiment.fl_rounds,
    )


def _push_to_listeners(experiment_id: int, message: dict) -> None:
    for queue in _listeners.get(experiment_id, []):
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            pass


@router.post("/start/{experiment_id}")
async def start_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> dict:
    """Kicks off training and returns immediately -- training itself
    (spawning collaborator processes, running real HTTP rounds) used to
    run synchronously inside this request handler, which meant an HTTP
    client had to hold a connection open for the entire multi-round run.
    It now runs in a background task with its own DB session; poll
    GET /experiments/{id}/rounds or open the WebSocket below for progress.

    NOTE: this endpoint must be `async def`, not a plain sync `def`. FastAPI
    runs sync path functions in a worker thread pool, where
    asyncio.get_running_loop() always raises (there's no event loop bound
    to that thread) -- an earlier version of this endpoint was sync and
    silently ended up always falling back to pushing into the WebSocket
    queue directly from a background thread, which is unsafe against
    asyncio.Queue and would intermittently fail to wake up a listening
    client. As `async def`, this coroutine runs on the real event loop, so
    the captured `loop` below is the one the WebSocket handler is also
    running on, and call_soon_threadsafe works as intended.
    """
    experiment = session.query(models.FederatedExperiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.status == "running":
        raise HTTPException(status_code=400, detail="Experiment already running")

    loop = asyncio.get_running_loop()

    def on_round_complete(round_idx: int, metrics: dict) -> None:
        # Called from the background task's worker thread -- must hop back
        # onto the event loop thread before touching the asyncio.Queue.
        message = {"type": "round_complete", "round": round_idx, "metrics": metrics}
        loop.call_soon_threadsafe(_push_to_listeners, experiment_id, message)

    background_tasks.add_task(
        OpenFLService.run_experiment_sync, experiment_id, on_round_complete
    )
    return {"status": "started", "experiment_id": experiment_id}


@router.websocket("/{experiment_id}/ws")
async def experiment_progress_ws(websocket: WebSocket, experiment_id: int) -> None:
    """Streams {"type": "round_complete", "round": N, "metrics": {...}}
    messages as each federated round finishes. Closes automatically once
    the experiment's status leaves "running" (checked via a short poll
    loop alongside the push queue, so a client that connects after
    completion still gets a clean close rather than hanging forever).
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _listeners.setdefault(experiment_id, []).append(queue)
    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                # Periodic keepalive / status check instead of blocking
                # forever if the run already finished before we connected.
                from db.database import session_scope
                with session_scope() as session:
                    experiment = session.query(models.FederatedExperiment).get(experiment_id)
                    status = experiment.status if experiment else "unknown"
                if status in ("completed", "failed"):
                    await websocket.send_json({"type": "status", "status": status})
                    break
    except WebSocketDisconnect:
        pass
    finally:
        _listeners.get(experiment_id, []).remove(queue)


@router.get("/{experiment_id}/rounds")
def get_rounds(experiment_id: int, session: Session = Depends(get_session)) -> list[dict]:
    rounds = (
        session.query(models.TrainingRound)
        .filter(models.TrainingRound.experiment_id == experiment_id)
        .order_by(models.TrainingRound.round_number)
        .all()
    )
    return [
        {
            "round": r.round_number,
            "status": r.status,
            "metrics": r.metrics,
            "created_at": r.created_at,
        }
        for r in rounds
    ]


@router.get("/{experiment_id}/status")
def get_status(experiment_id: int, session: Session = Depends(get_session)) -> dict:
    experiment = session.query(models.FederatedExperiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "id": experiment.id,
        "status": experiment.status,
        "rounds": experiment.fl_rounds,
        "fedprox_mu": experiment.fedprox_mu,
        "has_trained_model": experiment.global_model_json is not None,
        "dp_enabled": experiment.dp_enabled,
        "dp_clip_norm": experiment.dp_clip_norm if experiment.dp_enabled else None,
        "dp_noise_multiplier": experiment.dp_noise_multiplier if experiment.dp_enabled else None,
        "dp_delta": experiment.dp_delta if experiment.dp_enabled else None,
        "dp_epsilon_spent": experiment.dp_epsilon_spent,
        "aggregation_strategy": experiment.aggregation_strategy,
        "secagg_enabled": experiment.secagg_enabled,
    }


@router.post("/{experiment_id}/predict")
def predict(
    experiment_id: int, payload: PredictRequest, session: Session = Depends(get_session)
) -> dict:
    """Real-time inference against the persisted global model from a
    completed federated run -- the point of training a model at all,
    rather than just reporting metrics and discarding it (which is what
    the previous version of this service did).
    """
    experiment = session.query(models.FederatedExperiment).get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.status != "completed" or not experiment.global_model_json:
        raise HTTPException(status_code=400, detail="Experiment has no completed trained model yet")

    try:
        return OpenFLService.predict(experiment, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
