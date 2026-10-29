from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_session
from db import models
from services.access_service import AccessService
from services.research_project_service import ResearchProjectService
from services.openfl_service import OpenFLService
import logging

from api.errors import raise_for

logger = logging.getLogger(__name__)


router = APIRouter()


# ---------------------------------------------------------------------------
# Requests (backward-compatible surface: dataset_id + requester_id + purpose
# + FL protocol config, same shape the existing frontend already sends).
# Internally this now ALWAYS goes through ResearchProjectService, because
# every DatasetAccessRequest belongs to a project -- a direct
# single-hospital request is just hospital_scope="specific" with one
# hospital under the hood.
# ---------------------------------------------------------------------------

class AccessRequestCreate(BaseModel):
    dataset_id: int
    requester_id: int
    purpose: str | None = None
    fl_rounds: int = 5
    fedprox_mu: float = 0.05
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    dp_delta: float = 1e-5
    aggregation_strategy: str = "fedavg"
    secagg_enabled: bool = False


class AccessApprovalIn(BaseModel):
    request_id: int
    approver_id: int
    decision: str  # approved | rejected
    comment: str | None = None


class CancelRequestIn(BaseModel):
    request_id: int
    researcher_id: int


class AccessRequestOut(BaseModel):
    id: int
    dataset_id: int
    requester_id: int
    project_id: int
    status: str
    purpose: str | None = None
    fl_rounds: int
    fedprox_mu: float
    dp_enabled: bool
    aggregation_strategy: str
    secagg_enabled: bool


@router.post("/request")
def request_dataset_access(payload: AccessRequestCreate, session: Session = Depends(get_session)) -> dict:
    """Backward-compatible single-hospital request. Internally creates a
    one-hospital ResearchProject (hospital_scope="specific") -- see
    services/research_project_service.py. This is what makes even the
    existing frontend's plain "request access to dataset X" flow subject
    to the one-active-project rule and hospital blocking, without the
    frontend needing to know about projects explicitly.
    """
    ds = session.query(models.Dataset).get(payload.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    user = session.query(models.User).get(payload.requester_id)
    if not user:
        raise HTTPException(status_code=404, detail="Requester not found")

    disease = session.query(models.DiseaseRegistry).get(ds.disease_id)
    topic = disease.code if disease else ds.name

    proj_svc = ResearchProjectService(session)
    try:
        project = proj_svc.create_project(
            researcher_id=payload.requester_id,
            topic=topic,
            purpose=payload.purpose,
            hospital_scope="specific",
            hospital_ids=[ds.institution_id],
            fl_rounds=payload.fl_rounds, fedprox_mu=payload.fedprox_mu,
            dp_enabled=payload.dp_enabled, dp_clip_norm=payload.dp_clip_norm,
            dp_noise_multiplier=payload.dp_noise_multiplier, dp_delta=payload.dp_delta,
            aggregation_strategy=payload.aggregation_strategy, secagg_enabled=payload.secagg_enabled,
        )
    except ValueError as e:
        raise_for(e)

    req = project.requests[0]
    return {
        "request_id": req.id,
        "project_id": project.id,
        "status": req.status,
        "blockchain_tx_id": req.blockchain_tx_id,
    }


@router.post("/request/cancel")
def cancel_request(payload: CancelRequestIn, session: Session = Depends(get_session)) -> dict:
    svc = AccessService(session)
    try:
        req = svc.cancel_request(payload.request_id, payload.researcher_id)
    except ValueError as e:
        raise_for(e)
    return {"request_id": req.id, "status": req.status}


@router.post("/approve")
def approve_dataset_access(
    payload: AccessApprovalIn,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> dict:
    approver = session.query(models.User).get(payload.approver_id)
    if not approver:
        raise HTTPException(status_code=404, detail="Approver not found")

    if payload.decision not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="decision must be approved or rejected")

    svc = AccessService(session)
    try:
        req = svc.approve(payload.request_id, payload.approver_id, payload.decision, payload.comment)
    except ValueError as e:
        raise_for(e)

    # Everything below is a SIDE EFFECT of the approval, not the approval
    # itself: the vote and status change are already flushed. It runs in a
    # SAVEPOINT so a failure here (project-status sync, creating the experiment
    # row) cannot roll the approval back with it -- the hospital's decision is
    # the thing that must never be silently lost.
    #
    # The training run is retriggered whenever the project's approved-hospital
    # set changes. Participants are resolved at RUN time from the project's
    # CURRENT approved requests (services/openfl_service.py), so the experiment
    # reflects every hospital approved as of now, not just this one request.
    triggered_experiment_id = None
    warning = None
    try:
        with session.begin_nested():
            ResearchProjectService(session).sync_project_status(req.project_id)
            if req.status == "approved":
                exp = models.FederatedExperiment(
                    name=f"Auto FL Experiment for project {req.project_id}",
                    dataset_id=req.dataset_id,
                    project_id=req.project_id,
                    requester_id=req.requester_id,
                    fl_rounds=req.fl_rounds,
                    fedprox_mu=req.fedprox_mu,
                    dp_enabled=req.dp_enabled,
                    dp_clip_norm=req.dp_clip_norm,
                    dp_noise_multiplier=req.dp_noise_multiplier,
                    dp_delta=req.dp_delta,
                    aggregation_strategy=req.aggregation_strategy,
                    secagg_enabled=req.secagg_enabled,
                    status="created",
                )
                session.add(exp)
                session.flush()
                triggered_experiment_id = exp.id
    except Exception:
        logger.exception("Post-approval follow-up failed for request %s", req.id)
        triggered_experiment_id = None
        warning = "The decision was recorded, but starting the follow-up step failed. Please contact support."

    request_id, request_status = req.id, req.status

    # COMMIT BEFORE scheduling the background task. get_session() commits in its
    # post-yield code, which in current FastAPI runs AFTER the response (and
    # Starlette's background tasks). The training job opens its OWN session, so
    # without this it can run before the experiment row is committed, find
    # nothing, and silently never train.
    session.commit()

    if triggered_experiment_id is not None:
        background_tasks.add_task(OpenFLService.run_experiment_sync, triggered_experiment_id, None)

    response = {"request_id": request_id, "status": request_status, "experiment_id": triggered_experiment_id}
    if warning:
        response["warning"] = warning
    return response


@router.get("/requests", response_model=list[AccessRequestOut])
def list_requests(requester_id: int, session: Session = Depends(get_session)) -> list[AccessRequestOut]:
    """List all requests for a given requester."""
    rows = (
        session.query(models.DatasetAccessRequest)
        .filter(models.DatasetAccessRequest.requester_id == requester_id)
        .order_by(models.DatasetAccessRequest.id.desc())
        .all()
    )
    return [
        AccessRequestOut(
            id=r.id,
            dataset_id=r.dataset_id,
            requester_id=r.requester_id,
            project_id=r.project_id,
            status=r.status,
            purpose=r.purpose,
            fl_rounds=r.fl_rounds,
            fedprox_mu=r.fedprox_mu,
            dp_enabled=r.dp_enabled,
            aggregation_strategy=r.aggregation_strategy,
            secagg_enabled=r.secagg_enabled,
        )
        for r in rows
    ]


@router.get("/requests/pending")
def list_pending_requests(institution_id: int, session: Session = Depends(get_session)) -> list[dict]:
    """Kept as its own endpoint (rather than folded into the generic
    by-institution listing below) since it's the hospital dashboard's
    default/most-used view and existing callers depend on this exact path.
    """
    return _list_institution_requests(institution_id, "pending", session)


@router.get("/requests/by-institution")
def list_institution_requests(
    institution_id: int, status: str = "pending", session: Session = Depends(get_session)
) -> list[dict]:
    """General-purpose listing for the hospital dashboard's pending /
    approved / rejected / blocked tabs (status=all returns every status).
    """
    return _list_institution_requests(institution_id, status, session)


def _list_institution_requests(institution_id: int, status: str, session: Session) -> list[dict]:
    q = (
        session.query(models.DatasetAccessRequest, models.Dataset, models.User)
        .join(models.Dataset, models.DatasetAccessRequest.dataset_id == models.Dataset.id)
        .join(models.User, models.DatasetAccessRequest.requester_id == models.User.id)
        .filter(models.Dataset.institution_id == institution_id)
    )
    if status != "all":
        q = q.filter(models.DatasetAccessRequest.status == status)
    q = q.order_by(models.DatasetAccessRequest.id.desc())

    results: list[dict] = []
    for req, ds, user in q.all():
        results.append(
            {
                "request_id": req.id,
                "project_id": req.project_id,
                "dataset_id": ds.id,
                "dataset_name": ds.name,
                "requester_id": user.id,
                "requester_email": user.email,
                "status": req.status,
                "purpose": req.purpose,
                "created_at": req.created_at.isoformat() if req.created_at else None,
                # The proposed FL protocol -- shown to the admin so
                # "approve" means approving a specific, visible set of
                # privacy/security settings, not a blank check.
                "fl_rounds": req.fl_rounds,
                "fedprox_mu": req.fedprox_mu,
                "dp_enabled": req.dp_enabled,
                "dp_noise_multiplier": req.dp_noise_multiplier if req.dp_enabled else None,
                "aggregation_strategy": req.aggregation_strategy,
                "secagg_enabled": req.secagg_enabled,
            }
        )
    return results
