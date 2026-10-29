from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_session
from db import models
from services.research_project_service import ResearchProjectService
from api.errors import raise_for

router = APIRouter()


class ProjectCreate(BaseModel):
    researcher_id: int
    topic: str
    purpose: Optional[str] = None
    hospital_scope: str  # "specific" | "multiple" | "all_eligible"
    hospital_ids: Optional[list[int]] = None
    fl_rounds: int = 5
    fedprox_mu: float = 0.05
    dp_enabled: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    dp_delta: float = 1e-5
    aggregation_strategy: str = "fedavg"
    secagg_enabled: bool = False


class AddHospitalIn(BaseModel):
    researcher_id: int
    hospital_id: int


def _request_summary(req: models.DatasetAccessRequest) -> dict:
    return {
        "request_id": req.id,
        "dataset_id": req.dataset_id,
        "institution_id": req.dataset.institution_id if req.dataset else None,
        "institution_name": req.dataset.institution.name if req.dataset and req.dataset.institution else None,
        "status": req.status,
    }


def _project_summary(project: models.ResearchProject) -> dict:
    return {
        "project_id": project.id,
        "researcher_id": project.researcher_id,
        "topic": project.topic,
        "purpose": project.purpose,
        "status": project.status,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "hospital_requests": [_request_summary(r) for r in project.requests],
    }


@router.post("")
def create_project(payload: ProjectCreate, session: Session = Depends(get_session)) -> dict:
    user = session.query(models.User).get(payload.researcher_id)
    if not user:
        raise HTTPException(status_code=404, detail="Researcher not found")

    svc = ResearchProjectService(session)
    try:
        project = svc.create_project(
            researcher_id=payload.researcher_id,
            topic=payload.topic,
            purpose=payload.purpose,
            hospital_scope=payload.hospital_scope,
            hospital_ids=payload.hospital_ids,
            fl_rounds=payload.fl_rounds, fedprox_mu=payload.fedprox_mu,
            dp_enabled=payload.dp_enabled, dp_clip_norm=payload.dp_clip_norm,
            dp_noise_multiplier=payload.dp_noise_multiplier, dp_delta=payload.dp_delta,
            aggregation_strategy=payload.aggregation_strategy, secagg_enabled=payload.secagg_enabled,
        )
    except ValueError as e:
        raise_for(e)
    return _project_summary(project)


@router.get("")
def list_projects(researcher_id: int, session: Session = Depends(get_session)) -> list[dict]:
    projects = (
        session.query(models.ResearchProject)
        .filter(models.ResearchProject.researcher_id == researcher_id)
        .order_by(models.ResearchProject.id.desc())
        .all()
    )
    return [_project_summary(p) for p in projects]


@router.get("/active")
def get_active_project(researcher_id: int, session: Session = Depends(get_session)):
    """Lets the frontend ask "do I already have an active project" up
    front, so it can explain why a new one can't be started BEFORE the
    researcher fills out an entire multi-step form only to hit a 409 at
    the end.
    """
    svc = ResearchProjectService(session)
    project = svc.get_active_project(researcher_id)
    return _project_summary(project) if project else None


@router.get("/eligible-hospitals")
def eligible_hospitals(topic: str, researcher_id: int, session: Session = Depends(get_session)) -> list[dict]:
    """Powers the hospital-scope selection step of the request flow --
    only hospitals that actually have a matching dataset AND haven't
    blocked this researcher are ever shown as selectable, so the
    researcher never even sees "Hospital 2" as an option to pick if it
    has blocked them (see db/models.py's ResearcherBlock: they're not
    told this is why -- it simply never appears as eligible).
    """
    svc = ResearchProjectService(session)
    datasets = svc.eligible_hospital_datasets(topic, researcher_id)
    return [
        {
            "institution_id": ds.institution_id,
            "institution_name": ds.institution.name,
            "dataset_id": ds.id,
            "dataset_name": ds.name,
        }
        for ds in datasets
    ]


@router.get("/{project_id}")
def get_project(project_id: int, session: Session = Depends(get_session)) -> dict:
    project = session.query(models.ResearchProject).get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Research project not found")
    return _project_summary(project)


@router.post("/{project_id}/access-requests")
def add_hospital_to_project(project_id: int, payload: AddHospitalIn, session: Session = Depends(get_session)) -> dict:
    svc = ResearchProjectService(session)
    try:
        req = svc.add_hospital_to_project(project_id, payload.researcher_id, payload.hospital_id)
    except ValueError as e:
        raise_for(e)
    return _request_summary(req)


@router.post("/{project_id}/request-all-eligible-hospitals")
def request_all_eligible(project_id: int, researcher_id: int, session: Session = Depends(get_session)) -> dict:
    """Adds every currently-eligible hospital NOT already requested under
    this project. Each addition is an independent DatasetAccessRequest --
    one hospital being ineligible/blocked never prevents the others from
    being added (matches the same "independent decisions" principle as
    project creation's all_eligible scope).
    """
    project = session.query(models.ResearchProject).get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Research project not found")
    if project.researcher_id != researcher_id:
        raise HTTPException(status_code=403, detail="Not authorized for this action")

    svc = ResearchProjectService(session)
    already_requested = {r.dataset.institution_id for r in project.requests if r.dataset}
    eligible = svc.eligible_hospital_datasets(project.topic, researcher_id)

    added, skipped = [], []
    for ds in eligible:
        if ds.institution_id in already_requested:
            continue
        try:
            req = svc.add_hospital_to_project(project_id, researcher_id, ds.institution_id)
            added.append(_request_summary(req))
        except ValueError as e:
            skipped.append({"institution_id": ds.institution_id, "reason": str(e)})

    return {"added": added, "skipped": skipped}


@router.post("/{project_id}/cancel")
def cancel_project(project_id: int, researcher_id: int, session: Session = Depends(get_session)) -> dict:
    svc = ResearchProjectService(session)
    try:
        project = svc.cancel_project(project_id, researcher_id)
    except ValueError as e:
        raise_for(e)
    return _project_summary(project)
