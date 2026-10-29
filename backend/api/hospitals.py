from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_session
from db import models
from services.access_service import AccessService
from api.errors import raise_for

router = APIRouter()


class BlockIn(BaseModel):
    blocked_by_user_id: int
    reason: str | None = None


class UnblockIn(BaseModel):
    unblocked_by_user_id: int


@router.post("/{institution_id}/researchers/{researcher_id}/block")
def block_researcher(
    institution_id: int, researcher_id: int, payload: BlockIn, session: Session = Depends(get_session)
) -> dict:
    svc = AccessService(session)
    try:
        block = svc.block_researcher(institution_id, researcher_id, payload.blocked_by_user_id, payload.reason)
    except ValueError as e:
        raise_for(e)
    return {"block_id": block.id, "institution_id": institution_id, "researcher_id": researcher_id, "blocked": True}


@router.delete("/{institution_id}/researchers/{researcher_id}/block")
def unblock_researcher(
    institution_id: int, researcher_id: int, payload: UnblockIn, session: Session = Depends(get_session)
) -> dict:
    svc = AccessService(session)
    try:
        svc.unblock_researcher(institution_id, researcher_id, payload.unblocked_by_user_id)
    except ValueError as e:
        raise_for(e)
    return {"institution_id": institution_id, "researcher_id": researcher_id, "blocked": False}


@router.get("/{institution_id}/blocked-researchers")
def list_blocked_researchers(
    institution_id: int, admin_user_id: int, session: Session = Depends(get_session)
) -> list[dict]:
    """Admin-only view -- verifies the caller is actually a hospital_admin
    at THIS institution before returning anything, rather than trusting
    institution_id in the URL alone (that would let any admin at any
    hospital enumerate another hospital's blocked-researcher list, which
    is exactly the kind of IDOR this platform's existing approve()
    ownership check already guards against elsewhere).
    """
    admin = session.query(models.User).get(admin_user_id)
    if not admin or admin.role != "hospital_admin" or admin.institution_id != institution_id:
        raise HTTPException(status_code=403, detail="Not authorized for this action")

    blocks = (
        session.query(models.ResearcherBlock)
        .filter(
            models.ResearcherBlock.institution_id == institution_id,
            models.ResearcherBlock.unblocked_at.is_(None),
        )
        .order_by(models.ResearcherBlock.id.desc())
        .all()
    )
    results = []
    for b in blocks:
        researcher = session.query(models.User).get(b.researcher_id)
        results.append({
            "block_id": b.id,
            "researcher_id": b.researcher_id,
            "researcher_email": researcher.email if researcher else None,
            "blocked_at": b.created_at.isoformat() if b.created_at else None,
            # `reason` is intentionally included here -- this endpoint is
            # already admin-authenticated to THIS institution, so it's
            # fine for the hospital's own admins to see their own
            # institution's reason for blocking. It must never appear in
            # any researcher-facing endpoint (see api/errors.py's generic
            # "researcher_blocked" message).
            "reason": b.reason,
        })
    return results
