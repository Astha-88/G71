from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_session
from db import models

router = APIRouter()


def _serialize(n: models.Notification) -> dict:
    return {
        "id": n.id,
        "type": n.notification_type,
        "message": n.message,
        "request_id": n.request_id,
        "project_id": n.project_id,
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "read": n.read_at is not None,
    }


@router.get("/institution")
def institution_notifications(
    institution_id: int, admin_user_id: int, unread_only: bool = False,
    session: Session = Depends(get_session),
) -> list[dict]:
    """A hospital admin's inbox. Verifies the caller is a hospital_admin AT THIS
    institution (identity derived from the DB row, never from the URL alone),
    otherwise any admin could read another hospital's notifications -- the same
    IDOR guard used by api/hospitals.py's blocked-researchers listing.
    """
    admin = session.query(models.User).get(admin_user_id)
    if not admin or admin.role != "hospital_admin" or admin.institution_id != institution_id:
        raise HTTPException(status_code=403, detail="Not authorized for this action")

    q = session.query(models.Notification).filter(
        models.Notification.recipient_institution_id == institution_id
    )
    if unread_only:
        q = q.filter(models.Notification.read_at.is_(None))
    return [_serialize(n) for n in q.order_by(models.Notification.id.desc()).limit(100).all()]


@router.get("/mine")
def my_notifications(user_id: int, unread_only: bool = False, session: Session = Depends(get_session)) -> list[dict]:
    user = session.query(models.User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    q = session.query(models.Notification).filter(models.Notification.recipient_user_id == user_id)
    if unread_only:
        q = q.filter(models.Notification.read_at.is_(None))
    return [_serialize(n) for n in q.order_by(models.Notification.id.desc()).limit(100).all()]


@router.post("/{notification_id}/read")
def mark_read(notification_id: int, user_id: int, session: Session = Depends(get_session)) -> dict:
    """Only the actual recipient may mark a notification read: either the
    specific user it was addressed to, or a hospital_admin at the institution
    it was addressed to. Anyone else gets 404 (not 403) -- no confirming that a
    notification with that ID exists.
    """
    n = session.query(models.Notification).get(notification_id)
    user = session.query(models.User).get(user_id)
    if not n or not user:
        raise HTTPException(status_code=404, detail="Notification not found")

    is_recipient = (
        (n.recipient_user_id is not None and n.recipient_user_id == user.id)
        or (
            n.recipient_institution_id is not None
            and user.role == "hospital_admin"
            and user.institution_id == n.recipient_institution_id
        )
    )
    if not is_recipient:
        raise HTTPException(status_code=404, detail="Notification not found")

    if n.read_at is None:
        n.read_at = datetime.now(timezone.utc)
    return {"id": n.id, "read": True}
