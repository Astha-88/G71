from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from db.database import get_session
from db import models


router = APIRouter()


class InstitutionCreate(BaseModel):
    name: str
    type: str  # hospital, research, regulator
    location: str | None = None
    blockchain_org: str | None = None


class InstitutionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    type: str
    location: str | None
    blockchain_org: str | None


@router.post("/register", response_model=InstitutionOut)
def register_institution(
    payload: InstitutionCreate, session: Session = Depends(get_session)
) -> models.Institution:
    existing = (
        session.query(models.Institution)
        .filter(models.Institution.name == payload.name)
        .one_or_none()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Institution already exists")

    inst = models.Institution(
        name=payload.name,
        type=payload.type,
        location=payload.location,
        blockchain_org=payload.blockchain_org,
    )
    session.add(inst)
    session.flush()
    return inst


@router.get("", response_model=list[InstitutionOut])
def list_institutions(session: Session = Depends(get_session)) -> list[models.Institution]:
    """Feeds the login page's institution picker with the institutions that
    actually exist in the DB, instead of the previous hardcoded dropdown
    ("AIIMS", "Johns Hopkins", ...) which didn't correspond to anything
    seed_data.py actually creates (Hospital1/2/3, ResearchInstitute1/2).
    """
    return session.query(models.Institution).order_by(models.Institution.name).all()

