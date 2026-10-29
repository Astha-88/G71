from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, EmailStr
from sqlalchemy.orm import Session

from db.database import get_session
from db import models


router = APIRouter()


ALLOWED_ROLES = {"researcher", "hospital_admin"}


class UserCreate(BaseModel):
    email: EmailStr
    name: str
    role: str
    institution_id: int | None = None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    name: str
    role: str
    institution_id: int | None


@router.post("/register", response_model=UserOut)
def register_user(payload: UserCreate, session: Session = Depends(get_session)) -> models.User:
    existing = (
        session.query(models.User)
        .filter(models.User.email == payload.email)
        .one_or_none()
    )
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    if payload.role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"role must be one of {sorted(ALLOWED_ROLES)}",
        )

    if payload.institution_id is not None:
        institution = session.query(models.Institution).get(payload.institution_id)
        if not institution:
            raise HTTPException(status_code=400, detail="Institution not found")

    user = models.User(
        email=payload.email,
        name=payload.name,
        role=payload.role,
        institution_id=payload.institution_id,
    )
    session.add(user)
    session.flush()
    return user


@router.get("/by-email", response_model=UserOut)
def get_user_by_email(email: EmailStr, session: Session = Depends(get_session)) -> models.User:
    """Look up an existing user by email.

    Prototype "login": no password is stored in this schema yet (see
    IMPLEMENTATION_ROADMAP.md security next-steps). The client should call
    this first; a 404 means the person needs to register instead. This
    replaces the old flow where the client trusted a role typed into a
    dropdown with nothing checked server-side.
    """
    user = (
        session.query(models.User)
        .filter(models.User.email == email)
        .one_or_none()
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

