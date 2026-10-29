from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from db.database import get_session
from db import models
from services.dataset_service import DatasetService


router = APIRouter()


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: str
    name: str
    disease_code: str
    institution_id: int


@router.post("/register", response_model=DatasetOut)
async def register_dataset(
    name: str = Form(...),
    disease_code: str = Form(...),
    institution_id: int = Form(...),
    format: str = Form("csv"),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> DatasetOut:
    """Registers a dataset from an uploaded file, not a local path.

    This used to accept `local_file_path` as a JSON string and read it off
    the backend's own filesystem -- which only worked because backend/data
    was volume-mounted into the backend container, giving it read access
    to every hospital's raw CSV even though it only ever needed to *train*
    via the isolated fl-hospitalN containers. Now the backend never touches
    a local dataset file at all; the bytes have to be sent to it, and the
    only thing that sends them is fl/register_dataset_client.py running
    *inside* a hospital's own container (see docker-compose.yml) -- the one
    process that legitimately had that file on disk in the first place.
    """
    service = DatasetService(session)
    file_bytes = await file.read()

    dataset = service.register_dataset(
        name=name,
        disease_code=disease_code,
        institution_id=institution_id,
        file_bytes=file_bytes,
        file_name=file.filename or f"{name}.{format}",
        file_format=format,
    )

    return DatasetOut(
        id=dataset.id,
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        disease_code=disease_code,
        institution_id=dataset.institution_id,
    )


@router.get("", response_model=list[DatasetOut])
def list_datasets(session: Session = Depends(get_session)) -> list[DatasetOut]:
    datasets = session.query(models.Dataset).all()
    result: list[DatasetOut] = []
    for ds in datasets:
        result.append(
            DatasetOut(
                id=ds.id,
                dataset_id=ds.dataset_id,
                name=ds.name,
                disease_code=ds.disease.code,
                institution_id=ds.institution_id,
            )
        )
    return result

