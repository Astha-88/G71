from __future__ import annotations

import hashlib
import io
import os
import uuid
from typing import Optional

from minio import Minio
from sqlalchemy.orm import Session

from db import models
from blockchain.fabric_client import FabricClient

# Mirrors the IN_DOCKER pattern already used for DATABASE_URL in
# scripts/seed_data.py and scripts/run_demo_fl.py, so this service behaves
# the same way whether it's run inside docker-compose or on the host.
MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "minio:9000" if os.getenv("IN_DOCKER") == "1" else "localhost:9000",
)
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"


class DatasetService:
    def __init__(
        self,
        session: Session,
        minio_client: Optional[Minio] = None,
        fabric_client: Optional[FabricClient] = None,
    ) -> None:
        self.session = session
        self.minio = minio_client or Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        self.fabric = fabric_client or FabricClient()
        self.bucket = "datasets"

        found = self.minio.bucket_exists(self.bucket)
        if not found:
            self.minio.make_bucket(self.bucket)

    def register_dataset(
        self,
        name: str,
        disease_code: str,
        institution_id: int,
        file_bytes: bytes,
        file_name: str,
        file_format: str = "csv",
    ) -> models.Dataset:
        """Registers a dataset from bytes already in memory -- the backend
        never needs local filesystem access to the source file. In the
        docker-compose deployment, these bytes arrive over HTTP from a
        hospital's own fl-hospitalN container via
        fl/register_dataset_client.py, which is the only process that ever
        had that file on disk in the first place (see that container's
        Dockerfile, which only COPYs in its own hospital's CSV).
        """
        disease = (
            self.session.query(models.DiseaseRegistry)
            .filter(models.DiseaseRegistry.code == disease_code)
            .one_or_none()
        )
        if not disease:
            disease = models.DiseaseRegistry(code=disease_code, name=disease_code)
            self.session.add(disease)
            self.session.flush()

        dataset_uuid = str(uuid.uuid4())
        object_name = f"{institution_id}/{dataset_uuid}.{file_format}"
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        self.minio.put_object(
            self.bucket,
            object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
        )

        dataset = models.Dataset(
            dataset_id=dataset_uuid,
            name=name,
            disease_id=disease.id,
            institution_id=institution_id,
            storage_path=object_name,
            dataset_hash=file_hash,
            storage_type="minio",
        )
        self.session.add(dataset)
        self.session.flush()

        ds_file = models.DatasetFile(
            dataset_id=dataset.id,
            file_name=file_name,
            file_path=object_name,
            file_hash=file_hash,
            format=file_format,
        )
        self.session.add(ds_file)

        self.fabric.register_dataset(
            {
                "dataset_id": dataset.dataset_id,
                "hospital_id": institution_id,
                "disease": disease_code,
                "dataset_hash": file_hash,
                "storage_path": object_name,
            }
        )

        return dataset

    def register_dataset_from_path(
        self,
        name: str,
        disease_code: str,
        institution_id: int,
        local_file_path: str,
        file_format: str = "csv",
    ) -> models.Dataset:
        """Convenience wrapper for LOCAL, NON-DOCKER development only, where
        there's one filesystem and one developer and no cross-institution
        isolation to protect. NOT used by the docker-compose deployment --
        scripts/seed_data.py skips calling this when IN_DOCKER=1 and prints
        instructions to run fl/register_dataset_client.py from inside each
        hospital's own container instead. If you find yourself calling
        this from inside a multi-tenant deployment, that's the bug the
        upload-based register_dataset() above exists to avoid.
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(local_file_path)
        with open(local_file_path, "rb") as f:
            file_bytes = f.read()
        return self.register_dataset(
            name=name,
            disease_code=disease_code,
            institution_id=institution_id,
            file_bytes=file_bytes,
            file_name=os.path.basename(local_file_path),
            file_format=file_format,
        )

