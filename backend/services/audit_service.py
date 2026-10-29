from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from db import models


class AuditService:
    """Blockchain-style audit log with hash chaining.

    Each new event stores previous_hash and current_hash where:
      current_hash = SHA256(previous_hash + canonical_json(event_payload))
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def _get_last_hash(self) -> Optional[str]:
        last = (
            self.session.query(models.AuditLog)
            .order_by(models.AuditLog.id.desc())
            .limit(1)
            .one_or_none()
        )
        return last.current_hash if last else None

    @staticmethod
    def _canonical_json(data: Dict[str, Any]) -> str:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def append_event(
        self,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str,
        event_metadata: Optional[Dict[str, Any]] = None,
    ) -> models.AuditLog:
        prev = self._get_last_hash()
        payload: Dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "actor": actor,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": event_metadata or {},
        }
        canonical = self._canonical_json(payload)
        current = self._sha256((prev or "") + canonical)

        row = models.AuditLog(
            event_id=payload["event_id"],
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            previous_hash=prev,
            current_hash=current,
            event_metadata=event_metadata or {},
        )
        self.session.add(row)
        self.session.flush()
        return row

