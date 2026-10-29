from __future__ import annotations

from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from blockchain.fabric_client import FabricClient
from db import models
from services.audit_service import AuditService


class AccessService:
    """Access request + approval workflow with majority threshold."""

    def __init__(
        self,
        session: Session,
        fabric_client: Optional[FabricClient] = None,
    ) -> None:
        self.session = session
        self.fabric = fabric_client or FabricClient()
        self.audit = AuditService(session)

    def _active_block(self, institution_id: int, researcher_id: int) -> Optional[models.ResearcherBlock]:
        return (
            self.session.query(models.ResearcherBlock)
            .filter(
                models.ResearcherBlock.institution_id == institution_id,
                models.ResearcherBlock.researcher_id == researcher_id,
                models.ResearcherBlock.unblocked_at.is_(None),
            )
            .one_or_none()
        )

    def _notify(
        self,
        *,
        notification_type: str,
        message: str,
        request_id: Optional[int] = None,
        project_id: Optional[int] = None,
        recipient_institution_id: Optional[int] = None,
        recipient_user_id: Optional[int] = None,
    ) -> None:
        self.session.add(models.Notification(
            recipient_institution_id=recipient_institution_id,
            recipient_user_id=recipient_user_id,
            notification_type=notification_type,
            request_id=request_id,
            project_id=project_id,
            message=message,
        ))

    def create_request(
        self,
        dataset_id: int,
        requester_id: int,
        purpose: str | None,
        project_id: int,
        fl_rounds: int = 5,
        fedprox_mu: float = 0.05,
        dp_enabled: bool = False,
        dp_clip_norm: float = 1.0,
        dp_noise_multiplier: float = 1.0,
        dp_delta: float = 1e-5,
        aggregation_strategy: str = "fedavg",
        secagg_enabled: bool = False,
    ) -> models.DatasetAccessRequest:
        dataset = self.session.query(models.Dataset).get(dataset_id)
        if not dataset:
            raise ValueError("dataset_not_found")

        # A hospital that has blocked this researcher must never even
        # receive a new request from them -- checked here, not just at
        # the UI layer, so "request from all eligible hospitals" and
        # direct API calls are both covered.
        if self._active_block(dataset.institution_id, requester_id):
            raise ValueError("researcher_blocked")

        existing_pending = (
            self.session.query(models.DatasetAccessRequest)
            .filter(
                models.DatasetAccessRequest.dataset_id == dataset_id,
                models.DatasetAccessRequest.requester_id == requester_id,
                models.DatasetAccessRequest.status == "pending",
            )
            .one_or_none()
        )
        if existing_pending:
            raise ValueError("duplicate_pending_request")

        if secagg_enabled and aggregation_strategy == "coordinate_median":
            raise ValueError("secagg_coordinate_median_conflict")

        req = models.DatasetAccessRequest(
            dataset_id=dataset_id,
            requester_id=requester_id,
            project_id=project_id,
            status="pending",
            purpose=purpose,
            fl_rounds=fl_rounds,
            fedprox_mu=fedprox_mu,
            dp_enabled=dp_enabled,
            dp_clip_norm=dp_clip_norm,
            dp_noise_multiplier=dp_noise_multiplier,
            dp_delta=dp_delta,
            aggregation_strategy=aggregation_strategy,
            secagg_enabled=secagg_enabled,
        )
        self.session.add(req)
        try:
            self.session.flush()
        except IntegrityError:
            # The application-level check above is a check-then-insert race
            # -- this is the actual backstop (a Postgres partial unique
            # index, see db/models.py's DatasetAccessRequest), for
            # concurrent calls that both passed the check before either
            # committed (e.g. two near-simultaneous "request all eligible
            # hospitals" operations, or a rapid double-click).
            self.session.rollback()
            raise ValueError("duplicate_pending_request")

        tx = self.fabric.request_dataset_access(
            {"request_id": str(req.id), "dataset_id": str(dataset_id), "requester_id": str(requester_id)}
        )
        req.blockchain_tx_id = tx

        self.audit.append_event(
            actor=str(requester_id),
            action="request_dataset_access",
            entity_type="dataset_access_request",
            entity_id=str(req.id),
            event_metadata={"dataset_id": dataset_id, "purpose": purpose, "project_id": project_id},
        )
        self._notify(
            notification_type="ACCESS_REQUEST",
            message=f"New research access request #{req.id} for dataset '{dataset.name}'.",
            request_id=req.id,
            project_id=project_id,
            recipient_institution_id=dataset.institution_id,
        )
        return req

    def approve(self, request_id: int, approver_id: int, decision: str, comment: str | None = None) -> models.DatasetAccessRequest:
        req = self.session.query(models.DatasetAccessRequest).get(request_id)
        if not req:
            raise ValueError("request_not_found")
        if req.status != "pending":
            raise ValueError("request_not_pending")

        approver = self.session.query(models.User).get(approver_id)
        if not approver or approver.role != "hospital_admin":
            raise ValueError("approver_not_authorized")

        dataset = self.session.query(models.Dataset).get(req.dataset_id)
        if not dataset or approver.institution_id != dataset.institution_id:
            # Only admins at the institution that owns the dataset may vote.
            # Without this check any authenticated user could approve access
            # to data they have no custodial relationship to.
            raise ValueError("approver_not_authorized")

        # Defense in depth: even if the request somehow stayed "pending"
        # while a block was recorded in the meantime (e.g. a second admin
        # blocked the researcher between page load and this admin's click),
        # a blocked researcher can never be approved.
        if self._active_block(dataset.institution_id, req.requester_id):
            raise ValueError("approver_researcher_blocked")

        # Record local approval vote
        existing = (
            self.session.query(models.AccessApproval)
            .filter(models.AccessApproval.request_id == request_id, models.AccessApproval.approver_id == approver_id)
            .one_or_none()
        )
        if existing:
            raise ValueError("already_voted")

        vote = models.AccessApproval(
            request_id=request_id,
            approver_id=approver_id,
            decision=decision,
            comment=comment,
        )
        self.session.add(vote)
        self.session.flush()

        # Call chaincode (mocked for now)
        if decision == "approved":
            tx = self.fabric.approve_access({"request_id": str(request_id), "approver_id": str(approver_id)})
            self.audit.append_event(
                actor=str(approver_id),
                action="approve_access",
                entity_type="dataset_access_request",
                entity_id=str(request_id),
                event_metadata={"tx_id": tx},
            )
        else:
            tx = self.fabric.reject_access({"request_id": str(request_id), "approver_id": str(approver_id)})
            req.status = "rejected"
            self.audit.append_event(
                actor=str(approver_id),
                action="reject_access",
                entity_type="dataset_access_request",
                entity_id=str(request_id),
                event_metadata={"tx_id": tx},
            )
            self._notify(
                notification_type="REQUEST_REJECTED",
                message=f"Your access request #{request_id} was not approved.",
                request_id=request_id,
                project_id=req.project_id,
                recipient_user_id=req.requester_id,
            )
            return req

        # Approval threshold: with the ownership check above, every voter is
        # now an admin at the institution that actually owns the dataset, so
        # a single authorized approval is sufficient. (The previous "2 of 3
        # hospitals" rule assumed unrelated institutions voted on each
        # other's data, which the ownership check above makes impossible —
        # keeping that threshold would have made approval unreachable.)
        approvals = (
            self.session.query(models.AccessApproval)
            .filter(models.AccessApproval.request_id == request_id, models.AccessApproval.decision == "approved")
            .count()
        )
        if approvals >= 1:
            req.status = "approved"
            self.audit.append_event(
                actor="system",
                action="grant_access",
                entity_type="dataset_access_request",
                entity_id=str(request_id),
                event_metadata={"approvals": approvals},
            )
            self._notify(
                notification_type="REQUEST_APPROVED",
                message=f"Your access request #{request_id} was approved.",
                request_id=request_id,
                project_id=req.project_id,
                recipient_user_id=req.requester_id,
            )

        return req

    def cancel_request(self, request_id: int, researcher_id: int) -> models.DatasetAccessRequest:
        """A researcher withdrawing their own still-pending request. Only
        the requester themselves can cancel it -- not any other
        researcher, and not a hospital admin (that's a rejection, a
        different transition with different meaning).
        """
        req = self.session.query(models.DatasetAccessRequest).get(request_id)
        if not req:
            raise ValueError("request_not_found")
        if req.requester_id != researcher_id:
            raise ValueError("not_authorized")
        if req.status != "pending":
            raise ValueError("request_not_pending")

        req.status = "cancelled"
        self.audit.append_event(
            actor=str(researcher_id),
            action="cancel_access_request",
            entity_type="dataset_access_request",
            entity_id=str(request_id),
        )
        return req

    def block_researcher(
        self, institution_id: int, researcher_id: int, blocked_by_user_id: int, reason: str | None = None
    ) -> models.ResearcherBlock:
        """Hospital-scoped block: only affects requests to THIS
        institution -- see ResearcherBlock's docstring. Policy for
        existing pending requests: they transition to "blocked" (a
        distinct terminal status from "rejected", so a researcher can
        tell "a human reviewed and said no" from "this institution
        blocked me" without exposing WHY they were blocked).
        """
        blocker = self.session.query(models.User).get(blocked_by_user_id)
        if not blocker or blocker.role != "hospital_admin" or blocker.institution_id != institution_id:
            raise ValueError("not_authorized")

        if self._active_block(institution_id, researcher_id):
            raise ValueError("already_blocked")

        block = models.ResearcherBlock(
            institution_id=institution_id,
            researcher_id=researcher_id,
            blocked_by_user_id=blocked_by_user_id,
            reason=reason,
        )
        self.session.add(block)

        # Policy: cancel this researcher's PENDING requests to THIS
        # institution's datasets only -- other institutions' requests
        # from the same researcher are entirely untouched (block is
        # hospital-scoped, not global).
        pending_reqs = (
            self.session.query(models.DatasetAccessRequest)
            .join(models.Dataset, models.DatasetAccessRequest.dataset_id == models.Dataset.id)
            .filter(
                models.Dataset.institution_id == institution_id,
                models.DatasetAccessRequest.requester_id == researcher_id,
                models.DatasetAccessRequest.status == "pending",
            )
            .all()
        )
        for req in pending_reqs:
            req.status = "blocked"

        self.session.flush()
        self.audit.append_event(
            actor=str(blocked_by_user_id),
            action="block_researcher",
            entity_type="researcher_block",
            entity_id=str(block.id),
            event_metadata={
                "institution_id": institution_id,
                "researcher_id": researcher_id,
                "cancelled_request_ids": [r.id for r in pending_reqs],
                # Reason IS recorded in the audit trail (accountability for
                # the hospital's own decision) but this event is never
                # surfaced to the researcher -- see api/access.py.
                "reason": reason,
            },
        )
        return block

    def unblock_researcher(self, institution_id: int, researcher_id: int, unblocked_by_user_id: int) -> None:
        unblocker = self.session.query(models.User).get(unblocked_by_user_id)
        if not unblocker or unblocker.role != "hospital_admin" or unblocker.institution_id != institution_id:
            raise ValueError("not_authorized")

        block = self._active_block(institution_id, researcher_id)
        if not block:
            raise ValueError("not_blocked")

        from datetime import datetime, timezone
        block.unblocked_at = datetime.now(timezone.utc)
        self.session.flush()
        self.audit.append_event(
            actor=str(unblocked_by_user_id),
            action="unblock_researcher",
            entity_type="researcher_block",
            entity_id=str(block.id),
            event_metadata={"institution_id": institution_id, "researcher_id": researcher_id},
        )

    def is_approved(self, request_id: int) -> bool:
        req = self.session.query(models.DatasetAccessRequest).get(request_id)
        return bool(req and req.status == "approved")

