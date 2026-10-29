from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship

from db.database import Base


class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class Institution(Base, TimestampMixin):
    __tablename__ = "institutions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    type = Column(String(50), nullable=False)  # hospital, research, regulator
    location = Column(String(255), nullable=True)
    blockchain_org = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    users = relationship("User", back_populates="institution")
    datasets = relationship("Dataset", back_populates="institution")


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    role = Column(
        String(50),
        nullable=False,
    )  # researcher, hospital_admin, regulator
    institution_id = Column(Integer, ForeignKey("institutions.id"), nullable=True)

    institution = relationship("Institution", back_populates="users")


class DiseaseRegistry(Base, TimestampMixin):
    __tablename__ = "disease_registry"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    disease_id = Column(Integer, ForeignKey("disease_registry.id"), nullable=False)
    institution_id = Column(Integer, ForeignKey("institutions.id"), nullable=False)
    storage_path = Column(String(512), nullable=False)
    storage_type = Column(String(50), default="minio", nullable=False)
    dataset_hash = Column(String(128), nullable=False)
    encryption_key_id = Column(String(128), nullable=True)

    disease = relationship("DiseaseRegistry")
    institution = relationship("Institution", back_populates="datasets")
    files = relationship("DatasetFile", back_populates="dataset")

    __table_args__ = (
        Index("ix_datasets_institution", "institution_id"),
        Index("ix_datasets_disease", "disease_id"),
    )


class DatasetFile(Base, TimestampMixin):
    __tablename__ = "dataset_files"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(128), nullable=False)
    format = Column(String(32), nullable=False)  # csv, parquet

    dataset = relationship("Dataset", back_populates="files")


class DatasetAccessRequest(Base, TimestampMixin):
    __tablename__ = "dataset_access_requests"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # A request always belongs to exactly one ResearchProject -- this is
    # what lets "request from all eligible hospitals" create N independent
    # hospital-scoped requests that are still understood as ONE research
    # effort (one active project can have several of these, one per
    # participating hospital), rather than N unrelated top-level requests.
    project_id = Column(Integer, ForeignKey("research_projects.id"), nullable=False)
    status = Column(
        String(32),
        default="pending",
        nullable=False,
    )  # pending, approved, rejected, cancelled, blocked -- see REQUEST_TRANSITIONS
    purpose = Column(Text, nullable=True)
    blockchain_tx_id = Column(String(128), nullable=True)

    # The researcher's proposed FL protocol -- mirrors FederatedExperiment's
    # config fields, but lives here so the hospital admin can see and
    # evaluate the actual privacy/security settings as part of what
    # they're approving, not just the free-text "purpose". This is what
    # gets used to construct the FederatedExperiment on approval, closing
    # a real gap: previously the auto-trigger on approval always used
    # hardcoded defaults (dp_enabled=False, secagg_enabled=False, plain
    # fedavg) regardless of what DP-SGD, mTLS-backed SecAgg, or robust
    # aggregation this codebase actually supports -- none of that was
    # reachable from the product flow at all.
    fl_rounds = Column(Integer, default=5, nullable=False)
    fedprox_mu = Column(Float, default=0.05, nullable=False)
    dp_enabled = Column(Boolean, default=False, nullable=False)
    dp_clip_norm = Column(Float, default=1.0, nullable=False)
    dp_noise_multiplier = Column(Float, default=1.0, nullable=False)
    dp_delta = Column(Float, default=1e-5, nullable=False)
    aggregation_strategy = Column(String(32), default="fedavg", nullable=False)
    secagg_enabled = Column(Boolean, default=False, nullable=False)

    dataset = relationship("Dataset")
    requester = relationship("User")
    project = relationship("ResearchProject", back_populates="requests")

    __table_args__ = (
        Index("ix_access_requests_dataset", "dataset_id"),
        Index("ix_access_requests_project", "project_id"),
        Index("ix_access_requests_requester", "requester_id"),
        # Backstop for the duplicate-pending check in
        # services/access_service.py's create_request, which is otherwise a
        # check-then-insert race: two near-simultaneous calls (a rapid
        # double-click, or two concurrent "request all eligible hospitals"
        # operations) could both pass the application check before either
        # commits. A partial unique index makes Postgres itself reject the
        # second INSERT -- same pattern as one-active-project-per-researcher.
        Index(
            "ux_one_pending_request_per_dataset_researcher",
            "dataset_id",
            "requester_id",
            unique=True,
            postgresql_where=status == "pending",
        ),
    )


class AccessApproval(Base, TimestampMixin):
    __tablename__ = "access_approvals"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("dataset_access_requests.id"), nullable=False)
    approver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    decision = Column(String(32), nullable=False)  # approved, rejected
    comment = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("request_id", "approver_id", name="uq_request_approver"),
    )


class FederatedExperiment(Base, TimestampMixin):
    __tablename__ = "federated_experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # Which ResearchProject this run belongs to -- nullable for backward
    # compatibility with any experiment created before projects existed.
    # When set, this is the SOURCE OF TRUTH for which hospitals actually
    # participate in training (see services/openfl_service.py): the
    # collaborator list is derived from this project's APPROVED requests,
    # not from a hardcoded constant. dataset_id above is kept as "the
    # first/primary dataset" for backward-compat display purposes only --
    # it is NOT what determines which collaborators train.
    project_id = Column(Integer, ForeignKey("research_projects.id"), nullable=True)
    status = Column(
        String(32),
        default="created",
        nullable=False,
    )  # created, running, completed, failed
    fl_rounds = Column(Integer, default=5, nullable=False)
    mlflow_experiment_id = Column(String(128), nullable=True)
    blockchain_tx_id = Column(String(128), nullable=True)
    # Persisted final model from a real FedAvg/FedProx run (fl/aggregator.py):
    # {"weights": [...], "bias": ..., "mean": [...], "std": [...], "feature_order": [...]}
    # Populated once the experiment reaches "completed"; used by
    # POST /experiments/{id}/predict for real-time inference.
    global_model_json = Column(Text, nullable=True)
    fedprox_mu = Column(Float, default=0.0, nullable=False)
    # DP-SGD settings for this experiment. Privacy is actually enforced
    # INSIDE each collaborator (fl/collaborator_server.py), never here --
    # these columns just record what was requested/used, and
    # dp_epsilon_spent is filled in from what the collaborators' own
    # accountants reported once training completes (see
    # services/openfl_service.py). Epsilon is a per-collaborator quantity
    # (privacy belongs to each hospital's dataset separately); this stores
    # the max across collaborators, i.e. the worst-case bound for the run.
    dp_enabled = Column(Boolean, default=False, nullable=False)
    dp_clip_norm = Column(Float, default=1.0, nullable=False)
    dp_noise_multiplier = Column(Float, default=1.0, nullable=False)
    dp_delta = Column(Float, default=1e-5, nullable=False)
    dp_epsilon_spent = Column(Float, nullable=True)
    # "fedavg" (default) or "coordinate_median" -- see fl/aggregator.py's
    # docstring on aggregation_strategy for the robustness trade-off.
    # Measured: median resists a simulated malicious collaborator much
    # better than fedavg (accuracy 0.65 vs 0.38 with 1-of-3 malicious) but
    # doesn't fully recover clean performance with only 3 total clients.
    aggregation_strategy = Column(String(32), default="fedavg", nullable=False)
    # Secure Aggregation: mutually exclusive with aggregation_strategy=
    # "coordinate_median" (enforced in fl/aggregator.py's __init__ -- median
    # needs individual values; SecAgg exists to hide them). When true, the
    # aggregator never sees any collaborator's individual update, only the
    # sum -- verified to produce numerically identical results to plain
    # FedAvg (matched to 1e-14) while never exposing an individual update.
    secagg_enabled = Column(Boolean, default=False, nullable=False)

    dataset = relationship("Dataset")
    requester = relationship("User")


class ExperimentCollaborator(Base, TimestampMixin):
    __tablename__ = "experiment_collaborators"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("federated_experiments.id"), nullable=False)
    institution_id = Column(Integer, ForeignKey("institutions.id"), nullable=False)
    role = Column(String(32), nullable=False)  # hospital, researcher

    __table_args__ = (
        UniqueConstraint("experiment_id", "institution_id", name="uq_experiment_institution"),
    )


class TrainingRound(Base, TimestampMixin):
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("federated_experiments.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    status = Column(String(32), default="running", nullable=False)
    metrics = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint("experiment_id", "round_number", name="uq_experiment_round"),
    )


class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(128), unique=True, nullable=False)
    actor = Column(String(255), nullable=False)
    action = Column(String(128), nullable=False)
    entity_type = Column(String(128), nullable=False)
    entity_id = Column(String(128), nullable=False)
    previous_hash = Column(String(128), nullable=True)
    current_hash = Column(String(128), nullable=False)
    # SQLAlchemy Declarative reserves attribute name `metadata`
    event_metadata = Column("metadata", JSON, nullable=True)


# States a ResearchProject can be in, and the transitions the service layer
# allows between them. Not enforced at the DB level (a CHECK constraint on
# valid transitions would need a trigger, not just a column constraint) --
# enforced in services/research_project_service.py instead. Listed here so
# the state machine has one canonical definition instead of being implied
# by scattered if-statements.
PROJECT_STATUSES = ("draft", "pending", "active", "running", "completed", "cancelled", "rejected")

# A project is "occupying" the researcher's one-active-project slot while
# in any of these states. Used both by the partial unique index below
# (enforced by Postgres itself, not just application code -- see
# ResearchProject.__table_args__) and by ResearchProjectService's
# pre-check (which exists purely to give a friendly 409 error message
# instead of letting a race lose to a raw IntegrityError).
ACTIVE_PROJECT_STATUSES = ("pending", "active", "running")

# Valid status transitions for DatasetAccessRequest, enforced in
# services/access_service.py. A request that's already approved/rejected/
# cancelled/blocked cannot transition again -- these are all terminal
# except "pending".
REQUEST_TRANSITIONS = {
    "pending": {"approved", "rejected", "cancelled", "blocked"},
    "approved": set(),
    "rejected": set(),
    "cancelled": set(),
    "blocked": set(),
}


class ResearchProject(Base, TimestampMixin):
    """Groups one or more hospital-specific DatasetAccessRequests under a
    single research effort. This is what makes "Lung Cancer Research
    across Hospital 1, 2, and 3" ONE thing a researcher can reason about
    and be limited to one-at-a-time, rather than three unrelated requests
    that happen to share a purpose string.
    """

    __tablename__ = "research_projects"

    id = Column(Integer, primary_key=True, index=True)
    researcher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic = Column(String(255), nullable=False)  # e.g. a disease code/name -- drives hospital eligibility
    purpose = Column(Text, nullable=True)
    status = Column(String(32), default="pending", nullable=False)

    researcher = relationship("User")
    requests = relationship("DatasetAccessRequest", back_populates="project")

    __table_args__ = (
        Index("ix_research_projects_researcher", "researcher_id"),
        # THE enforcement mechanism for "one active project per researcher"
        # -- a partial unique index that Postgres itself rejects a second
        # INSERT against, regardless of what the application layer checked
        # first. This is what makes the rule hold under concurrent request
        # creation (two simultaneous "create project" calls from the same
        # researcher can't both win a plain application-level check-then-
        # insert race; the DB constraint is the actual backstop).
        Index(
            "ux_one_active_project_per_researcher",
            "researcher_id",
            unique=True,
            postgresql_where=status.in_(ACTIVE_PROJECT_STATUSES),
        ),
    )


class ResearcherBlock(Base, TimestampMixin):
    """A hospital-scoped block: THIS institution refuses new requests from
    THIS researcher. Never global -- a researcher blocked by Hospital2 can
    still be approved by Hospital3, since hospitals are independent data
    owners (see services/access_service.py's block enforcement).
    """

    __tablename__ = "researcher_blocks"

    id = Column(Integer, primary_key=True, index=True)
    institution_id = Column(Integer, ForeignKey("institutions.id"), nullable=False)
    researcher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    blocked_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # Admin-only context for WHY -- never returned to the researcher (see
    # api/access.py: the researcher-facing status is a generic "this
    # institution is not accepting requests from you" with no reason).
    reason = Column(Text, nullable=True)
    unblocked_at = Column(DateTime, nullable=True)

    institution = relationship("Institution")
    researcher = relationship("User", foreign_keys=[researcher_id])
    blocked_by = relationship("User", foreign_keys=[blocked_by_user_id])

    __table_args__ = (
        Index("ix_researcher_blocks_institution", "institution_id"),
        Index("ix_researcher_blocks_researcher", "researcher_id"),
        # Only one ACTIVE block per (institution, researcher) pair --
        # unblocking then re-blocking creates a new row rather than
        # reusing/mutating the old one, preserving history. Partial on
        # unblocked_at IS NULL so a past, lifted block doesn't prevent a
        # new one from being recorded.
        Index(
            "ux_active_block_per_institution_researcher",
            "institution_id",
            "researcher_id",
            unique=True,
            postgresql_where=unblocked_at.is_(None),
        ),
    )


class Notification(Base, TimestampMixin):
    """Persistent, not frontend-only -- must survive a page refresh, a different
    device, or the recipient simply not being online when it was created.
    Created by AccessService on request submission (recipient = the dataset's
    owning institution) and on approve/reject (recipient = the researcher).
    Deliberately NOT created on researcher-blocking: a researcher must never be
    told a hospital blocked them (see api/errors.py's generic block message).
    """

    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    # Exactly one of these is set: an institution-wide notice (any admin at
    # that hospital should see it) or a specific user's notice.
    recipient_institution_id = Column(Integer, ForeignKey("institutions.id"), nullable=True)
    recipient_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    notification_type = Column(String(64), nullable=False)  # ACCESS_REQUEST, REQUEST_APPROVED, REQUEST_REJECTED
    request_id = Column(Integer, ForeignKey("dataset_access_requests.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("research_projects.id"), nullable=True)
    message = Column(Text, nullable=False)
    read_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_notifications_institution", "recipient_institution_id"),
        Index("ix_notifications_user", "recipient_user_id"),
    )

