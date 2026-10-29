from __future__ import annotations

from typing import List, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db import models
from services.access_service import AccessService
from services.audit_service import AuditService

HOSPITAL_SCOPES = ("specific", "multiple", "all_eligible")


class ResearchProjectService:
    """Groups hospital-specific access requests under one researcher
    project, and is where "one active project per researcher" and
    "hospital eligibility" actually live -- see db/models.py's
    ResearchProject docstring for why this exists instead of treating
    each hospital request as an independent top-level thing.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.audit = AuditService(session)

    def get_active_project(self, researcher_id: int) -> Optional[models.ResearchProject]:
        return (
            self.session.query(models.ResearchProject)
            .filter(
                models.ResearchProject.researcher_id == researcher_id,
                models.ResearchProject.status.in_(models.ACTIVE_PROJECT_STATUSES),
            )
            .one_or_none()
        )

    def _blocked_institution_ids(self, researcher_id: int) -> set:
        return {
            b.institution_id
            for b in self.session.query(models.ResearcherBlock)
            .filter(
                models.ResearcherBlock.researcher_id == researcher_id,
                models.ResearcherBlock.unblocked_at.is_(None),
            )
            .all()
        }

    @staticmethod
    def _resolve_topic_to_disease_code(topic: str, known_codes) -> Optional[str]:
        """Maps a free-text topic to exactly ONE canonical DiseaseRegistry code,
        or None if it is empty, unknown, or ambiguous.

        Strict on purpose. Every hospital now holds several diseases, so a vague
        topic ("e", "patient", "records") that fuzzily matched multiple diseases
        would let eligible_by_institution (one dataset per hospital, keyed by
        institution_id) silently keep an ARBITRARY one -- i.e. request the wrong
        disease. An ambiguous topic is treated as "no eligible hospitals", never
        guessed.
          1. exact match after normalising case, "-" and "_" ("COVID-19",
             "Lung_Cancer" -> COVID19, LUNG CANCER);
          2. otherwise an unambiguous substring match, only for >= 3 characters
             ("lung", "cancer" -> LUNG_CANCER; "e" matches several -> rejected).
        """
        t = " ".join(topic.strip().lower().replace("-", "").replace("_", " ").split())
        if not t:
            return None
        codes = sorted(set(c for c in known_codes if c))

        def norm(code: str) -> str:
            return code.lower().replace("_", " ")

        exact = [c for c in codes if norm(c) == t]
        if len(exact) == 1:
            return exact[0]
        if len(t) >= 3:
            partial = [c for c in codes if t in norm(c)]
            if len(partial) == 1:
                return partial[0]
        return None

    def eligible_hospital_datasets(self, topic: str, researcher_id: int) -> List[models.Dataset]:
        """A (hospital, dataset) pairing is eligible for a project topic if the
        topic resolves to ONE disease (see _resolve_topic_to_disease_code),
        the hospital owns a dataset for that disease, AND the hospital has not
        blocked this researcher. Returns the matching Dataset rows -- by
        construction at most one per institution.

        This is the ONE place eligibility is computed -- reused by both
        project creation and "request from all eligible hospitals" so they can
        never disagree with each other. Topic resolution deliberately considers
        ALL hospitals' diseases (including ones that blocked this researcher),
        so a block never changes how a topic is interpreted.
        """
        blocked = self._blocked_institution_ids(researcher_id)

        candidates = (
            self.session.query(models.Dataset)
            .join(models.DiseaseRegistry, models.Dataset.disease_id == models.DiseaseRegistry.id)
            .join(models.Institution, models.Dataset.institution_id == models.Institution.id)
            .filter(models.Institution.type == "hospital")
            .all()
        )

        resolved = self._resolve_topic_to_disease_code(
            topic, [ds.disease.code for ds in candidates if ds.disease and ds.disease.code]
        )
        if resolved is None:
            return []

        return [
            ds for ds in candidates
            if ds.institution_id not in blocked and ds.disease and ds.disease.code == resolved
        ]

    def create_project(
        self,
        researcher_id: int,
        topic: str,
        purpose: Optional[str],
        hospital_scope: str,
        hospital_ids: Optional[List[int]] = None,
        fl_rounds: int = 5,
        fedprox_mu: float = 0.05,
        dp_enabled: bool = False,
        dp_clip_norm: float = 1.0,
        dp_noise_multiplier: float = 1.0,
        dp_delta: float = 1e-5,
        aggregation_strategy: str = "fedavg",
        secagg_enabled: bool = False,
    ) -> models.ResearchProject:
        if hospital_scope not in HOSPITAL_SCOPES:
            raise ValueError("invalid_hospital_scope")

        # Friendly pre-check -- gives a clean 409 with a clear reason
        # instead of surfacing a raw IntegrityError to the caller. The
        # actual race-safety backstop is the partial unique index on
        # research_projects (see db/models.py); this check can still lose
        # a race to a concurrent request, which is exactly what that
        # index is for (caught below as IntegrityError).
        if self.get_active_project(researcher_id):
            raise ValueError("active_project_exists")

        if secagg_enabled and aggregation_strategy == "coordinate_median":
            raise ValueError("secagg_coordinate_median_conflict")

        eligible = self.eligible_hospital_datasets(topic, researcher_id)
        eligible_by_institution = {ds.institution_id: ds for ds in eligible}

        if hospital_scope == "all_eligible":
            selected_datasets = list(eligible_by_institution.values())
        else:
            if not hospital_ids:
                raise ValueError("no_hospitals_selected")
            selected_datasets = [
                eligible_by_institution[hid] for hid in hospital_ids if hid in eligible_by_institution
            ]
            if hospital_scope == "specific" and len(hospital_ids) > 1:
                raise ValueError("specific_scope_requires_one_hospital")

        if not selected_datasets:
            # Distinguish "nothing matches this topic at all" from "every
            # matching hospital has blocked you" -- the researcher gets a
            # generic reason either way (never told WHICH hospital blocked
            # them or that blocking is even the specific cause), but the
            # two ValueError codes let the API return a slightly more
            # useful message without leaking anything sensitive.
            if eligible_by_institution:
                raise ValueError("no_eligible_hospitals_selected")
            raise ValueError("no_eligible_hospitals_for_topic")

        project = models.ResearchProject(
            researcher_id=researcher_id,
            topic=topic,
            purpose=purpose,
            status="pending",
        )
        self.session.add(project)
        try:
            self.session.flush()
        except IntegrityError:
            self.session.rollback()
            raise ValueError("active_project_exists")

        access_svc = AccessService(self.session)
        created_requests = []
        skipped_blocked = 0
        for ds in selected_datasets:
            try:
                req = access_svc.create_request(
                    dataset_id=ds.id,
                    requester_id=researcher_id,
                    purpose=purpose,
                    project_id=project.id,
                    fl_rounds=fl_rounds,
                    fedprox_mu=fedprox_mu,
                    dp_enabled=dp_enabled,
                    dp_clip_norm=dp_clip_norm,
                    dp_noise_multiplier=dp_noise_multiplier,
                    dp_delta=dp_delta,
                    aggregation_strategy=aggregation_strategy,
                    secagg_enabled=secagg_enabled,
                )
                created_requests.append(req)
            except ValueError as exc:
                if str(exc) == "researcher_blocked":
                    # Shouldn't normally happen (eligible_hospital_datasets
                    # already filters blocked institutions out), but a
                    # block recorded in the split second between that
                    # query and this insert is possible -- skip silently
                    # rather than failing the whole project creation over
                    # one hospital's block.
                    skipped_blocked += 1
                    continue
                raise

        if not created_requests:
            self.session.rollback()
            raise ValueError("all_selected_hospitals_blocked")

        self.audit.append_event(
            actor=str(researcher_id),
            action="create_research_project",
            entity_type="research_project",
            entity_id=str(project.id),
            event_metadata={
                "topic": topic,
                "hospital_scope": hospital_scope,
                "n_requests": len(created_requests),
                "skipped_blocked": skipped_blocked,
            },
        )
        return project

    def add_hospital_to_project(
        self, project_id: int, researcher_id: int, hospital_id: int
    ) -> models.DatasetAccessRequest:
        """Adds one more hospital's request to an EXISTING project -- e.g.
        the researcher initially requested Hospital1 only, and later wants
        to add Hospital2 to the same Lung Cancer Research project rather
        than starting a second (which the one-active-project rule would
        block anyway).
        """
        project = self.session.query(models.ResearchProject).get(project_id)
        if not project:
            raise ValueError("project_not_found")
        if project.researcher_id != researcher_id:
            raise ValueError("not_authorized")
        if project.status not in models.ACTIVE_PROJECT_STATUSES:
            raise ValueError("project_not_active")

        eligible = self.eligible_hospital_datasets(project.topic, researcher_id)
        eligible_by_institution = {ds.institution_id: ds for ds in eligible}
        if hospital_id not in eligible_by_institution:
            raise ValueError("hospital_not_eligible")

        ds = eligible_by_institution[hospital_id]
        # Reuse the FL protocol config from an existing request under this
        # project, so "add a hospital" doesn't silently reset DP/SecAgg/etc
        # back to defaults.
        template = project.requests[0] if project.requests else None

        access_svc = AccessService(self.session)
        return access_svc.create_request(
            dataset_id=ds.id,
            requester_id=researcher_id,
            purpose=project.purpose,
            project_id=project.id,
            fl_rounds=template.fl_rounds if template else 5,
            fedprox_mu=template.fedprox_mu if template else 0.05,
            dp_enabled=template.dp_enabled if template else False,
            dp_clip_norm=template.dp_clip_norm if template else 1.0,
            dp_noise_multiplier=template.dp_noise_multiplier if template else 1.0,
            dp_delta=template.dp_delta if template else 1e-5,
            aggregation_strategy=template.aggregation_strategy if template else "fedavg",
            secagg_enabled=template.secagg_enabled if template else False,
        )

    def cancel_project(self, project_id: int, researcher_id: int) -> models.ResearchProject:
        project = self.session.query(models.ResearchProject).get(project_id)
        if not project:
            raise ValueError("project_not_found")
        if project.researcher_id != researcher_id:
            raise ValueError("not_authorized")
        if project.status not in models.ACTIVE_PROJECT_STATUSES:
            raise ValueError("project_not_active")

        for req in project.requests:
            if req.status == "pending":
                req.status = "cancelled"
        project.status = "cancelled"
        self.session.flush()

        self.audit.append_event(
            actor=str(researcher_id),
            action="cancel_research_project",
            entity_type="research_project",
            entity_id=str(project.id),
        )
        return project

    def sync_project_status(self, project_id: int) -> models.ResearchProject:
        """Recomputes a project's status from its requests' current
        statuses. Called after any approve/reject so the project-level
        status (what the researcher's dashboard shows as the headline
        state) stays consistent with the request-level detail, without
        duplicating that logic at every call site that touches a request.
        """
        project = self.session.query(models.ResearchProject).get(project_id)
        if not project or project.status in ("cancelled",):
            return project

        statuses = {r.status for r in project.requests}
        if "approved" in statuses:
            # At least one hospital on board -- the project can run,
            # regardless of other hospitals still pending/rejected/blocked.
            project.status = "active"
        elif statuses and statuses <= {"rejected", "blocked"}:
            # Every single hospital said no, in one way or another.
            project.status = "rejected"
        elif statuses and statuses <= {"cancelled", "rejected", "blocked"}:
            project.status = "cancelled"
        # else: still "pending" -- no change needed.
        self.session.flush()
        return project
