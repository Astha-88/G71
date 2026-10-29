from __future__ import annotations

from fastapi import HTTPException

# Every ValueError code raised by AccessService / ResearchProjectService /
# hospital-blocking logic lands here exactly once, shared across every
# router that calls into those services, so a new failure mode added to
# any of them can't accidentally leak as an unhandled 500 in one router
# while being handled correctly in another. Anything not in this map
# still falls through to a generic 400 rather than crashing, but should
# be added here with the right status code as soon as it's noticed.
ERROR_STATUS = {
    "dataset_not_found": (404, "Dataset not found"),
    "request_not_found": (404, "Request not found"),
    "project_not_found": (404, "Research project not found"),
    "not_authorized": (403, "Not authorized for this action"),
    "approver_not_authorized": (403, "Only a hospital_admin at the dataset's owning institution can approve this request"),
    "request_not_pending": (400, "Request is not pending"),
    "already_voted": (400, "Approver already voted"),
    "already_blocked": (409, "This researcher is already blocked by this institution"),
    "not_blocked": (400, "This researcher is not currently blocked by this institution"),
    "duplicate_pending_request": (409, "A pending request already exists for this dataset and researcher"),
    "active_project_exists": (409, "You already have an active research project. Complete or cancel it before starting another."),
    "project_not_active": (400, "This project is not in an active state"),
    "invalid_hospital_scope": (400, "hospital_scope must be one of: specific, multiple, all_eligible"),
    "no_hospitals_selected": (400, "No hospitals were selected"),
    "specific_scope_requires_one_hospital": (400, "hospital_scope='specific' requires exactly one hospital_id"),
    "hospital_not_eligible": (400, "That hospital is not available for this research topic."),
    "no_eligible_hospitals_for_topic": (404, "No hospital currently has a dataset matching this topic"),
    "no_eligible_hospitals_selected": (400, "None of the selected hospitals are available for this research topic."),
    "all_selected_hospitals_blocked": (400, "None of the selected hospitals are available for this research topic."),
    # Deliberately generic and identical in shape to "no hospital has this
    # dataset" -- a blocked researcher should not be able to distinguish
    # "blocked" from "no eligible hospital exists" from the response alone
    # (see db/models.py's ResearcherBlock docstring: reason is admin-only).
    "researcher_blocked": (400, "That hospital is not available for this research topic."),
    # Admin-facing ONLY (raised from approve(), which only a hospital_admin of the
    # blocking institution can reach) -- so, unlike researcher_blocked above, it
    # may say plainly that the researcher is blocked.
    "approver_researcher_blocked": (409, "This researcher is currently blocked by your institution. Unblock them before reviewing this request."),
    "secagg_coordinate_median_conflict": (
        400,
        "secagg_enabled and aggregation_strategy='coordinate_median' are mutually exclusive: "
        "SecAgg hides individual updates from the aggregator; coordinate-median aggregation "
        "requires seeing them.",
    ),
}


def raise_for(exc: ValueError) -> None:
    code = str(exc)
    status, detail = ERROR_STATUS.get(code, (400, code))
    raise HTTPException(status_code=status, detail=detail)
