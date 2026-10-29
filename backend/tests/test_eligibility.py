"""Tests the REAL ResearchProjectService.eligible_hospital_datasets against
stubbed dataset rows (3 hospitals x 3 diseases) -- no database needed, so these
run anywhere. Covers: "request from all eligible hospitals" resolves to every
hospital holding the disease; blocking is hospital-scoped and silent; vague or
ambiguous topics are refused instead of guessed.

NOT covered here (needs a live PostgreSQL): the partial unique indexes and
concurrent-request race behaviour.

Run:  python3 tests/test_eligibility.py     (or: python3 -m pytest tests/)
"""
import os
import sys
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

_STUBS = ["sqlalchemy", "sqlalchemy.exc", "sqlalchemy.orm", "db", "db.models", "db.database",
          "services.access_service", "services.audit_service", "blockchain", "blockchain.fabric_client"]
_saved = {m: sys.modules.get(m) for m in _STUBS}
for _m in _STUBS:
    sys.modules[_m] = MagicMock()
try:
    from services.research_project_service import ResearchProjectService
finally:  # don't leave MagicMocks behind for other test modules
    for _m, _orig in _saved.items():
        if _orig is None:
            sys.modules.pop(_m, None)
        else:
            sys.modules[_m] = _orig

DISEASES = ["COVID19", "LUNG_CANCER", "DIABETES"]
PRETTY = {"COVID19": "COVID-19", "LUNG_CANCER": "Lung Cancer", "DIABETES": "Diabetes"}
ROWS = [
    NS(id=i * 10 + j, institution_id=i, disease=NS(code=d), name=f"{PRETTY[d]} Patient Records - Hospital{i}")
    for i in (1, 2, 3) for j, d in enumerate(DISEASES)
]


class _Query:
    def join(self, *a, **k): return self
    def filter(self, *a, **k): return self
    def all(self): return ROWS


def _eligible(topic, blocked=()):
    svc = ResearchProjectService.__new__(ResearchProjectService)
    svc.session = MagicMock()
    svc.session.query.return_value = _Query()
    svc._blocked_institution_ids = lambda researcher_id: set(blocked)
    return svc.eligible_hospital_datasets(topic, 4)


def test_all_eligible_resolves_to_every_hospital_holding_that_one_disease():
    cases = {
        "Diabetes": "DIABETES", "DIABETES": "DIABETES", "diabetes": "DIABETES",
        "LUNG CANCER": "LUNG_CANCER", "Lung_Cancer": "LUNG_CANCER", "lung": "LUNG_CANCER", "cancer": "LUNG_CANCER",
        "COVID-19": "COVID19", "covid19": "COVID19", "covid": "COVID19",
    }
    for topic, code in cases.items():
        result = _eligible(topic)
        assert sorted(ds.institution_id for ds in result) == [1, 2, 3], topic
        assert {ds.disease.code for ds in result} == {code}, topic
        # exactly one dataset per hospital, so keying by institution_id can't collapse two diseases
        assert len({ds.institution_id for ds in result}) == len(result), topic


def test_a_blocking_hospital_is_silently_excluded_and_others_are_unaffected():
    assert sorted(ds.institution_id for ds in _eligible("Diabetes", {3})) == [1, 2]
    assert sorted(ds.institution_id for ds in _eligible("Diabetes", {1, 2})) == [3]
    assert _eligible("Diabetes", {1, 2, 3}) == []


def test_a_block_does_not_change_how_the_topic_is_interpreted():
    assert {ds.disease.code for ds in _eligible("Diabetes", {3})} == {"DIABETES"}


def test_vague_ambiguous_or_unknown_topics_are_refused_never_guessed():
    for topic in ["e", "a", "patient", "records", "hospital", "co", "", "   ", "Parkinsons", "xyz"]:
        assert _eligible(topic) == [], repr(topic)

if __name__ == "__main__":
    import traceback
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception:
                failed += 1
                print("FAIL", _name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)

