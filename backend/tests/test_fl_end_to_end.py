"""End-to-end federated training against REAL collaborator subprocesses over
real HTTP (loopback). ~30-60s total. No database needed.

Run:  python3 tests/test_fl_end_to_end.py     (or: python3 -m pytest tests/)
"""
import os
import sys
import tempfile
import time

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from fl.aggregator import FederatedTrainer  # noqa: E402

HOSPITALS = ["hospital1", "hospital2", "hospital3"]
_DATA_DIR = None


def _data_dir():
    global _DATA_DIR
    if _DATA_DIR is None:
        import scripts.generate_synthetic_data as gen
        _DATA_DIR = tempfile.mkdtemp(prefix="vienko-data-")
        gen.OUT_DIR = _DATA_DIR
        gen.main()
    return _DATA_DIR


def _trainer(disease="DIABETES", **kw):
    return FederatedTrainer(hospitals=list(HOSPITALS), disease_code=disease, data_dir=_data_dir(),
                            local_epochs=5, lr=0.2, mu=0.05, **kw)


def _run(trainer, rounds=4, **run_kw):
    trainer.spawn()
    try:
        return trainer.run(rounds=rounds, client_timeout=15.0, seed=42, **run_kw)
    finally:
        trainer.shutdown()


def test_all_three_hospitals_train_one_shared_model_for_the_same_disease():
    last = _run(_trainer())["history"][-1]
    assert last["status"] == "completed"
    assert sorted(last["samples_per_client"]) == HOSPITALS
    assert last["auc"] > 0.6


def test_secure_aggregation_computes_exactly_the_same_model_as_plain_fedavg():
    plain = _run(_trainer(secagg_enabled=False), rounds=6)["global_model"]
    secure = _run(_trainer(secagg_enabled=True), rounds=6)["global_model"]
    assert np.allclose(plain["weights"], secure["weights"], atol=1e-8)
    assert abs(plain["bias"] - secure["bias"]) < 1e-8


def test_a_hospital_that_is_down_is_excluded_and_training_continues_with_the_rest():
    t = _trainer()
    t.spawn()
    try:
        t.collaborators["hospital2"].process.kill()
        t.collaborators["hospital2"].process.wait()
        time.sleep(0.3)
        history = t.run(rounds=3, client_timeout=3.0, seed=42)["history"]
    finally:
        t.shutdown()
    for round_ in history:
        assert round_["status"] == "completed"
        assert sorted(m["hospital"] for m in round_["per_client"]) == ["hospital1", "hospital3"]


def test_an_unknown_disease_fails_cleanly_instead_of_silently_training_on_something_else():
    t = _trainer(disease="NOT_A_DISEASE")
    t.spawn()
    try:
        try:
            t.run(rounds=1, client_timeout=5.0)
        except RuntimeError:
            return
        raise AssertionError("expected a RuntimeError for an unknown disease_code")
    finally:
        t.shutdown()


def test_secagg_and_coordinate_median_are_rejected_as_mutually_exclusive():
    try:
        _trainer(secagg_enabled=True, aggregation_strategy="coordinate_median")
    except ValueError:
        return
    raise AssertionError("expected ValueError")

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

