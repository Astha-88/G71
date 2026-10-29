"""Fast, dependency-light tests for Shamir sharing, Secure Aggregation masking,
the RDP privacy accountant, and the DP-SGD step. No database, no subprocesses.

Run:  python3 tests/test_crypto_and_privacy.py     (or: python3 -m pytest tests/)
"""
import math
import os
import random
import secrets
import sys

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

import fl.shamir as shamir  # noqa: E402
from fl.dp_accountant import Accountant, compute_epsilon  # noqa: E402
from fl.secure_agg import generate_keypair, pairwise_mask_contribution, serialize_public_key  # noqa: E402


def test_shamir_reconstructs_at_threshold_and_not_below():
    rng = random.Random(0)
    for _ in range(50):
        secret = secrets.randbelow(2 ** 64)
        t = rng.randint(2, 5)
        n = rng.randint(t, t + 3)
        shares = shamir.split(secret, t, n)
        assert shamir.reconstruct(rng.sample(shares, t)) == secret
        assert shamir.reconstruct(shares[: t - 1]) != secret


def _masked_updates(round_index, dim=7):
    hospitals = ["hospital1", "hospital2", "hospital3"]
    keys = {h: generate_keypair() for h in hospitals}
    pub = {h: serialize_public_key(k[1]) for h, k in keys.items()}
    true = {h: np.random.default_rng(i).normal(size=dim) for i, h in enumerate(hospitals)}
    masked = {
        h: true[h] + pairwise_mask_contribution(h, keys[h][0], pub, round_index, dim)
        for h in hospitals
    }
    return true, masked


def test_secagg_masks_cancel_in_the_sum_but_hide_each_individual_update():
    true, masked = _masked_updates(round_index=3)
    assert np.allclose(sum(masked.values()), sum(true.values()), atol=1e-9)
    for h in true:
        # Each individual masked value is dominated by the mask, not the signal.
        assert np.linalg.norm(masked[h] - true[h]) > 10


def test_secagg_masks_are_fresh_every_round():
    hospitals = ["hospital1", "hospital2", "hospital3"]
    keys = {h: generate_keypair() for h in hospitals}
    pub = {h: serialize_public_key(k[1]) for h, k in keys.items()}
    m1 = pairwise_mask_contribution("hospital1", keys["hospital1"][0], pub, 1, 7)
    m2 = pairwise_mask_contribution("hospital1", keys["hospital1"][0], pub, 2, 7)
    assert not np.allclose(m1, m2)


def test_epsilon_decreases_with_noise_and_increases_with_steps():
    assert compute_epsilon(50, 2.0, 1e-5) < compute_epsilon(50, 1.0, 1e-5)
    assert compute_epsilon(10, 1.0, 1e-5) < compute_epsilon(50, 1.0, 1e-5)


def test_rdp_composition_is_much_tighter_than_naive_composition():
    single_shot = math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
    naive_100_steps = 100 * single_shot
    assert compute_epsilon(100, 1.0, 1e-5) < 0.5 * naive_100_steps


def test_stateful_accountant_matches_the_closed_form():
    acc = Accountant()
    for _ in range(50):
        acc.step(1.0)
    assert abs(acc.get_epsilon(1e-5) - compute_epsilon(50, 1.0, 1e-5)) < 1e-6


def test_dp_step_equals_plain_step_when_noise_is_zero_and_nothing_is_clipped():
    os.environ.setdefault("HOSPITAL_NAME", "hospital1")
    import fl.collaborator_server as cs

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    y = (rng.random(50) > 0.5).astype(float)
    w0 = rng.normal(size=6) * 0.01

    class ZeroNoise:
        def normal(self, loc, scale, size):
            return np.zeros(size)

    plain_w, plain_b = cs._train_step_plain(w0.copy(), 0.0, w0.copy(), X, y, 0.0, 0.2)
    dp_w, dp_b = cs._train_step_dp(w0.copy(), 0.0, w0.copy(), X, y, 0.0, 0.2, 1000.0, 0.0, ZeroNoise())
    assert np.allclose(plain_w, dp_w, atol=1e-12)
    assert abs(plain_b - dp_b) < 1e-12

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

