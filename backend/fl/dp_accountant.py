"""Privacy accounting for DP-SGD via Renyi Differential Privacy (RDP).

We deliberately do NOT implement Poisson-subsampled minibatch amplification
(what Opacus/TF Privacy do) -- each local DP-SGD step here uses the client's
FULL local training set as one "batch" (matching how the rest of this
codebase already does full-batch local training, see
fl/collaborator_server.py's _train_local). That keeps the accounting exact
and simple: composing T applications of the plain (non-subsampled) Gaussian
mechanism via RDP is a closed-form, well-understood bound. It also means
this bound is CONSERVATIVE (looser / worse) than what a subsampled analysis
would give for the same noise level -- subsampling amplification is a real
technique that would let you claim a smaller epsilon for the same noise,
and is the natural next step if this matters more than architectural
simplicity. We're not claiming the tighter bound we didn't implement.

Math, for the record:
- Gaussian mechanism with noise multiplier sigma (noise stddev = sigma *
  L2-sensitivity) has RDP epsilon at order alpha > 1 of:
      eps_RDP(alpha) = alpha / (2 * sigma^2)
  (Mironov, 2017 -- "Renyi Differential Privacy")
- RDP composes ADDITIVELY across T independent applications at the same
  order alpha:
      eps_RDP_total(alpha) = T * alpha / (2 * sigma^2)
- Converting an RDP guarantee at order alpha to (epsilon, delta)-DP:
      eps = eps_RDP(alpha) + ln(1/delta) / (alpha - 1)
  We minimize this over a grid of alpha to get the tightest bound our
  accountant can find at that grid resolution.
"""
from __future__ import annotations

import math
from typing import Iterable

# Alpha search grid. Real accountants (Opacus, TF Privacy) search a denser,
# adaptively-refined grid; this fixed grid is coarser but sufficient to
# demonstrate the mechanism correctly and gets within a small fraction of
# the true minimum for the sigma ranges we use (sigma in [0.3, 5]).
DEFAULT_ALPHA_GRID = [1 + x / 10 for x in range(1, 100)] + [12, 14, 16, 20, 24, 32, 48, 64, 128, 256]


def rdp_gaussian(alpha: float, sigma: float) -> float:
    """RDP epsilon of the Gaussian mechanism at order alpha, noise multiplier sigma."""
    if sigma <= 0:
        return float("inf")
    return alpha / (2.0 * sigma * sigma)


def rdp_to_eps(total_rdp_at_alpha, alpha: float, delta: float) -> float:
    if alpha <= 1:
        return float("inf")
    return total_rdp_at_alpha + math.log(1.0 / delta) / (alpha - 1.0)


def compute_epsilon(
    steps: int,
    noise_multiplier: float,
    delta: float,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
) -> float:
    """Total (epsilon, delta)-DP guarantee for `steps` compositions of the
    Gaussian mechanism with the given noise multiplier, found by minimizing
    the RDP-to-DP conversion over the alpha search grid.
    """
    if steps <= 0:
        return 0.0
    if noise_multiplier <= 0:
        return float("inf")  # no noise = no privacy guarantee

    best_eps = float("inf")
    for alpha in alpha_grid:
        total_rdp = steps * rdp_gaussian(alpha, noise_multiplier)
        eps = rdp_to_eps(total_rdp, alpha, delta)
        if eps < best_eps:
            best_eps = eps
    return best_eps


def required_noise_multiplier_for_epsilon(
    steps: int,
    target_epsilon: float,
    delta: float,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    lo: float = 0.01,
    hi: float = 50.0,
    tol: float = 1e-3,
) -> float:
    """Binary search for the smallest noise_multiplier such that
    compute_epsilon(steps, noise_multiplier, delta) <= target_epsilon.
    Larger noise_multiplier -> smaller epsilon (monotonic), so binary
    search is valid.
    """
    if compute_epsilon(steps, hi, delta, alpha_grid) > target_epsilon:
        raise ValueError(f"Even noise_multiplier={hi} can't reach epsilon={target_epsilon} for {steps} steps")

    while hi - lo > tol:
        mid = (lo + hi) / 2
        eps = compute_epsilon(steps, mid, delta, alpha_grid)
        if eps <= target_epsilon:
            hi = mid
        else:
            lo = mid
    return hi


class Accountant:
    """Stateful RDP accountant -- tracks privacy spent as a running total
    per alpha, so it composes correctly even across calls with DIFFERENT
    noise multipliers (e.g. if a collaborator trains under different DP
    settings across separate experiments). This is more general than
    compute_epsilon() above, which assumes one fixed sigma for all steps.

    Privacy budget belongs to the DATASET, not to any one experiment -- so
    a collaborator process should keep exactly one Accountant for its
    entire lifetime and call .step(sigma) once per DP-SGD gradient step it
    ever performs, across however many experiments it participates in.
    """

    def __init__(self, alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID) -> None:
        self.alpha_grid = list(alpha_grid)
        self._cumulative_rdp = {alpha: 0.0 for alpha in self.alpha_grid}
        self.total_steps = 0

    def step(self, noise_multiplier: float) -> None:
        for alpha in self.alpha_grid:
            self._cumulative_rdp[alpha] += rdp_gaussian(alpha, noise_multiplier)
        self.total_steps += 1

    def get_epsilon(self, delta: float) -> float:
        if self.total_steps == 0:
            return 0.0
        best = float("inf")
        for alpha, rdp in self._cumulative_rdp.items():
            eps = rdp_to_eps(rdp, alpha, delta)
            if eps < best:
                best = eps
        return best
