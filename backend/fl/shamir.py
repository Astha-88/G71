"""Shamir's Secret Sharing (t-of-n threshold), over a prime field.

Standard polynomial construction: to share a secret s among n parties with
threshold t, pick a random degree-(t-1) polynomial f(x) = s + a1*x + a2*x^2
+ ... + a_{t-1}*x^{t-1} over GF(p), and give party i the point (i, f(i)).
Any t of the n points can reconstruct f (and hence f(0) = s) via Lagrange
interpolation; fewer than t points reveal nothing about s (information-
theoretically, not just computationally -- this doesn't rely on any
cryptographic hardness assumption, unlike the ECDH used elsewhere in
fl/secure_agg.py).

Used here to share the 64-bit integer SEED that generates a pairwise
SecAgg mask (see fl/secure_agg.py's derive_mask) -- not the mask vector
itself, which isn't directly shareable as a single field element. Sharing
the seed is equivalent: anyone who reconstructs it can regenerate the
exact same mask via derive_mask.
"""
from __future__ import annotations

import secrets
from typing import List, Tuple

# A prime larger than any 64-bit seed we'll ever share (2^64 - 1 is the max
# seed value from secure_agg.py's `int.from_bytes(..., "big")` on 8 bytes).
# This is a standard, well-known safe prime used in several reference
# Shamir implementations, comfortably larger than 2^64.
_PRIME = 2**127 - 1  # a Mersenne prime, convenient and large enough


def split(secret: int, threshold: int, num_shares: int) -> List[Tuple[int, int]]:
    """Returns num_shares (x, y) points; any `threshold` of them reconstruct `secret`."""
    if not (0 <= secret < _PRIME):
        raise ValueError("secret out of range for the field")
    if threshold < 1 or threshold > num_shares:
        raise ValueError("threshold must be between 1 and num_shares")

    # Random coefficients for degree-(threshold-1) polynomial, constant term = secret
    coeffs = [secret] + [secrets.randbelow(_PRIME) for _ in range(threshold - 1)]

    def eval_poly(x: int) -> int:
        result = 0
        for coeff in reversed(coeffs):
            result = (result * x + coeff) % _PRIME
        return result

    return [(x, eval_poly(x)) for x in range(1, num_shares + 1)]


def reconstruct(shares: List[Tuple[int, int]]) -> int:
    """Lagrange interpolation at x=0 to recover the secret, given >= threshold shares."""
    secret = 0
    for i, (xi, yi) in enumerate(shares):
        numerator, denominator = 1, 1
        for j, (xj, _yj) in enumerate(shares):
            if i == j:
                continue
            numerator = (numerator * (-xj % _PRIME)) % _PRIME
            denominator = (denominator * (xi - xj)) % _PRIME
        lagrange_coeff = (numerator * pow(denominator, -1, _PRIME)) % _PRIME
        secret = (secret + yi * lagrange_coeff) % _PRIME
    return secret
