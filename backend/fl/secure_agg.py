"""Secure Aggregation (SecAgg) -- pairwise additive masking.

The gap this closes, named explicitly in the security posture table: DP-SGD
(fl/dp_accountant.py, fl/collaborator_server.py) protects what's IN an
update -- a noised update can't be reverse-engineered back to individual
patient records with any confidence. It does NOT stop the aggregator from
seeing hospital1's update in isolation and correlating "this specific
(noisy) update" with "this specific hospital." SecAgg protects THAT: the
aggregator only ever recovers a SUM across all participating collaborators,
never any individual term of it.

The mechanism (Bonawitz et al., 2017, simplified -- no dropout-resilience
via Shamir secret sharing, see the caveat at the bottom):

1. Every collaborator generates an X25519 keypair once per experiment run.
2. The aggregator collects every collaborator's PUBLIC key (via /keyshare)
   and relays the full set to every collaborator -- the aggregator never
   sees anyone's PRIVATE key, so it cannot derive any pairwise secret
   itself; it's a pure relay for this step.
3. For every ordered pair of collaborators (i, j) with i's hospital name
   sorting before j's, both i and j independently derive the SAME shared
   secret via ECDH (each only needs its own private key + the other's
   public key -- this is the whole point of Diffie-Hellman). From that
   shared secret (mixed with the round number, so masks are fresh every
   round even though keys are reused), both derive the SAME pseudorandom
   mask vector.
4. i ADDS that mask to its update; j SUBTRACTS it. Every other party,
   including the aggregator, sees only i's or j's *masked* value -- which
   is statistically swamped by the mask and reveals nothing about the true
   update on its own.
5. When the aggregator sums every collaborator's masked update, each
   pairwise mask appears exactly once with a + and once with a -, so the
   sum cancels ALL masks exactly, leaving precisely the sum (or weighted
   sum, if each client pre-scaled locally -- see fl/aggregator.py) of the
   true updates. The aggregator computed the correct aggregate without
   ever holding a single individual update in the clear.

IMPORTANT CAVEATS, stated up front rather than discovered later:

- THREAT MODEL. Security holds against an HONEST-BUT-CURIOUS aggregator that
  only requests share reveals for collaborators that genuinely dropped. It does
  NOT hold against an ACTIVE aggregator (see the CAVEAT near holders_for_pair:
  no dropout verification, no double-masking). Separately, with n = 3 any two
  colluding hospitals can recover the third's update from the aggregate itself,
  so it is DP-SGD -- not secure aggregation -- that protects against that case.
- MASKING PRIMITIVES. Masks come from NumPy's default_rng (not a cryptographic
  PRG) seeded with only 64 bits, and are real-valued uniform masks over
  [-50, 50] rather than modular masks over a finite group, so they are not
  information-theoretically hiding and cancel only up to floating-point
  rounding (~1e-14 observed). This is a research prototype of the protocol's
  structure, not a production SecAgg implementation.

- This implementation does NOT handle a collaborator dropping out mid-round
  gracefully the way real SecAgg does. If collaborator C drops out after
  masks were agreed but before submitting its masked update, every OTHER
  collaborator's mask involving C never gets cancelled, and the sum is
  corrupted (not just missing C's contribution -- actively wrong, by the
  leftover mask terms). Real SecAgg solves this with Shamir secret sharing
  of each pairwise seed across all OTHER participants, so the aggregator
  can reconstruct (only) the masks belonging to a dropped client without
  ever learning that client's actual update. That's a substantial amount
  of additional protocol and is explicitly NOT implemented here -- if a
  collaborator you're using SecAgg with might drop out, this round's
  result is unusable and must be discarded, not trusted. See
  fl/aggregator.py's `run()` for where this is enforced (SecAgg rounds
  require ALL registered collaborators to respond -- unlike normal
  fault-tolerant rounds, there is no partial-quorum path here, because a
  partial SecAgg round produces a silently WRONG number, not a missing one).
- Mutually exclusive with coordinate-median robust aggregation, by
  construction: median needs to see individual values; SecAgg exists
  specifically so the aggregator cannot. See fl/aggregator.py's validation.
"""
from __future__ import annotations

import base64
import os
from typing import Dict, Tuple

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def generate_keypair() -> Tuple[x25519.X25519PrivateKey, x25519.X25519PublicKey]:
    private_key = x25519.X25519PrivateKey.generate()
    return private_key, private_key.public_key()


def serialize_public_key(pub: x25519.X25519PublicKey) -> str:
    from cryptography.hazmat.primitives import serialization
    raw = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def deserialize_public_key(b64: str) -> x25519.X25519PublicKey:
    raw = base64.b64decode(b64)
    return x25519.X25519PublicKey.from_public_bytes(raw)


def derive_shared_secret(private_key: x25519.X25519PrivateKey, peer_public: x25519.X25519PublicKey) -> bytes:
    """ECDH shared secret, run through HKDF to get well-distributed key
    material -- the raw ECDH output shouldn't be used directly as a PRG
    seed (it has structure; HKDF's whole job is stripping that out).
    """
    raw_shared = private_key.exchange(peer_public)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"vienko-secagg-pairwise-secret",
    ).derive(raw_shared)


def derive_round_seed(shared_secret: bytes, round_index: int) -> int:
    """Splits mask derivation into two steps so the round-specific SEED
    (not the raw ECDH shared_secret, and not the final float mask vector)
    can be Shamir-shared for dropout recovery -- see share_mask_seed /
    reconstruct_mask_seed below. Kept separate from mask_from_seed so both
    the normal masking path and the recovery path use the exact same
    derivation and can never silently diverge.
    """
    round_specific_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"vienko-secagg-round-{round_index}".encode(),
    ).derive(shared_secret)
    return int.from_bytes(round_specific_key[:8], "big")


def mask_from_seed(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=-50.0, high=50.0, size=dim)


def derive_mask(shared_secret: bytes, round_index: int, dim: int) -> np.ndarray:
    """Deterministic pseudorandom mask vector, fresh every round from the
    SAME underlying shared secret (round_index is mixed into the HKDF info,
    not the shared secret itself, so we don't need a fresh key exchange
    every round -- just a fresh derived mask). Both parties in a pair
    compute this independently and get the identical vector, without ever
    exchanging it.
    """
    seed = derive_round_seed(shared_secret, round_index)
    return mask_from_seed(seed, dim)


def pairwise_mask_contribution(
    my_hospital: str,
    my_private_key: x25519.X25519PrivateKey,
    peer_public_keys: dict,  # {hospital_name: base64 pubkey}, includes self (ignored)
    round_index: int,
    dim: int,
) -> np.ndarray:
    """The total mask ONE collaborator adds to its own update this round --
    the sum of its pairwise mask with every other participant, signed by a
    fixed, deterministic ordering (alphabetical hospital name) so both
    sides of every pair agree on who adds and who subtracts without any
    extra coordination.
    """
    total = np.zeros(dim)
    for peer_hospital, peer_pub_b64 in peer_public_keys.items():
        if peer_hospital == my_hospital:
            continue
        peer_pub = deserialize_public_key(peer_pub_b64)
        shared_secret = derive_shared_secret(my_private_key, peer_pub)
        mask = derive_mask(shared_secret, round_index, dim)
        sign = 1.0 if my_hospital < peer_hospital else -1.0
        total += sign * mask
    return total


# ---------------------------------------------------------------------------
# Dropout recovery via Shamir secret sharing.
#
# Problem this closes: if collaborator j drops out AFTER masks are agreed
# but BEFORE it submits its masked update, every other collaborator i that
# already computed a pairwise mask WITH j still has an uncancelled
# +-mask_ij term baked into its submission -- the sum is not just missing
# j's data, it's actively wrong by those leftover terms.
#
# Fix: for every pair (i, j), the ROUND SEED that generates mask_ij (see
# derive_round_seed above) is split via Shamir secret sharing (fl/shamir.py)
# among a set of HOLDER collaborators -- every OTHER participant besides i
# and j themselves. If j drops, the aggregator asks the holders to reveal
# their shares; once enough respond (>= threshold), the seed is
# reconstructed, mask_ij is regenerated via mask_from_seed, and subtracted
# from i's submission to cancel the leftover term exactly.
#
# Why holders, not just asking i directly: i already knows mask_ij outright
# (it computed it via ECDH) and COULD simply be asked to reveal it. But a
# malicious aggregator could then falsely claim "j dropped" on a round
# where j didn't, purely to trick i into revealing mask_ij, then use that
# to partially unmask a future submission of i's that also involves j. A
# threshold split among several holders means the aggregator can't get a
# mask value revealed by asking a single party a single lie -- it needs
# cooperation from `threshold` distinct collaborators, each of whom is
# independently trusting that the aggregator's dropout claim is real.
#
# CAVEAT -- this is a speed bump, not a defence against an ACTIVE aggregator:
# holders have no way to VERIFY that a collaborator really dropped, and there
# is no double-masking (the per-client self-mask of Bonawitz et al., 2017). An
# aggregator that falsely declares collaborator j dropped can therefore collect
# the shares of j's pairwise seeds from the holders, regenerate j's masks, and
# unmask the update j actually submitted. With the 3-hospital deployment every
# pair has exactly one holder and the threshold is 1, so there is no protection
# even against a single lie. See the THREAT MODEL note in the module docstring.
# ---------------------------------------------------------------------------


def holders_for_pair(all_hospitals: list, hospital_i: str, hospital_j: str) -> list:
    return sorted(h for h in all_hospitals if h not in (hospital_i, hospital_j))


def encrypt_share_for_holder(
    my_private_key: x25519.X25519PrivateKey,
    holder_public_key: x25519.X25519PublicKey,
    share: Tuple[int, int],
) -> str:
    """Encrypts one Shamir share (x, y) so only the intended holder can
    read it -- the aggregator relays this blob but, lacking the holder's
    private key, cannot decrypt it. Uses a symmetric key derived from an
    ECDH exchange between the share's OWNER and this specific HOLDER (a
    different pairing than the (i, j) being masked -- this key exists
    purely to protect shares in transit).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key_material = my_private_key.exchange(holder_public_key)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
               info=b"vienko-secagg-share-transport").derive(key_material)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    plaintext = f"{share[0]}:{share[1]}".encode()
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_share_from_owner(
    my_private_key: x25519.X25519PrivateKey,
    owner_public_key: x25519.X25519PublicKey,
    blob_b64: str,
) -> Tuple[int, int]:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key_material = my_private_key.exchange(owner_public_key)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
               info=b"vienko-secagg-share-transport").derive(key_material)
    aesgcm = AESGCM(key)
    raw = base64.b64decode(blob_b64)
    nonce, ciphertext = raw[:12], raw[12:]
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    x_str, y_str = plaintext.decode().split(":")
    return int(x_str), int(y_str)


def share_all_pairwise_seeds(
    my_hospital: str,
    my_private_key: x25519.X25519PrivateKey,
    all_hospitals: list,
    peer_public_keys: dict,
    round_index: int,
    threshold: int,
) -> dict:
    """For every peer j this collaborator masks against, splits the round
    seed for pair (my_hospital, j) and encrypts one share per holder.
    Returns {holder_hospital: {peer_hospital: encrypted_share_b64}} -- what
    the aggregator needs to relay to each holder.

    Only shares for peers that sort AFTER my_hospital alphabetically: since
    the ECDH shared secret (and therefore the derived seed) is identical
    from either side of a pair, having BOTH parties independently share the
    same seed would just double the protocol's message count for no
    benefit. By convention, the alphabetically-earlier hospital in each
    pair is the one responsible for sharing that pair's seed.
    """
    import fl.shamir as shamir

    out: Dict[str, Dict[str, str]] = {h: {} for h in all_hospitals if h != my_hospital}
    for peer_hospital, peer_pub_b64 in peer_public_keys.items():
        if peer_hospital == my_hospital or peer_hospital < my_hospital:
            continue
        peer_pub = deserialize_public_key(peer_pub_b64)
        shared_secret = derive_shared_secret(my_private_key, peer_pub)
        seed = derive_round_seed(shared_secret, round_index)

        holders = holders_for_pair(all_hospitals, my_hospital, peer_hospital)
        if not holders:
            continue  # only 2 total participants -- nobody else to hold a share
        n = len(holders)
        t = min(threshold, n)
        shares = shamir.split(seed, threshold=t, num_shares=n)
        for holder_hospital, share in zip(holders, shares):
            holder_pub = deserialize_public_key(peer_public_keys[holder_hospital])
            encrypted = encrypt_share_for_holder(my_private_key, holder_pub, share)
            out[holder_hospital][peer_hospital] = encrypted
    return out
