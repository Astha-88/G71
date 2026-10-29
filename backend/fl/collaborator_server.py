"""Run as: python3 -m fl.collaborator_server
Env vars: HOSPITAL_NAME, PORT, DATA_DIR (defaults to ../data relative to this file)

This is intentionally a plain stdlib http.server, not FastAPI — a real
collaborator runs on a hospital's own infrastructure, which shouldn't need
to share the aggregator's dependency stack (SQLAlchemy, the ORM models,
etc). The only contract between aggregator and collaborator is this small
JSON-over-HTTP protocol:

  GET  /health              -> {"status": "ok", "hospital": ..., "n_samples": ...}
  POST /stats                -> {"sum": [...], "sumsq": [...], "count": N}
                                 (used once to compute a FEDERATED mean/std
                                 across all hospitals WITHOUT pooling raw
                                 data anywhere)
  POST /train  {weights, bias, mean, std, mu, epochs, lr,
                dp_enabled, clip_norm, noise_multiplier, delta}
                              -> {"weights": [...], "bias": ..., "n_samples": ...,
                                  "val_loss": ..., "val_acc": ..., "val_auc": ...,
                                  "dp": {"enabled": bool, "epsilon_spent": ...,
                                         "delta": ..., "total_steps": ...}}
                                 (local SGD, optionally with a FedProx
                                 proximal term toward the received global
                                 weights, and optionally with DP-SGD --
                                 per-sample gradient clipping + calibrated
                                 Gaussian noise, see _train_local_dp below;
                                 evaluation happens locally too — the
                                 aggregator only ever receives numbers,
                                 never a single patient record)
  GET  /privacy               -> {"total_steps": N, "epsilon": ..., "delta": ...}
                                 (cumulative privacy spent by THIS
                                 collaborator across every DP-SGD step it
                                 has ever run, independent of which
                                 experiment triggered it -- privacy budget
                                 belongs to the dataset, not to one run)

DP-SGD here means: for each local "epoch" (one full-batch gradient step,
matching how the non-DP path already works), compute the PER-SAMPLE
gradient for every training row, clip each one's L2 norm to `clip_norm`,
sum the clipped gradients, add Gaussian noise with stddev
`noise_multiplier * clip_norm` to that sum, then divide by the batch size
to get the (now private) mean gradient used for the update. This matches
the standard DP-SGD mechanism (Abadi et al., 2016) EXCEPT we don't do
Poisson-subsampled minibatches (see fl/dp_accountant.py's module docstring
for why, and what that costs us in bound tightness).
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fl.dp_accountant import Accountant  # noqa: E402
from fl.local_data import find_hospital_csv, list_available_diseases, load_csv  # noqa: E402
from fl.protocol import array_to_list, list_to_array  # noqa: E402

HOSPITAL_NAME = os.environ["HOSPITAL_NAME"]
PORT = int(os.environ.get("PORT", "9001"))
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)

# mTLS: if enabled, this server will refuse any connection that doesn't
# present a client certificate signed by the same CA (see fl/tls.py). This
# is what actually enforces "only the real aggregator can call /train" --
# without it, anything that can reach this port over the network could.
TLS_ENABLED = os.environ.get("FL_TLS_ENABLED", "0") == "1"
TLS_CERT = os.environ.get("FL_TLS_CERT")
TLS_KEY = os.environ.get("FL_TLS_KEY")
TLS_CA = os.environ.get("FL_TLS_CA")

# TEST-ONLY. See the comment at its use site in the /train handler.
BYZANTINE_MODE = os.environ.get("FL_BYZANTINE_MODE", "none")


def _stable_seed(name: str, salt: int = 0) -> int:
    """Python's built-in hash() randomizes string hashing PER PROCESS by
    default (PEP 456, a security feature) unless PYTHONHASHSEED is fixed --
    so `hash(HOSPITAL_NAME)` gives a DIFFERENT value every time a fresh
    collaborator subprocess/container starts, despite looking like a fixed
    per-hospital seed. This was a real, latent bug: it meant the train/val
    split silently differed across separate experiment runs (or container
    restarts), which surfaced as a ~0.03 unexplained discrepancy while
    cross-checking Secure Aggregation's output against plain FedAvg's --
    the two should be bit-for-bit comparable given the same global seed,
    and they weren't, because each run's collaborators had actually split
    their data differently underneath. hashlib is NOT subject to this
    randomization and gives the same digest for the same input in any
    process, which is what "seeded per-hospital" actually requires.
    """
    digest = hashlib.sha256(f"{name}:{salt}".encode()).hexdigest()
    return int(digest[:8], 16)


# Every hospital holds data for every disease, so there is no single dataset to
# load eagerly at startup. Each (hospital, disease) pair's data + train/val
# split + DP noise stream is loaded LAZILY on first use, keyed by disease_code.
# /stats and /train REQUIRE disease_code -- no default, so a caller can never
# accidentally get a different disease's data than the one it asked for.
_dataset_cache: dict = {}
_EXPERIMENT_SEED = int(os.environ.get("FL_EXPERIMENT_SEED", "0"))


def _get_dataset(disease_code: str) -> dict:
    if disease_code not in _dataset_cache:
        X_raw, y = load_csv(find_hospital_csv(DATA_DIR, HOSPITAL_NAME, disease_code))
        seed_key = f"{HOSPITAL_NAME}:{disease_code}"
        if _EXPERIMENT_SEED:
            # Opt-in (default 0 = unchanged behaviour): lets repeated experiment
            # trials differ in train/val partition AND DP-noise stream. Without
            # it every trial would reuse identical noise, understating variance.
            seed_key = f"{seed_key}:trial{_EXPERIMENT_SEED}"
        rng = np.random.default_rng(seed=_stable_seed(seed_key))
        n = X_raw.shape[0]
        idx = rng.permutation(n)
        cut = max(1, int(n * 0.8))
        # Separate DP rng so noise draws never perturb the reproducible split.
        _dataset_cache[disease_code] = {
            "X_raw": X_raw, "y": y, "train_idx": idx[:cut], "val_idx": idx[cut:], "n": n,
            "dp_rng": np.random.default_rng(seed=_stable_seed(seed_key, salt=1)),
        }
    return _dataset_cache[disease_code]

# One accountant per collaborator PROCESS, not per experiment -- privacy
# budget is a property of this hospital's dataset, and should accumulate
# across every DP-SGD step this process ever runs, however many
# experiments trigger them. Restarting the container resets it, which is
# an honest limitation to name: a real deployment would persist this to
# disk so a container restart can't be used to "reset" a spent budget.
_dp_accountant = Accountant()

# Secure Aggregation identity for this collaborator process -- one keypair
# for the process's lifetime (fine to reuse across rounds; masks are still
# fresh every round because round_index is folded into the derived mask,
# not the key itself). See fl/secure_agg.py for the full protocol writeup.
from fl.secure_agg import (  # noqa: E402
    decrypt_share_from_owner as _secagg_decrypt_share,
    deserialize_public_key as _secagg_deserialize_pubkey,
    generate_keypair as _secagg_generate_keypair,
    mask_from_seed as _secagg_mask_from_seed,
    pairwise_mask_contribution as _secagg_pairwise_mask,
    serialize_public_key as _secagg_serialize_pubkey,
    share_all_pairwise_seeds as _secagg_share_seeds,
)
_secagg_private_key, _secagg_public_key = _secagg_generate_keypair()

# Decrypted Shamir shares this collaborator holds ON BEHALF OF OTHERS,
# for dropout recovery -- {(owner_hospital, peer_hospital): (x, y)}.
# Populated by /store-shares, read by /reveal-share. In-memory only: a
# container restart loses these, same limitation as the DP accountant
# (see fl/dp_accountant.py's Accountant) -- a real deployment would
# persist this.
_held_shares: dict = {}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _bce(preds: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    preds = np.clip(preds, eps, 1 - eps)
    return float(-(y * np.log(preds) + (1 - y) * np.log(1 - preds)).mean())


def _acc(preds: np.ndarray, y: np.ndarray) -> float:
    return float(((preds >= 0.5).astype(np.float64) == y).mean())


def _auc(preds: np.ndarray, y: np.ndarray) -> float:
    pos, neg = preds[y == 1], preds[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(preds)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(preds) + 1)
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _train_step_plain(w: np.ndarray, b: float, global_w: np.ndarray, X: np.ndarray, y: np.ndarray, mu: float, lr: float) -> tuple:
    n = X.shape[0]
    preds = _sigmoid(X @ w + b)
    error = preds - y
    grad_w = X.T @ error / n + mu * (w - global_w)  # FedProx penalty
    grad_b = error.mean()
    return w - lr * grad_w, b - lr * grad_b


def _train_step_dp(
    w: np.ndarray, b: float, global_w: np.ndarray, X: np.ndarray, y: np.ndarray,
    mu: float, lr: float, clip_norm: float, noise_multiplier: float, dp_rng: np.random.Generator,
) -> tuple:
    """One DP-SGD step: per-sample gradient clipping + calibrated Gaussian
    noise on the SUM, then divide by n. For logistic regression the
    per-sample gradient w.r.t. [w, b] is exactly
    (pred_i - y_i) * [x_i, 1] -- no autodiff needed, it's closed-form.
    """
    n = X.shape[0]
    preds = _sigmoid(X @ w + b)
    error = preds - y  # shape (n,)

    # Per-sample gradients: grad_w_i = error_i * x_i, grad_b_i = error_i
    per_sample_grad_w = X * error[:, None]           # (n, n_features)
    per_sample_grad_b = error[:, None]                # (n, 1)
    per_sample_grad = np.concatenate([per_sample_grad_w, per_sample_grad_b], axis=1)  # (n, n_features+1)

    # Clip each sample's gradient to L2 norm <= clip_norm. This bounds how
    # much any single patient's record can influence the update -- the
    # sensitivity guarantee the noise below is calibrated against.
    norms = np.linalg.norm(per_sample_grad, axis=1, keepdims=True)  # (n, 1)
    clip_factor = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    clipped = per_sample_grad * clip_factor  # (n, n_features+1)

    clipped_sum = clipped.sum(axis=0)  # (n_features+1,) -- L2-sensitivity of this sum is exactly clip_norm
    noise = dp_rng.normal(loc=0.0, scale=noise_multiplier * clip_norm, size=clipped_sum.shape)
    noisy_sum = clipped_sum + noise

    grad_mean = noisy_sum / n
    grad_w = grad_mean[:-1] + mu * (w - global_w)  # FedProx penalty added post-noise: it's a
    grad_b = grad_mean[-1]                          # deterministic function of w/w_global, not of the data,
                                                      # so it doesn't need to be part of the private sum.
    _dp_accountant.step(noise_multiplier)
    return w - lr * grad_w, b - lr * grad_b


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet default stdout spam
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "hospital": HOSPITAL_NAME,
                "diseases_available": list_available_diseases(DATA_DIR, HOSPITAL_NAME),
            })
        elif self.path == "/keyshare":
            # Publishes this collaborator's PUBLIC key only -- the
            # aggregator relays this to every other collaborator so they
            # can each independently derive pairwise shared secrets via
            # ECDH. The aggregator never sees a private key and cannot
            # derive any shared secret itself.
            self._send_json(200, {
                "hospital": HOSPITAL_NAME,
                "public_key": _secagg_serialize_pubkey(_secagg_public_key),
            })
        elif self.path.startswith("/privacy"):
            # delta is a query param, e.g. /privacy?delta=1e-5 (defaults to 1e-5)
            delta = 1e-5
            if "?" in self.path:
                from urllib.parse import parse_qs, urlsplit
                qs = parse_qs(urlsplit(self.path).query)
                if "delta" in qs:
                    delta = float(qs["delta"][0])
            self._send_json(200, {
                "hospital": HOSPITAL_NAME,
                "total_dp_steps": _dp_accountant.total_steps,
                "epsilon": _dp_accountant.get_epsilon(delta) if _dp_accountant.total_steps > 0 else 0.0,
                "delta": delta,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        if self.path == "/stats":
            disease_code = payload.get("disease_code")
            if not disease_code:
                self._send_json(400, {"error": "disease_code is required"})
                return
            try:
                dataset = _get_dataset(disease_code)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json(404, {"error": str(exc)})
                return
            X_train = dataset["X_raw"][dataset["train_idx"]]
            self._send_json(200, {
                "sum": array_to_list(X_train.sum(axis=0)),
                "sumsq": array_to_list((X_train ** 2).sum(axis=0)),
                "count": int(X_train.shape[0]),
            })
            return

        if self.path == "/share-mask-seeds":
            # Step 1 of dropout-recovery setup for this round: split the
            # round seed for every pair this collaborator is responsible
            # for (see fl/secure_agg.py's share_all_pairwise_seeds -- only
            # the alphabetically-earlier hospital in each pair shares it)
            # and encrypt one share per holder. The aggregator relays the
            # result to /store-shares on each holder; it cannot read any
            # share itself, only pass along ciphertext.
            all_hospitals = payload["all_hospitals"]
            peer_public_keys = payload["peer_public_keys"]
            round_index = int(payload["round_index"])
            threshold = int(payload.get("threshold", 1))
            result = _secagg_share_seeds(
                HOSPITAL_NAME, _secagg_private_key, all_hospitals, peer_public_keys, round_index, threshold
            )
            self._send_json(200, {"shares_by_holder": result})
            return

        if self.path == "/store-shares":
            # Step 2: this collaborator is a HOLDER for one or more pairs
            # it isn't itself part of. Decrypt (it can, using its own
            # private key + the owner's public key) and hold onto each
            # share locally, in case it's later asked to reveal one for
            # dropout recovery. The aggregator relayed this but never saw
            # the plaintext share.
            owner_hospital = payload["owner_hospital"]
            owner_public_key = _secagg_deserialize_pubkey(payload["owner_public_key"])
            for peer_hospital, encrypted_blob in payload["shares"].items():
                share = _secagg_decrypt_share(_secagg_private_key, owner_public_key, encrypted_blob)
                _held_shares[(owner_hospital, peer_hospital)] = share
            self._send_json(200, {"stored": len(payload["shares"])})
            return

        if self.path == "/reveal-share":
            # Step 3, only invoked when the aggregator has ALREADY
            # determined (via /health or a failed /train call) that a
            # specific collaborator didn't respond this round, and needs
            # to reconstruct the seed for a pair involving it to cancel
            # that pair's leftover mask term. Reveals only ONE share of
            # ONE pairwise seed -- not this collaborator's own update, not
            # any other pair's share.
            owner_hospital = payload["owner_hospital"]
            peer_hospital = payload["peer_hospital"]
            key = (owner_hospital, peer_hospital)
            if key not in _held_shares:
                self._send_json(404, {"error": f"no share held for pair {key}"})
                return
            x, y = _held_shares[key]
            self._send_json(200, {"x": x, "y": y})
            return

        if self.path == "/train":
            disease_code = payload.get("disease_code")
            if not disease_code:
                self._send_json(400, {"error": "disease_code is required"})
                return
            try:
                dataset = _get_dataset(disease_code)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json(404, {"error": str(exc)})
                return

            mean = list_to_array(payload["mean"])
            std = list_to_array(payload["std"])
            w = list_to_array(payload["weights"]).copy()
            b = float(payload["bias"])
            mu = float(payload.get("mu", 0.0))  # FedProx coefficient; 0 = plain FedAvg
            epochs = int(payload.get("epochs", 5))
            lr = float(payload.get("lr", 0.2))
            global_w = w.copy()  # anchor for the FedProx proximal term

            dp_enabled = bool(payload.get("dp_enabled", False))
            clip_norm = float(payload.get("clip_norm", 1.0))
            noise_multiplier = float(payload.get("noise_multiplier", 1.0))
            delta = float(payload.get("delta", 1e-5))

            secagg_enabled = bool(payload.get("secagg_enabled", False))
            round_index = int(payload.get("round_index", 0))
            peer_public_keys = payload.get("peer_public_keys", {})
            total_n_all_clients = payload.get("total_n")

            X_raw, y = dataset["X_raw"], dataset["y"]
            train_idx, val_idx = dataset["train_idx"], dataset["val_idx"]
            X_train = (X_raw[train_idx] - mean) / std
            y_train = y[train_idx]
            X_val = (X_raw[val_idx] - mean) / std
            y_val = y[val_idx]

            n = X_train.shape[0]
            for _ in range(epochs):
                if dp_enabled:
                    w, b = _train_step_dp(
                        w, b, global_w, X_train, y_train, mu, lr, clip_norm, noise_multiplier, dataset["dp_rng"]
                    )
                else:
                    w, b = _train_step_plain(w, b, global_w, X_train, y_train, mu, lr)

            # TEST-ONLY byzantine-client simulation, gated behind an env
            # var never set by anything in normal operation (docker-compose,
            # scripts/, the API). Exists so robust-aggregation defenses
            # (fl/aggregator.py's "coordinate_median" strategy) can be
            # measured against an ACTUAL malicious update instead of just
            # asserted to work. A real attacker wouldn't announce itself
            # this way -- this simulates the OUTCOME (a collaborator
            # returning an adversarial update) to test the aggregator's
            # response to it, not a realistic attack delivery mechanism.
            if BYZANTINE_MODE == "scaled_negation":
                # Push hard in the opposite direction of what was actually
                # learned, at large magnitude -- a classic "poison the
                # average" update.
                w = -20.0 * w
                b = -20.0 * b
                # Also lies about its sample count, which is what actually
                # breaks plain sample-weighted FedAvg: a bigger claimed n
                # buys this update more influence in the weighted average.
                n = 5000

            val_preds = _sigmoid(X_val @ w + b)
            val_metrics = {
                "val_n": int(X_val.shape[0]),
                "val_loss": _bce(val_preds, y_val),
                "val_acc": _acc(val_preds, y_val),
                "val_auc": _auc(val_preds, y_val),
            }
            # IMPORTANT: evaluation above happens on the HONEST (post-DP,
            # post-byzantine-simulation) model -- masking below only
            # affects what goes ON THE WIRE for aggregation, never the
            # locally-computed performance numbers. Reporting a masked
            # AUC would be meaningless (it's not even the same model
            # anymore) -- the val metrics are a legitimate thing to
            # release in the clear; they don't reveal the weights.
            wire_w, wire_b = w, b
            secagg_info = {"enabled": False}

            if secagg_enabled:
                if not total_n_all_clients:
                    self._send_json(400, {"error": "secagg_enabled requires total_n"})
                    return
                if not peer_public_keys or HOSPITAL_NAME not in peer_public_keys:
                    self._send_json(400, {"error": "secagg_enabled requires this collaborator's own key in peer_public_keys"})
                    return

                # Pre-scale by this client's share of the total, BEFORE
                # masking -- this is what lets the aggregator recover the
                # correct WEIGHTED sum just by adding up masked values,
                # without ever separately weighting by (revealed)
                # per-client sample counts at aggregation time.
                frac = n / float(total_n_all_clients)
                scaled = np.concatenate([w * frac, [b * frac]])

                mask = _secagg_pairwise_mask(
                    HOSPITAL_NAME, _secagg_private_key, peer_public_keys, round_index, dim=scaled.shape[0]
                )
                masked = scaled + mask
                wire_w, wire_b = masked[:-1], masked[-1]
                secagg_info = {"enabled": True, "round_index": round_index, "weight_fraction": frac}

            response = {
                "weights": array_to_list(wire_w),
                "bias": float(wire_b),
                "n_samples": int(n),
                **val_metrics,
                "dp": {
                    "enabled": dp_enabled,
                    "epsilon_spent": _dp_accountant.get_epsilon(delta) if dp_enabled else None,
                    "delta": delta if dp_enabled else None,
                    "total_steps": _dp_accountant.total_steps,
                },
                "secagg": secagg_info,
            }
            self._send_json(200, response)
            return

        self._send_json(404, {"error": "not found"})


def main() -> None:
    # Bind to 0.0.0.0, not 127.0.0.1: inside a Docker container, a socket
    # bound to 127.0.0.1 only accepts connections that originate from
    # WITHIN that same container's network namespace. Other containers on
    # the same docker-compose network (e.g. `backend` calling
    # "fl-hospital1:9101") would get connection-refused, because their
    # traffic arrives on the container's real network interface, not its
    # loopback. 0.0.0.0 is a strict superset -- it still accepts local
    # loopback connections too, so subprocess-mode local dev (which talks
    # to 127.0.0.1) is unaffected by this fix.
    if TLS_ENABLED:
        if not (TLS_CERT and TLS_KEY and TLS_CA):
            raise RuntimeError("FL_TLS_ENABLED=1 requires FL_TLS_CERT, FL_TLS_KEY, and FL_TLS_CA to be set")
        from fl.tls import build_server_context
        ctx = build_server_context(TLS_CERT, TLS_KEY, TLS_CA)
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    else:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
        scheme = "http"

    diseases = list_available_diseases(DATA_DIR, HOSPITAL_NAME)
    print(f"[{HOSPITAL_NAME}] collaborator listening on {scheme}://0.0.0.0:{PORT} "
          f"(diseases available: {diseases}, loaded lazily, never sent raw)"
          f"{'  [mTLS required]' if TLS_ENABLED else '  [NO TLS -- plaintext]'}")
    server.serve_forever()


if __name__ == "__main__":
    main()
