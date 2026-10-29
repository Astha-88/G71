from __future__ import annotations

import json
import math
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from fl.protocol import array_to_list, list_to_array
from fl.secure_agg import holders_for_pair

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BASE_PORT = int(os.environ.get("FL_COLLABORATOR_BASE_PORT", "9101"))
N_FEATURES = 6


def _post(url: str, payload: dict, timeout: float = 30.0, ssl_context: Optional[ssl.SSLContext] = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as resp:
        return json.loads(resp.read())


def _get(url: str, timeout: float = 5.0, ssl_context: Optional[ssl.SSLContext] = None) -> dict:
    with urllib.request.urlopen(url, timeout=timeout, context=ssl_context) as resp:
        return json.loads(resp.read())


@dataclass
class Collaborator:
    hospital: str
    port: Optional[int]
    process: Optional[subprocess.Popen]
    base_url_override: Optional[str] = None
    scheme: str = "http"

    @property
    def base_url(self) -> str:
        if self.base_url_override:
            return self.base_url_override
        return f"{self.scheme}://127.0.0.1:{self.port}"


class FederatedTrainer:
    """Orchestrates a real multi-process FedAvg/FedProx run.

    Two deployment modes, same wire protocol either way:

    - "subprocess" (default): spawns each collaborator as a local OS
      process on 127.0.0.1. Used by scripts/run_demo_fl.py and local dev
      without docker-compose -- convenient, but all collaborators still
      share one machine's filesystem and network namespace.

    - "network": collaborators are already-running containers (see
      docker-compose.yml's fl-hospital1/2/3 services, each built from a
      separate image containing only that hospital's data). The trainer
      just health-checks and talks to them by hostname. This is the mode
      that makes the "separate hospital" story actually true: each
      collaborator is a separate container, on a separate Docker network
      namespace, built from an image that never had the other hospitals'
      CSVs copied into it in the first place.

    Either way, only aggregate statistics (for standardization) and model
    weight deltas cross the wire -- never a raw feature/label row.
    """

    def __init__(
        self,
        hospitals: List[str],
        disease_code: str,
        local_epochs: int = 5,
        lr: float = 0.2,
        mu: float = 0.05,
        data_dir: str = DATA_DIR,
        mode: str = "subprocess",
        collaborator_urls: Optional[Dict[str, str]] = None,
        dp_enabled: bool = False,
        dp_clip_norm: float = 1.0,
        dp_noise_multiplier: float = 1.0,
        dp_delta: float = 1e-5,
        tls_enabled: bool = False,
        tls_cert: Optional[str] = None,
        tls_key: Optional[str] = None,
        tls_ca: Optional[str] = None,
        collaborator_certs_dir: Optional[str] = None,
        aggregation_strategy: str = "fedavg",
        secagg_enabled: bool = False,
    ) -> None:
        self.hospitals = hospitals
        # Which disease's data each collaborator trains on -- REQUIRED, forwarded
        # on every /stats and /train call.
        self.disease_code = disease_code
        self.local_epochs = local_epochs
        self.lr = lr
        self.mu = mu  # FedProx proximal coefficient; 0.0 reduces to plain FedAvg
        self.data_dir = data_dir
        self.mode = mode
        # DP-SGD params, forwarded verbatim to each collaborator's /train
        # call -- the actual clipping/noising happens INSIDE the
        # collaborator (fl/collaborator_server.py's _train_step_dp), never
        # here, so the aggregator never even momentarily holds an
        # un-noised gradient.
        self.dp_enabled = dp_enabled
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_delta = dp_delta

        # "fedavg": sample-count-weighted average (the default so far in
        # this codebase). A malicious/compromised collaborator can corrupt
        # this arbitrarily -- it can both send an adversarial weight
        # vector AND lie about its sample count to buy itself more
        # influence in the average, since nothing here verifies either.
        # "coordinate_median": elementwise median across collaborators'
        # returned weight vectors, ignoring claimed sample counts entirely.
        # A median is robust to a MINORITY of arbitrarily bad values (up to
        # just under half the participants can be malicious/faulty and the
        # median still reflects the honest majority) -- see
        # fl/robust_aggregation_demo.py for a measured comparison against
        # an actual simulated malicious collaborator, not just an assertion
        # that this helps.
        if aggregation_strategy not in ("fedavg", "coordinate_median"):
            raise ValueError(f"Unknown aggregation_strategy: {aggregation_strategy}")
        self.aggregation_strategy = aggregation_strategy

        # Secure Aggregation: mutually exclusive with coordinate_median BY
        # CONSTRUCTION, not just policy -- median needs to see individual
        # values to compare them; SecAgg's entire point is that the
        # aggregator never can. See fl/secure_agg.py for the full protocol
        # and its caveats (in particular: SecAgg rounds require ALL
        # registered collaborators to respond, not just quorum -- a
        # partial SecAgg round produces a WRONG sum, not a missing one,
        # because uncancelled pairwise masks corrupt it).
        if secagg_enabled and aggregation_strategy == "coordinate_median":
            raise ValueError(
                "secagg_enabled and aggregation_strategy='coordinate_median' are mutually "
                "exclusive: SecAgg hides individual updates from the aggregator; "
                "coordinate-median aggregation requires seeing them."
            )
        self.secagg_enabled = secagg_enabled
        self._secagg_public_keys: Dict[str, str] = {}
        self._total_n: Optional[int] = None

        self.tls_enabled = tls_enabled
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._tls_ca = tls_ca
        # In subprocess mode (this process spawns all 3 collaborators
        # itself), it also needs each collaborator's OWN server cert to
        # hand it as that subprocess's env vars -- the aggregator's own
        # client cert (tls_cert/tls_key) is a different, separate identity
        # and must not be reused as a collaborator's server cert. Defaults
        # to the convention scripts/generate_certs.py writes:
        # backend/certs/{hospital}.crt / .key
        self._collaborator_certs_dir = collaborator_certs_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "certs"
        )
        if tls_enabled:
            if not (tls_cert and tls_key and tls_ca):
                raise ValueError("tls_enabled=True requires tls_cert, tls_key, and tls_ca")
            from fl.tls import build_client_context
            self._ssl_context = build_client_context(tls_cert, tls_key, tls_ca)

        scheme = "https" if tls_enabled else "http"
        # In network mode, URLs default to the docker-compose service
        # names ("fl-hospital1" etc.) on the shared fl-net network.
        self.collaborator_urls = collaborator_urls or {
            h: f"{scheme}://fl-{h}:9101" for h in hospitals
        }
        self.collaborators: Dict[str, Collaborator] = {}

    def spawn(self, timeout: float = 30.0) -> None:
        if self.mode == "network":
            self._connect_network(timeout)
        else:
            self._spawn_subprocesses(timeout)

    def _connect_network(self, timeout: float) -> None:
        deadline = time.time() + timeout
        for hospital in self.hospitals:
            url = self.collaborator_urls[hospital]
            self.collaborators[hospital] = Collaborator(hospital, None, None, base_url_override=url)
            while time.time() < deadline:
                try:
                    health = _get(f"{url}/health", timeout=1.0, ssl_context=self._ssl_context)
                    if health.get("status") == "ok":
                        break
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    time.sleep(0.3)
            else:
                raise RuntimeError(
                    f"Collaborator container for {hospital} ({url}) did not become healthy in "
                    f"time. Check `docker compose ps` / `docker compose logs fl-{hospital}`."
                )

    def _spawn_subprocesses(self, timeout: float) -> None:
        scheme = "https" if self.tls_enabled else "http"
        for i, hospital in enumerate(self.hospitals):
            port = BASE_PORT + i
            env = dict(os.environ)
            env["HOSPITAL_NAME"] = hospital
            env["PORT"] = str(port)
            env["DATA_DIR"] = self.data_dir
            if self.tls_enabled:
                env["FL_TLS_ENABLED"] = "1"
                env["FL_TLS_CERT"] = os.path.join(self._collaborator_certs_dir, f"{hospital}.crt")
                env["FL_TLS_KEY"] = os.path.join(self._collaborator_certs_dir, f"{hospital}.key")
                env["FL_TLS_CA"] = self._tls_ca
            proc = subprocess.Popen(
                [sys.executable, "-m", "fl.collaborator_server"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.collaborators[hospital] = Collaborator(hospital, port, proc, scheme=scheme)

        deadline = time.time() + timeout
        for hospital, c in self.collaborators.items():
            while time.time() < deadline:
                try:
                    health = _get(f"{c.base_url}/health", timeout=1.0, ssl_context=self._ssl_context)
                    if health.get("status") == "ok":
                        break
                except (urllib.error.URLError, ConnectionError, TimeoutError, ssl.SSLError):
                    time.sleep(0.15)
            else:
                self.shutdown()
                raise RuntimeError(f"Collaborator for {hospital} did not come up in time")

    def shutdown(self) -> None:
        if self.mode == "network":
            # Containers are managed by docker-compose's own lifecycle, not
            # this process -- nothing to tear down here.
            return
        for c in self.collaborators.values():
            if c.process and c.process.poll() is None:
                c.process.terminate()
                try:
                    c.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    c.process.kill()

    def _federated_mean_std(self, client_timeout: float = 30.0, min_quorum_fraction: float = 0.5) -> tuple:
        """Computes global feature mean/std from per-client (sum, sumsq,
        count) triples only -- no client ever sends a raw row to compute
        this, so standardization doesn't require pooling data centrally
        (unlike the earlier single-process version of this service).

        Also fault-tolerant now: this used to be an unguarded sequential
        loop that raised immediately (crashing the whole run before a
        single training round even started) if ANY collaborator was
        unreachable. A collaborator missing at this stage is excluded from
        `self.hospitals` for the rest of THIS run -- there's no sensible
        "carry over the previous value" fallback here the way there is for
        a mid-run round failure (there's no previous standardization to
        fall back to), so unlike a mid-run dropout, this permanently drops
        that hospital from the current experiment rather than retrying it
        later. If fewer than `min_quorum_fraction` of registered hospitals
        respond, this raises -- there's no reasonable partial result to
        return when we can't even establish a shared feature scale.
        """
        min_responders = max(1, math.ceil(len(self.hospitals) * min_quorum_fraction))
        stats_by_hospital: Dict[str, dict] = {}
        errors: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max(1, len(self.collaborators))) as pool:
            future_to_hospital = {
                pool.submit(_post, f"{c.base_url}/stats", {"disease_code": self.disease_code}, client_timeout, self._ssl_context): hospital
                for hospital, c in self.collaborators.items()
            }
            for future in as_completed(future_to_hospital, timeout=client_timeout + 5.0):
                hospital = future_to_hospital[future]
                try:
                    stats_by_hospital[hospital] = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors[hospital] = f"{type(exc).__name__}: {exc}"

        if len(stats_by_hospital) < min_responders:
            raise RuntimeError(
                f"Only {len(stats_by_hospital)}/{len(self.hospitals)} collaborators reachable for "
                f"federated standardization (need >= {min_responders}); cannot start training. "
                f"Errors: {errors}"
            )

        if errors:
            # Permanently narrow this run's participant set to whoever
            # actually responded -- the round loop in run() uses
            # self.collaborators, so drop the unreachable ones from it too.
            for hospital in errors:
                self.collaborators.pop(hospital, None)
            self.hospitals = list(stats_by_hospital.keys())

        total_sum = np.zeros(N_FEATURES)
        total_sumsq = np.zeros(N_FEATURES)
        total_n = 0
        for stats in stats_by_hospital.values():
            total_sum += list_to_array(stats["sum"])
            total_sumsq += list_to_array(stats["sumsq"])
            total_n += stats["count"]

        self._total_n = total_n  # needed by SecAgg's client-side weight-fraction pre-scaling

        mean = total_sum / total_n
        var = total_sumsq / total_n - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-8))

        if self.secagg_enabled:
            self._collect_secagg_public_keys(client_timeout)

        return mean, std

    def _collect_secagg_public_keys(self, client_timeout: float) -> None:
        """Fetches every (already-narrowed-to-responding) collaborator's
        PUBLIC key via GET /keyshare. Pure relay on the aggregator's part
        -- it stores and forwards these, but never sees a private key and
        cannot derive any pairwise shared secret itself. If a collaborator
        can't be reached even for this (very lightweight) call, SecAgg
        cannot proceed for this run: unlike a mid-round dropout, there's no
        partial-participant SecAgg here (see fl/secure_agg.py's caveats).
        """
        keys: Dict[str, str] = {}
        errors: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.collaborators))) as pool:
            future_to_hospital = {
                pool.submit(_get, f"{c.base_url}/keyshare", client_timeout, self._ssl_context): hospital
                for hospital, c in self.collaborators.items()
            }
            for future in as_completed(future_to_hospital, timeout=client_timeout + 5.0):
                hospital = future_to_hospital[future]
                try:
                    result = future.result()
                    keys[result["hospital"]] = result["public_key"]
                except Exception as exc:  # noqa: BLE001
                    errors[hospital] = f"{type(exc).__name__}: {exc}"

        if errors:
            raise RuntimeError(
                f"secagg_enabled requires ALL participating collaborators to be reachable for "
                f"key exchange; failed: {errors}"
            )
        self._secagg_public_keys = keys

    def _share_secagg_seeds(self, round_index: int, client_timeout: float, threshold: int = 1) -> None:
        """Runs the two-hop share distribution for one round, BEFORE
        training starts: (1) ask every collaborator to split and encrypt
        its pairwise round-seeds for the holders that need them, (2) relay
        each encrypted share to its intended holder via /store-shares. The
        aggregator only ever handles ciphertext in step 2 -- it collects
        {holder: {peer: encrypted_share}} from each owner and forwards the
        blob verbatim, never decrypting it.

        `threshold`: with this system's current 3 hospitals, every pair
        has exactly ONE possible holder (the third hospital), so threshold
        is clamped to 1 regardless of what's passed -- the genuine
        threshold protection this mechanism provides only shows up with 4+
        total participants (verified separately in a standalone 4-party
        test; see fl/secure_agg.py's module docstring for why holders,
        not direct reveal, still matters even at threshold=1: it at least
        keeps the REVEALING party different from the OWNER, so the
        aggregator can't just ask the owner "what mask did you use" and
        get a direct answer from the party with the most to lose from
        answering truthfully only when it's actually asked in good faith).
        """
        shares_by_owner: Dict[str, dict] = {}
        errors: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.collaborators))) as pool:
            future_to_hospital = {
                pool.submit(_post, f"{c.base_url}/share-mask-seeds", {
                    "all_hospitals": self.hospitals,
                    "peer_public_keys": self._secagg_public_keys,
                    "round_index": round_index,
                    "threshold": threshold,
                }, client_timeout, self._ssl_context): hospital
                for hospital, c in self.collaborators.items()
            }
            for future in as_completed(future_to_hospital, timeout=client_timeout + 5.0):
                hospital = future_to_hospital[future]
                try:
                    shares_by_owner[hospital] = future.result()["shares_by_holder"]
                except Exception as exc:  # noqa: BLE001
                    errors[hospital] = f"{type(exc).__name__}: {exc}"

        if errors:
            raise RuntimeError(f"Failed to collect shares for round {round_index}: {errors}")

        # Relay: for each owner's {holder: {peer: encrypted_share}}, deliver
        # the (owner, {peer: encrypted_share}) portion to that holder.
        relay_errors: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.collaborators) ** 2)) as pool:
            futures = []
            for owner_hospital, shares_by_holder in shares_by_owner.items():
                owner_pubkey = self._secagg_public_keys[owner_hospital]
                for holder_hospital, shares in shares_by_holder.items():
                    if not shares or holder_hospital not in self.collaborators:
                        continue
                    holder_url = self.collaborators[holder_hospital].base_url
                    futures.append((
                        holder_hospital,
                        pool.submit(_post, f"{holder_url}/store-shares", {
                            "owner_hospital": owner_hospital,
                            "owner_public_key": owner_pubkey,
                            "shares": shares,
                        }, client_timeout, self._ssl_context),
                    ))
            for holder_hospital, future in futures:
                try:
                    future.result(timeout=client_timeout)
                except Exception as exc:  # noqa: BLE001
                    relay_errors[holder_hospital] = f"{type(exc).__name__}: {exc}"

        if relay_errors:
            raise RuntimeError(f"Failed to relay shares for round {round_index}: {relay_errors}")

    def _attempt_secagg_recovery(
        self, responses: Dict[str, dict], dropped: Dict[str, str], client_timeout: float, threshold: int = 1
    ) -> bool:
        """A SecAgg round had one or more non-responders. For every pair
        (dropped_hospital, survivor_hospital), reconstruct that pair's
        mask via holder shares and subtract it from the survivor's
        submitted (masked) values in `responses` -- mutated in place.
        Returns True if every affected pair was successfully recovered
        (so the round can proceed as if nothing happened, modulo the
        dropped hospital's own data being absent); False if recovery
        couldn't be completed for some pair, in which case the caller
        must NOT trust `responses` and should fall back to discarding the
        round entirely.
        """
        import fl.shamir as shamir
        from fl.secure_agg import mask_from_seed

        for dropped_hospital in dropped:
            for survivor_hospital in list(responses.keys()):
                owner, peer = sorted([dropped_hospital, survivor_hospital])
                holders = [
                    h for h in holders_for_pair(self.hospitals, owner, peer)
                    if h in self.collaborators
                ]
                if not holders:
                    return False  # nobody left who could hold a share for this pair

                collected = []
                for holder_hospital in holders:
                    holder_url = self.collaborators[holder_hospital].base_url
                    try:
                        result = _post(
                            f"{holder_url}/reveal-share",
                            {"owner_hospital": owner, "peer_hospital": peer},
                            client_timeout, self._ssl_context,
                        )
                    except Exception:
                        continue
                    collected.append((result["x"], result["y"]))
                    if len(collected) >= threshold:
                        break

                if len(collected) < threshold:
                    return False  # not enough holders reachable to reconstruct

                seed = shamir.reconstruct(collected)
                dim = len(list_to_array(responses[survivor_hospital]["weights"])) + 1
                mask = mask_from_seed(seed, dim)

                sign = 1.0 if survivor_hospital < dropped_hospital else -1.0
                w = list_to_array(responses[survivor_hospital]["weights"])
                b = float(responses[survivor_hospital]["bias"])
                combined = np.concatenate([w, [b]]) - sign * mask
                responses[survivor_hospital] = {
                    **responses[survivor_hospital],
                    "weights": array_to_list(combined[:-1]),
                    "bias": float(combined[-1]),
                }
        return True

    def run(
        self,
        rounds: int,
        on_round_complete: Optional[Callable[[int, dict], None]] = None,
        seed: int = 0,
        client_timeout: float = 30.0,
        min_quorum_fraction: float = 0.5,
    ) -> dict:
        """Runs the federated training loop over real HTTP calls to real
        subprocesses/containers. Returns the final persisted model +
        per-round history.

        Fault tolerance, which this loop did NOT have before: collaborator
        calls now fire CONCURRENTLY (via a thread pool) instead of one at a
        time in sequence -- previously a single slow collaborator delayed
        every other collaborator's call behind it, even though they're
        fully independent requests. Each call also has its own
        `client_timeout`; a collaborator that times out or errors is
        DROPPED from that round rather than raising and failing the entire
        experiment (the previous behavior). A round only counts -- updates
        the global model -- if at least `min_quorum_fraction` of
        collaborators responded; otherwise the round is recorded as
        quorum-failed and the global model carries over unchanged into the
        next round, rather than training on a degraded/unrepresentative
        client set. Note this quorum check is intentionally conservative
        (no retry-with-backoff for transient failures within a round) --
        see the round's `dropped` list for what a production version would
        feed into a retry policy.
        """
        mean, std = self._federated_mean_std(client_timeout=client_timeout, min_quorum_fraction=min_quorum_fraction)

        rng = np.random.default_rng(seed=seed)
        global_w = rng.normal(scale=0.01, size=N_FEATURES)
        global_b = 0.0

        min_responders = max(1, math.ceil(len(self.hospitals) * min_quorum_fraction))
        history = []

        for round_idx in range(1, rounds + 1):
            secagg_recovered_this_round = False
            payload = {
                "disease_code": self.disease_code,
                "weights": array_to_list(global_w),
                "bias": global_b,
                "mean": array_to_list(mean),
                "std": array_to_list(std),
                "mu": self.mu,
                "epochs": self.local_epochs,
                "lr": self.lr,
                "dp_enabled": self.dp_enabled,
                "clip_norm": self.dp_clip_norm,
                "noise_multiplier": self.dp_noise_multiplier,
                "delta": self.dp_delta,
            }
            if self.secagg_enabled:
                payload["secagg_enabled"] = True
                payload["round_index"] = round_idx
                payload["peer_public_keys"] = self._secagg_public_keys
                payload["total_n"] = self._total_n
                try:
                    self._share_secagg_seeds(round_idx, client_timeout)
                except Exception as exc:  # noqa: BLE001
                    # Couldn't even set up dropout-recovery shares for this
                    # round -- treat identically to a mid-round dropout
                    # with no recovery possible: discard the round, carry
                    # the global model over unchanged, try again next round.
                    round_metrics = {
                        "status": "secagg_incomplete",
                        "reason": f"share setup failed: {exc}",
                        "required": len(self.collaborators),
                    }
                    history.append({"round": round_idx, **round_metrics})
                    if on_round_complete:
                        on_round_complete(round_idx, round_metrics)
                    continue

            responses: Dict[str, dict] = {}
            errors: Dict[str, str] = {}

            with ThreadPoolExecutor(max_workers=max(1, len(self.collaborators))) as pool:
                future_to_hospital = {
                    pool.submit(_post, f"{c.base_url}/train", payload, client_timeout, self._ssl_context): hospital
                    for hospital, c in self.collaborators.items()
                }
                for future in as_completed(future_to_hospital, timeout=client_timeout + 5.0):
                    hospital = future_to_hospital[future]
                    try:
                        responses[hospital] = future.result()
                    except Exception as exc:  # noqa: BLE001 -- any collaborator failure mode lands here
                        errors[hospital] = f"{type(exc).__name__}: {exc}"

            # SecAgg rounds used to be strictly all-or-nothing: any missing
            # collaborator left its mask partners' contributions
            # uncancelled, producing a wrong (not just incomplete) sum, so
            # the whole round had to be discarded. Now we first ATTEMPT
            # recovery via the Shamir shares distributed above -- if every
            # affected pair's seed can be reconstructed from its holders,
            # the leftover mask terms get cancelled out of the survivors'
            # submissions and the round proceeds using just their
            # (corrected) data. Only if recovery genuinely can't complete
            # (e.g. a pair's only holder is ALSO unreachable) does the
            # round still get discarded.
            if self.secagg_enabled and errors:
                recovered = self._attempt_secagg_recovery(responses, errors, client_timeout)
                if not recovered:
                    round_metrics = {
                        "status": "secagg_incomplete",
                        "responded": list(responses.keys()),
                        "dropped": errors,
                        "required": len(self.collaborators),
                        "recovery_attempted": True,
                    }
                    history.append({"round": round_idx, **round_metrics})
                    if on_round_complete:
                        on_round_complete(round_idx, round_metrics)
                    continue
                # else: responses now holds corrected values for every
                # survivor; fall through to normal aggregation below,
                # which will sum them (the dropped hospital's own data is
                # simply absent, same as any other fault-tolerant round --
                # but the SUM is no longer corrupted by its leftover masks)
                secagg_recovered_this_round = True

                # One more correction needed: each survivor's contribution
                # was pre-scaled INSIDE the collaborator using the
                # ORIGINAL total_n (from before the dropout was known) --
                # so naively summing the corrected values would UNDER-
                # weight the result by the dropped hospital's share,
                # rather than properly re-normalizing among just the
                # survivors the way the normal (non-SecAgg) fault-tolerant
                # path already does. Rescale by (original total_n) /
                # (surviving total_n) to match that same renormalization,
                # using each survivor's own n_samples -- already plaintext
                # metadata in the response, same as every other round.
                survivors_n = sum(r["n_samples"] for r in responses.values())
                if survivors_n > 0 and self._total_n:
                    rescale = self._total_n / survivors_n
                    for hospital in responses:
                        w = list_to_array(responses[hospital]["weights"]) * rescale
                        b = float(responses[hospital]["bias"]) * rescale
                        responses[hospital] = {
                            **responses[hospital],
                            "weights": array_to_list(w),
                            "bias": b,
                        }

            if len(responses) < min_responders:
                round_metrics = {
                    "status": "quorum_failed",
                    "responded": list(responses.keys()),
                    "dropped": errors,
                    "min_required": min_responders,
                    "of_total": len(self.hospitals),
                }
                history.append({"round": round_idx, **round_metrics})
                if on_round_complete:
                    on_round_complete(round_idx, round_metrics)
                continue  # global model carries over unchanged; try again next round

            local_weights, local_biases, sample_counts, val_metrics = [], [], [], []
            for hospital, result in responses.items():
                w, b = list_to_array(result["weights"]), float(result["bias"])
                n = result["n_samples"]
                local_weights.append(w)
                local_biases.append(b)
                sample_counts.append(n)
                val_metrics.append({
                    "hospital": hospital,
                    "val_n": result["val_n"],
                    "val_loss": result["val_loss"],
                    "val_acc": result["val_acc"],
                    "val_auc": result["val_auc"],
                    "dp": result.get("dp"),
                })

            total = sum(sample_counts)
            if self.secagg_enabled:
                # Client-side pre-scaling by weight fraction + pairwise
                # masking already happened inside each collaborator (see
                # fl/collaborator_server.py). Summing the masked values
                # directly is what makes the masks cancel AND recovers the
                # correctly-weighted average in one step -- re-weighting
                # by sample_counts here (like the fedavg branch does)
                # would double-apply the weighting AND break exact mask
                # cancellation, since the masks were derived independent
                # of any such rescaling.
                global_w = sum(local_weights)
                global_b = float(sum(local_biases))
            elif self.aggregation_strategy == "coordinate_median":
                # Ignores sample_counts entirely -- that's the point.
                # Weighting by a self-reported count is exactly what a
                # malicious client would lie about to gain influence.
                global_w = np.median(np.stack(local_weights, axis=0), axis=0)
                global_b = float(np.median(local_biases))
            else:
                global_w = sum(w * n for w, n in zip(local_weights, sample_counts)) / total
                global_b = sum(b * n for b, n in zip(local_biases, sample_counts)) / total

            # Aggregate validation metrics by weighting each client's
            # numbers by its own held-out sample count -- the aggregator
            # still never sees a single raw validation row.
            total_val_n = sum(m["val_n"] for m in val_metrics)
            agg_loss = sum(m["val_loss"] * m["val_n"] for m in val_metrics) / total_val_n
            agg_acc = sum(m["val_acc"] * m["val_n"] for m in val_metrics) / total_val_n
            agg_auc = sum(m["val_auc"] * m["val_n"] for m in val_metrics) / total_val_n

            round_metrics = {
                "status": "completed" if not errors else "completed_partial",
                "loss": agg_loss,
                "accuracy": agg_acc,
                "auc": agg_auc,
                "mu": self.mu,
                "dp_enabled": self.dp_enabled,
                "secagg_enabled": self.secagg_enabled,
                "secagg_recovered": secagg_recovered_this_round,
                "aggregation_strategy": self.aggregation_strategy,
                "per_client": val_metrics,
                "samples_per_client": dict(zip(responses.keys(), sample_counts)),
                "dropped": errors,
            }
            history.append({"round": round_idx, **round_metrics})
            if on_round_complete:
                on_round_complete(round_idx, round_metrics)

        return {
            "global_model": {
                "weights": array_to_list(global_w),
                "bias": global_b,
                "mean": array_to_list(mean),
                "std": array_to_list(std),
                "feature_order": ["age", "gender_female", "fever", "cough", "oxygen_level", "blood_pressure"],
            },
            "history": history,
        }
