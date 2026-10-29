"""Generates a private CA and per-service certificates for mTLS between the
aggregator and the fl-hospitalN collaborators.

Run ONCE, on the host, before starting the stack:
    python3 backend/scripts/generate_certs.py

This deliberately runs on the host, not inside any Docker build, for the
same reason backend/data/ isn't baked into the backend image: the CA
PRIVATE KEY must never end up inside a container image or version control.
Only the leaf certs (+ their own private keys) and the CA's PUBLIC cert get
mounted into containers; certs/ca.key never leaves this directory.

Output layout:
  backend/certs/ca.crt              (public -- distributed to everyone)
  backend/certs/ca.key              (SECRET -- stays on the host, .gitignored)
  backend/certs/aggregator.crt/.key (the backend/aggregator's client identity)
  backend/certs/fl-hospital{1,2,3}.crt/.key  (each collaborator's server identity)

NOTE on subprocess (local, non-docker) mode: every collaborator's cert
includes 127.0.0.1 in its Subject Alternative Names so local dev works
without regenerating certs. Because subprocess mode has all collaborators
on the same loopback IP distinguished only by PORT, and TLS hostname
verification operates on hostname/IP, not port, mTLS in that mode mainly
demonstrates the mechanism and protects against anything outside this
machine -- it does NOT distinguish "the real hospital1 process" from "some
other local process presenting hospital2's cert but bound to hospital1's
port." Docker-compose network mode does NOT have this weakness: each
collaborator has its own unique service hostname (fl-hospital1 etc.), so
hostname verification there is a real, meaningful check.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fl.tls import generate_ca, generate_leaf_cert, write_cert, write_key  # noqa: E402

CERTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "certs")
HOSPITALS = ["hospital1", "hospital2", "hospital3"]


def main() -> None:
    os.makedirs(CERTS_DIR, exist_ok=True)

    ca_path = os.path.join(CERTS_DIR, "ca.crt")
    ca_key_path = os.path.join(CERTS_DIR, "ca.key")
    if os.path.exists(ca_path):
        print(f"CA already exists at {ca_path} -- delete backend/certs/ entirely to regenerate everything.")
        return

    print("Generating CA...")
    ca_key, ca_cert = generate_ca()
    write_key(ca_key_path, ca_key)
    write_cert(ca_path, ca_cert)

    print("Generating aggregator client cert...")
    agg_key, agg_cert = generate_leaf_cert(ca_key, ca_cert, "aggregator", san_dns_names=["aggregator", "backend", "localhost"])
    write_key(os.path.join(CERTS_DIR, "aggregator.key"), agg_key)
    write_cert(os.path.join(CERTS_DIR, "aggregator.crt"), agg_cert)

    for hospital in HOSPITALS:
        print(f"Generating server cert for {hospital}...")
        # fl-{hospital} is the docker-compose service hostname; {hospital}
        # and localhost cover subprocess/local-dev mode.
        san = [f"fl-{hospital}", hospital, "localhost"]
        key, cert = generate_leaf_cert(ca_key, ca_cert, f"fl-{hospital}", san_dns_names=san)
        write_key(os.path.join(CERTS_DIR, f"{hospital}.key"), key)
        write_cert(os.path.join(CERTS_DIR, f"{hospital}.crt"), cert)

    print(f"\nDone. Certs written to {CERTS_DIR}/")
    print("IMPORTANT: certs/ca.key is a secret. It's covered by .gitignore already -- keep it that way.")


if __name__ == "__main__":
    main()
