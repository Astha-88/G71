"""Mutual TLS for the aggregator <-> collaborator protocol.

Closes two gaps at once, deliberately: (1) traffic between the aggregator
and each hospital's collaborator was plaintext HTTP -- anyone who could
reach the docker network could read or tamper with weight updates in
transit; (2) there was no authentication at all -- any process on the
network could POST to a collaborator's /train endpoint and feed it
arbitrary "global weights" to train against, or call it as if it were the
aggregator.

mTLS (mutual TLS -- both sides present a certificate, not just the server)
fixes both: traffic is encrypted, AND a collaborator refuses any connection
that doesn't present a certificate signed by the same private CA the
aggregator uses, so an unauthenticated process can't call /train at all.

Trust model: one self-signed CA, generated once (scripts/generate_certs.py,
run on the host -- NOT inside any container image, so the CA private key
never ends up baked into a Docker image alongside application code). The
aggregator gets one client certificate; each collaborator gets its own
server certificate, signed by the same CA. Every party trusts the CA cert
(public, fine to distribute) to verify whoever they're talking to; nobody
except whoever ran generate_certs.py ever has the CA's private key.
"""
from __future__ import annotations

import datetime
import ipaddress
import os
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def _generate_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def generate_ca(common_name: str = "Vienko FL Dev CA"):
    key = _generate_key()
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=True, key_cert_sign=True, crl_sign=True,
            content_commitment=False, key_encipherment=False, data_encipherment=False,
            key_agreement=False, encipher_only=False, decipher_only=False,
        ), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def generate_leaf_cert(ca_key, ca_cert, common_name: str, san_dns_names: list[str] | None = None):
    """Issues a cert signed by the CA. `san_dns_names` should include every
    hostname this cert will be presented under -- for a collaborator that's
    its docker-compose service name (e.g. "fl-hospital1") AND "localhost" /
    "127.0.0.1" so the same generated certs work in both subprocess-mode
    local dev and docker-compose network mode without regenerating.
    """
    key = _generate_key()
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    san_dns_names = san_dns_names or [common_name]

    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(n) for n in san_dns_names] + [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
    )
    cert = builder.sign(ca_key, hashes.SHA256())
    return key, cert


def write_key(path: str, key) -> None:
    with open(path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    os.chmod(path, 0o600)  # private keys should not be world-readable


def write_cert(path: str, cert) -> None:
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def build_server_context(cert_path: str, key_path: str, ca_path: str) -> ssl.SSLContext:
    """For a collaborator: present its own cert, and REQUIRE + verify the
    caller's client cert against the CA -- this is what actually enforces
    "only the real aggregator, or something holding its cert, can call
    /train", not just "traffic happens to be encrypted."
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    ctx.load_verify_locations(cafile=ca_path)
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def build_client_context(cert_path: str, key_path: str, ca_path: str) -> ssl.SSLContext:
    """For the aggregator: present its own client cert, and verify the
    collaborator's server cert against the same CA -- INCLUDING hostname
    verification (check_hostname=True). This matters more than it looks:
    with it on, if the aggregator dials "fl-hospital1" but something
    presents fl-hospital2's certificate (misconfiguration, DNS spoofing,
    a relabeled container), the handshake fails outright, because
    fl-hospital2's cert only lists its own name in its SAN entries (see
    generate_leaf_cert). It works in both subprocess-mode (SAN includes
    127.0.0.1) and docker-compose network mode (SAN includes the service
    hostname) because scripts/generate_certs.py issues each collaborator's
    cert with both.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=ca_path)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx
