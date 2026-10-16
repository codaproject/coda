"""Tests for the local certificate authority used by the phone setup."""

import stat
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from fastapi.testclient import TestClient

from coda.app.tls import (
    CA_CERT_NAME,
    CA_KEY_NAME,
    LEAF_CERT_NAME,
    LEAF_KEY_NAME,
    create_setup_app,
    ensure_certs,
)

NAMES = ["localhost", "laptop.local", "127.0.0.1", "10.1.2.3"]


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _load_cert(path):
    return x509.load_pem_x509_certificate(path.read_bytes())


def _names(path) -> set[str]:
    san = _load_cert(path).extensions.get_extension_for_class(
        x509.SubjectAlternativeName,
    ).value
    found = {value.lower() for value in san.get_values_for_type(x509.DNSName)}
    found.update(str(value) for value in san.get_values_for_type(x509.IPAddress))
    return found


def test_authority_is_created_once_and_leaf_keeps_its_names(tmp_path):
    first = ensure_certs(tmp_path, NAMES)
    ca_bytes = first.ca_cert.read_bytes()
    leaf_bytes = first.leaf_cert.read_bytes()

    second = ensure_certs(tmp_path, ["LocalHost", "10.1.2.3", "laptop.local", "127.0.0.1"])

    assert second.ca_cert.read_bytes() == ca_bytes
    assert second.leaf_cert.read_bytes() == leaf_bytes
    assert _names(first.leaf_cert) == set(NAMES)
    assert _mode(tmp_path / CA_KEY_NAME) == 0o600
    assert _mode(tmp_path / LEAF_KEY_NAME) == 0o600

    ca_cert = _load_cert(first.ca_cert)
    leaf = _load_cert(first.leaf_cert)
    ca_cert.public_key().verify(
        leaf.signature,
        leaf.tbs_certificate_bytes,
        padding.PKCS1v15(),
        leaf.signature_hash_algorithm,
    )


def test_new_address_reissues_the_leaf_only(tmp_path):
    ensure_certs(tmp_path, ["localhost", "127.0.0.1"])
    ca_bytes = (tmp_path / CA_CERT_NAME).read_bytes()
    leaf_bytes = (tmp_path / LEAF_CERT_NAME).read_bytes()

    ensure_certs(tmp_path, ["localhost", "127.0.0.1", "192.168.4.20"])

    assert (tmp_path / CA_CERT_NAME).read_bytes() == ca_bytes
    assert (tmp_path / CA_KEY_NAME).read_bytes()
    assert (tmp_path / LEAF_CERT_NAME).read_bytes() != leaf_bytes
    assert _names(tmp_path / LEAF_CERT_NAME) == {
        "localhost",
        "127.0.0.1",
        "192.168.4.20",
    }


def test_near_expiry_reissues_the_leaf_only(tmp_path):
    ensure_certs(tmp_path, ["localhost", "127.0.0.1"])
    ca_bytes = (tmp_path / CA_CERT_NAME).read_bytes()
    ca_key = serialization.load_pem_private_key(
        (tmp_path / CA_KEY_NAME).read_bytes(),
        password=None,
    )
    leaf = _load_cert(tmp_path / LEAF_CERT_NAME)
    leaf_key = serialization.load_pem_private_key(
        (tmp_path / LEAF_KEY_NAME).read_bytes(),
        password=None,
    )
    now = datetime.now(timezone.utc)
    short_lived = (
        x509.CertificateBuilder()
        .subject_name(leaf.subject)
        .issuer_name(_load_cert(tmp_path / CA_CERT_NAME).subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=10))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            leaf.extensions.get_extension_for_class(
                x509.SubjectAlternativeName,
            ).value,
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    short_bytes = short_lived.public_bytes(serialization.Encoding.PEM)
    (tmp_path / LEAF_CERT_NAME).write_bytes(short_bytes)

    ensure_certs(tmp_path, ["localhost", "127.0.0.1"])

    assert (tmp_path / CA_CERT_NAME).read_bytes() == ca_bytes
    assert (tmp_path / LEAF_CERT_NAME).read_bytes() != short_bytes
    assert _names(tmp_path / LEAF_CERT_NAME) == {"localhost", "127.0.0.1"}


def test_default_names_include_localhost(tmp_path):
    material = ensure_certs(tmp_path)
    found = _names(material.leaf_cert)
    assert "localhost" in found
    assert "127.0.0.1" in found


def test_incomplete_authority_is_left_in_place(tmp_path):
    (tmp_path / CA_CERT_NAME).write_text("partial", encoding="utf-8")

    with pytest.raises(RuntimeError):
        ensure_certs(tmp_path, ["localhost"])

    assert (tmp_path / CA_CERT_NAME).read_text(encoding="utf-8") == "partial"
    assert not (tmp_path / CA_KEY_NAME).exists()


def test_setup_page_serves_the_authority_and_not_the_key(tmp_path):
    material = ensure_certs(tmp_path, ["localhost", "127.0.0.1"])
    key_text = (tmp_path / CA_KEY_NAME).read_text(encoding="utf-8")
    client = TestClient(create_setup_app(material.ca_cert, https_port=8000))

    page = client.get("/", headers={"Host": "192.168.1.20:8001"})
    downloaded = client.get("/ca.pem")

    assert page.status_code == 200
    assert "https://192.168.1.20:8000" in page.text
    assert "Install from device storage" in page.text
    assert "Certificate Trust Settings" in page.text
    assert "Enable Full Trust for Root Certificates" in page.text
    assert "CA certificate" in page.text
    assert "screen lock" in page.text
    assert "set Microphone to Allow" in page.text
    assert key_text not in page.text
    assert downloaded.status_code == 200
    assert downloaded.content == material.ca_cert.read_bytes()
    assert b"PRIVATE KEY" not in downloaded.content
    for path in ("/ca-key.pem", "/key.pem", "/openapi.json", "/docs"):
        missing = client.get(path)
        assert missing.status_code == 404
        assert key_text not in missing.text
