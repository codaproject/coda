"""Local certificate authority for serving the web app over HTTPS.

The authority is created once and stays in place, so a phone that has already
trusted it does not need to be set up again. The leaf certificate is reissued
when the laptop's names or addresses change, or when it is close to expiry.
"""

import html
import ipaddress
import logging
import os
import re
import socket
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from starlette.requests import Request

logger = logging.getLogger(__name__)

CA_CERT_NAME = "ca.pem"
CA_KEY_NAME = "ca-key.pem"
LEAF_CERT_NAME = "cert.pem"
LEAF_KEY_NAME = "key.pem"

_CA_LIFETIME = timedelta(days=3650)
_LEAF_LIFETIME = timedelta(days=730)
_RENEW_BEFORE = timedelta(days=30)
_DNS_LABEL = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")


class TlsMaterial:
    """Paths to the authority certificate and the leaf the server presents."""

    def __init__(self, directory: Path, ca_cert: Path, leaf_cert: Path, leaf_key: Path):
        self.directory = directory
        self.ca_cert = ca_cert
        self.leaf_cert = leaf_cert
        self.leaf_key = leaf_key


def ensure_certs(directory, names=None) -> TlsMaterial:
    """Create a stable local CA and a leaf certificate for the current names.

    Parameters
    ----------
    directory : path-like
        Directory that holds the authority and the leaf. Created if missing.
    names : list of str or None
        DNS names and IP addresses to put on the leaf. When omitted, the
        current host names and addresses are used.

    Returns
    -------
    :
        Paths to the authority certificate and the leaf certificate and key.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ca_cert_path = directory / CA_CERT_NAME
    ca_key_path = directory / CA_KEY_NAME
    leaf_cert_path = directory / LEAF_CERT_NAME
    leaf_key_path = directory / LEAF_KEY_NAME

    desired = _dedupe(list(names) if names is not None else _host_names())
    ca_cert, ca_key = _ensure_ca(ca_cert_path, ca_key_path)
    if _leaf_is_current(leaf_cert_path, leaf_key_path, desired):
        logger.info("Leaf certificate already matches %s", ", ".join(desired))
    else:
        logger.info("Issuing leaf certificate for %s", ", ".join(desired))
        _issue_leaf(leaf_cert_path, leaf_key_path, ca_cert, ca_key, desired)
    return TlsMaterial(directory, ca_cert_path, leaf_cert_path, leaf_key_path)


def create_setup_app(ca_cert: Path, https_port: int) -> FastAPI:
    """Build the plain-HTTP app that helps a phone trust this laptop.

    Parameters
    ----------
    ca_cert :
        Authority certificate offered for download. The private key is never
        read here.
    https_port :
        Port where the interview app is served over HTTPS.

    Returns
    -------
    :
        Application with the setup page and the authority certificate.
    """
    ca_cert = Path(ca_cert)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def setup_page(request: Request):
        interview_url = f"https://{_hostname_from_request(request)}:{https_port}"
        return HTMLResponse(_setup_html(interview_url))

    @app.get("/ca.pem")
    def download_ca():
        if not ca_cert.is_file():
            raise HTTPException(status_code=404, detail="Certificate is not ready")
        return FileResponse(
            ca_cert,
            media_type="application/x-pem-file",
            filename="coda-ca.pem",
        )

    return app


def start_setup_server(ca_cert: Path, host: str, port: int, https_port: int) -> None:
    """Serve the phone setup page over plain HTTP in the background.

    Parameters
    ----------
    ca_cert :
        Authority certificate the page offers for download.
    host :
        Interface to bind.
    port :
        Plain HTTP port for the setup page.
    https_port :
        Port where the interview app is served over HTTPS.
    """
    setup = create_setup_app(ca_cert, https_port)
    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": setup,
            "host": host,
            "port": port,
            "log_config": None,
        },
        daemon=True,
        name="coda-tls-setup",
    )
    thread.start()
    logger.info("Interview UI: https://localhost:%s", https_port)
    addresses = _ipv4_addresses()
    if not addresses:
        logger.info("Phone setup: http://<this-computer>:%s", port)
        return
    for address in addresses:
        logger.info("Phone setup: http://%s:%s", address, port)


def _ensure_ca(cert_path: Path, key_path: Path):
    cert_exists = cert_path.is_file()
    key_exists = key_path.is_file()
    if cert_exists and key_exists:
        cert = _load_cert(cert_path)
        key = _load_key(key_path)
        if cert is not None and key is not None:
            return cert, key
        raise RuntimeError(
            f"Could not read the certificate authority in {cert_path.parent}. "
            "Leave those files in place, or remove both and start again."
        )
    if cert_exists or key_exists:
        raise RuntimeError(
            f"The certificate authority in {cert_path.parent} is incomplete. "
            "Leave a matching ca.pem and ca-key.pem in place, or remove both "
            "and start again."
        )
    logger.info("Creating local certificate authority in %s", cert_path.parent)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CODA Local CA")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + _CA_LIFETIME)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    _write_secret(key_path, _key_pem(key))
    _write_cert(cert_path, cert)
    return cert, key


def _issue_leaf(cert_path, key_path, ca_cert, ca_key, names: list[str]) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    san = []
    for name in names:
        ip = _parse_ip(name)
        if ip is not None:
            san.append(x509.IPAddress(ip))
        else:
            san.append(x509.DNSName(name))
    ski = ca_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "CODA"),
        ]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + _LEAF_LIFETIME)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_secret(key_path, _key_pem(key))
    _write_cert(cert_path, cert)


def _leaf_is_current(cert_path: Path, key_path: Path, desired: list[str]) -> bool:
    if not cert_path.is_file() or not key_path.is_file():
        return False
    cert = _load_cert(cert_path)
    key = _load_key(key_path)
    if cert is None or key is None:
        return False
    if _public_pem(cert.public_key()) != _public_pem(key.public_key()):
        return False
    if _certificate_names(cert) != set(desired):
        return False
    now = datetime.now(timezone.utc)
    not_before = _as_utc(_not_before(cert))
    not_after = _as_utc(_not_after(cert))
    if now < not_before:
        return False
    return not_after - now >= _RENEW_BEFORE


def _host_names() -> list[str]:
    names = ["localhost", "127.0.0.1"]
    short = socket.gethostname().split(".")[0].strip().lower()
    if short and short != "localhost" and _is_dns_name(short):
        names.append(f"{short}.local")
    names.extend(_ipv4_addresses())
    return names


def _ipv4_addresses() -> list[str]:
    from_command, command_ok = _ipv4_from_ip_command()
    if command_ok:
        return from_command
    return _ipv4_from_socket()


def _ipv4_from_ip_command():
    try:
        result = subprocess.run(
            ["ip", "-4", "-o", "addr"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return [], False
    if result.returncode != 0:
        return [], False
    addresses = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        cidr = parts[parts.index("inet") + 1]
        address = cidr.split("/")[0]
        if _usable_ipv4(address) and address not in addresses:
            addresses.append(address)
    return addresses, True


def _ipv4_from_socket() -> list[str]:
    addresses = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            address = sock.getsockname()[0]
            if _usable_ipv4(address):
                addresses.append(address)
    except OSError:
        pass
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    for info in infos:
        address = info[4][0]
        if _usable_ipv4(address) and address not in addresses:
            addresses.append(address)
    return addresses


def _usable_ipv4(address: str) -> bool:
    ip = _parse_ip(address)
    if not isinstance(ip, ipaddress.IPv4Address):
        return False
    return not (ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast)


def _dedupe(names: list[str]) -> list[str]:
    ordered = []
    seen = set()
    for name in names:
        key = _name_key(name)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    if not ordered:
        raise ValueError("At least one certificate name is required")
    return ordered


def _name_key(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("Certificate name must not be empty")
    ip = _parse_ip(text)
    if ip is not None:
        return str(ip)
    dns = text.lower().rstrip(".")
    if not _is_dns_name(dns):
        raise ValueError(f"Not a DNS name or IP address: {name}")
    return dns


def _parse_ip(name: str):
    try:
        return ipaddress.ip_address(str(name).strip())
    except ValueError:
        return None


def _is_dns_name(name: str) -> bool:
    if not name or len(name) > 253:
        return False
    return all(_DNS_LABEL.match(label) for label in name.split("."))


def _certificate_names(cert) -> set[str]:
    try:
        extension = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    except x509.ExtensionNotFound:
        return set()
    names = {value.lower() for value in extension.value.get_values_for_type(x509.DNSName)}
    names.update(str(value) for value in extension.value.get_values_for_type(x509.IPAddress))
    return names


def _load_cert(path: Path):
    try:
        return x509.load_pem_x509_certificate(path.read_bytes())
    except ValueError:
        return None


def _load_key(path: Path):
    try:
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    except ValueError:
        return None


def _public_pem(key) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _key_pem(key) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _write_cert(path: Path, cert) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _write_secret(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _not_before(cert):
    value = getattr(cert, "not_valid_before_utc", None)
    if value is None:
        value = cert.not_valid_before
    return value


def _not_after(cert):
    value = getattr(cert, "not_valid_after_utc", None)
    if value is None:
        value = cert.not_valid_after
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _hostname_from_request(request: Request) -> str:
    host_header = request.headers.get("host", "")
    if host_header.startswith("["):
        end = host_header.find("]")
        if end != -1:
            return host_header[1:end]
    if ":" in host_header:
        return host_header.rsplit(":", 1)[0]
    return host_header or "localhost"


def _setup_html(interview_url: str) -> str:
    safe_url = html.escape(interview_url, quote=True)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Connect this phone to CODA</title>
  <style>
    body {{
      font-family: sans-serif;
      line-height: 1.45;
      max-width: 38rem;
      margin: 1.5rem auto;
      padding: 0 1rem 2rem;
      color: #1c1917;
      background: #faf7f2;
    }}
    h1 {{ font-size: 1.6rem; }}
    a.button {{
      display: inline-block;
      margin: 0.4rem 0.6rem 0.4rem 0;
      padding: 0.7rem 1rem;
      background: #1c1917;
      color: #faf7f2;
      text-decoration: none;
      border-radius: 0.4rem;
    }}
    ol {{ padding-left: 1.2rem; }}
    li {{ margin: 0.35rem 0; }}
  </style>
</head>
<body>
  <h1>Connect this phone to CODA</h1>
  <p>Install the laptop's certificate once. After that, this phone can open the interview and use the microphone.</p>
  <p><a class="button" href="/ca.pem">Install the CODA certificate</a></p>
  <h2>Android</h2>
  <ol>
    <li>Tap <strong>Install the CODA certificate</strong> to download the file.</li>
    <li>Navigate to Settings &gt; Security (or Biometrics and security) &gt; (Other security settings &gt;) Install from device storage, and install the downloaded file as a CA certificate. The phone may ask you to set a screen lock.</li>
  </ol>
  <h2>iPhone or iPad</h2>
  <ol>
    <li>Tap <strong>Install the CODA certificate</strong> and allow the profile download.</li>
    <li>Open Settings and install the downloaded profile.</li>
    <li>Turn on full trust under Settings &gt; General &gt; About &gt; Certificate Trust Settings.</li>
    <li>Under "Enable Full Trust for Root Certificates," locate the installed CA and toggle the switch to On.</li>
  </ol>
  <h2>Open the interview</h2>
  <p>When the certificate is installed, open CODA:</p>
  <p><a class="button" href="{safe_url}">Open CODA</a></p>
  <p>The address is <a href="{safe_url}">{safe_url}</a>.</p>
  <p>Installing the certificate does not allow the microphone to be used. If the microphone button does nothing, check the browser's site settings, set Microphone to Allow for this site, and reload the page.</p>
</body>
</html>
"""
