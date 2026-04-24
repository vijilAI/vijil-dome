"""Tests for _verify_tool_identity — the real mTLS implementation."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock

import pytest

from vijil_dome.trust.runtime import TrustRuntime

CONSTRAINTS = {
    "agent_id": "test",
    "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
    "tool_permissions": [],
    "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
    "enforcement_mode": "warn",
    "updated_at": "2026-04-12T00:00:00Z",
}


class TestParseEndpoint:
    def test_mcp_tls_scheme(self):
        host, port = TrustRuntime._parse_endpoint("mcp+tls://flights.internal:8443")
        assert host == "flights.internal"
        assert port == 8443

    def test_https_scheme(self):
        host, port = TrustRuntime._parse_endpoint("https://api.example.com:443")
        assert host == "api.example.com"
        assert port == 443

    def test_no_scheme(self):
        host, port = TrustRuntime._parse_endpoint("tool.svc:9090")
        assert host == "tool.svc"
        assert port == 9090

    def test_default_port(self):
        host, port = TrustRuntime._parse_endpoint("https://tool.svc")
        assert host == "tool.svc"
        assert port == 443


class TestVerifyToolIdentity:
    @pytest.fixture
    def runtime(self):
        return TrustRuntime(agent_id="test", constraints=CONSTRAINTS, mode="warn")

    def test_local_endpoint_skips_verification(self, runtime):
        """Tools with endpoint='local' are verified without network call."""
        tool = MagicMock()
        tool.name = "search_flights"
        tool.identity = "spiffe://vijil.ai/tools/flights/v1"
        tool.endpoint = "local"

        result = runtime._verify_tool_identity(tool)
        assert result.verified
        assert result.error is None

    def test_empty_endpoint_skips_verification(self, runtime):
        """Tools with no endpoint skip verification."""
        tool = MagicMock()
        tool.name = "search"
        tool.identity = "spiffe://vijil.ai/tools/search/v1"
        tool.endpoint = ""

        result = runtime._verify_tool_identity(tool)
        assert result.verified

    def test_unreachable_endpoint_returns_error(self, runtime):
        """Tool at unreachable endpoint returns verification failure."""
        tool = MagicMock()
        tool.name = "remote_tool"
        tool.identity = "spiffe://vijil.ai/tools/remote/v1"
        tool.endpoint = "mcp+tls://nonexistent.internal:8443"

        result = runtime._verify_tool_identity(tool)
        assert not result.verified
        assert "Verification failed" in result.error

    def test_bad_endpoint_format(self, runtime):
        """Unparseable endpoint returns error."""
        tool = MagicMock()
        tool.name = "bad_tool"
        tool.identity = "spiffe://vijil.ai/tools/bad/v1"
        tool.endpoint = "mcp+tls://host:notaport"

        result = runtime._verify_tool_identity(tool)
        assert not result.verified


class TestExtractSpiffeId:
    def test_extracts_from_real_cert(self):
        """Extract SPIFFE ID from a self-signed cert with SAN URI."""
        from datetime import datetime, timedelta

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        # Generate a test cert with a SPIFFE ID in the SAN
        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-tool")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC))
            .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.UniformResourceIdentifier("spiffe://vijil.ai/ns/default/tool/echo-tool/v1"),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        from cryptography.hazmat.primitives.serialization import Encoding
        cert_der = cert.public_bytes(Encoding.DER)
        result = TrustRuntime._extract_spiffe_id_from_cert(cert_der)
        assert result == "spiffe://vijil.ai/ns/default/tool/echo-tool/v1"

    def test_returns_none_for_cert_without_spiffe(self):
        """Cert without SPIFFE SAN returns None."""
        from datetime import datetime, timedelta

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "regular-server")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC))
            .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("regular-server.example.com"),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        from cryptography.hazmat.primitives.serialization import Encoding
        cert_der = cert.public_bytes(Encoding.DER)
        result = TrustRuntime._extract_spiffe_id_from_cert(cert_der)
        assert result is None
