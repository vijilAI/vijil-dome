"""Agent identity — SPIFFE attestation with API key fallback.

Three identity modes, tried in order:
1. X.509 SVID via local SPIRE Agent socket (EKS-hosted agents)
2. JWT-SVID via Identity Delegate Service (managed AgentCore agents)
3. Static API key (development fallback)
"""

from __future__ import annotations

import logging
import os
import ssl
from typing import ClassVar

logger = logging.getLogger(__name__)

# Optional dependency: spiffe (SPIRE Workload API client)
_HAS_SPIFFE = False
try:
    from spiffe import WorkloadApiClient
    _HAS_SPIFFE = True
except ImportError:
    pass


class AgentIdentity:
    """Represents the cryptographic identity of an agent workload.

    Three identity modes, tried in order:
    1. X.509 SVID via SPIRE Agent socket (if socket exists + py-spiffe installed)
    2. JWT-SVID via Identity Delegate Service (if delegate_url provided)
    3. Static API key fallback
    """

    _UNSET: ClassVar[object] = object()

    def __init__(
        self,
        spire_socket: str = "/run/spire/sockets/agent.sock",
        delegate_url: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self._spire_socket = spire_socket
        self._delegate_url = delegate_url or os.environ.get("VIJIL_IDENTITY_DELEGATE_URL")
        self._agent_name = agent_name
        self._spiffe_id: str | None = None
        self._svid: object | None = None  # spiffe X509Svid when attested (mode 1)
        self._jwt_svid: str | None = None  # JWT-SVID token string (mode 2)
        self._trust_bundle: object | None = None
        self._api_key: str | None = None
        self._attested: bool = False

        # Try modes in order
        self._try_spire_attestation()
        if not self._attested and self._delegate_url:
            self._try_delegate_attestation()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_api_key(cls, api_key: str) -> AgentIdentity:
        """Create an unattested identity backed by a static API key."""
        identity = cls.__new__(cls)
        identity._spire_socket = ""
        identity._delegate_url = None
        identity._agent_name = None
        identity._spiffe_id = None
        identity._svid = None
        identity._jwt_svid = None
        identity._trust_bundle = None
        identity._api_key = api_key
        identity._attested = False
        return identity

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_spire_attestation(self) -> None:
        """Attempt SPIFFE attestation via the SPIRE socket.

        Logs a warning and remains unattested on any failure.
        """
        if not _HAS_SPIFFE:
            logger.warning(
                "spiffe package is not installed; SPIFFE attestation unavailable. "
                "Install with: pip install spiffe"
            )
            return

        if not os.path.exists(self._spire_socket):
            logger.warning(
                "SPIRE socket not found at %s; agent will run unattested.",
                self._spire_socket,
            )
            return

        try:
            socket_path = f"unix://{self._spire_socket}"
            client = WorkloadApiClient(socket_path)
            svid = client.fetch_x509_svid()
            bundles = client.fetch_x509_bundles()
            client.close()
            self._svid = svid
            self._spiffe_id = str(svid.spiffe_id)
            if bundles and hasattr(bundles, "get_x509_authorities"):
                self._trust_bundle = bundles.get_x509_authorities()
            self._attested = True
            logger.info("SPIFFE attestation successful: %s", self._spiffe_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SPIFFE attestation failed: %s. Agent will run unattested.", exc
            )

    def _try_delegate_attestation(self) -> None:
        """Fetch a JWT-SVID from the Identity Delegate Service.

        Used by agents in managed runtimes (AgentCore microVMs) that
        cannot access a local SPIRE Agent socket.
        """
        if not self._delegate_url or not self._agent_name:
            logger.info(
                "Delegate attestation skipped: delegate_url=%s, agent_name=%s",
                self._delegate_url,
                self._agent_name,
            )
            return

        try:
            import httpx

            response = httpx.post(
                f"{self._delegate_url.rstrip('/')}/v1/identity/jwt-svid",
                json={
                    # TODO: Replace with real AWS STS presigned GetCallerIdentity
                    # request once the Identity Delegate implements full validation.
                    # Currently the delegate accepts "auto" and uses its own creds.
                    "aws_identity_token": "auto",
                    "agent_name": self._agent_name,
                    "audience": ["vijil"],
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            self._jwt_svid = data["jwt_svid"]
            self._spiffe_id = data["spiffe_id"]
            self._attested = True
            logger.info(
                "JWT-SVID attestation via delegate successful: %s",
                self._spiffe_id,
            )
        except ImportError:
            logger.warning("httpx not installed; delegate attestation unavailable.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Delegate attestation failed: %s. Agent will run unattested.",
                exc,
            )

    async def _try_delegate_attestation_async(self) -> None:
        """Async variant of delegate attestation using httpx.AsyncClient."""
        if not self._delegate_url or not self._agent_name:
            return

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._delegate_url.rstrip('/')}/v1/identity/jwt-svid",
                    json={
                        # TODO: Replace with real AWS STS presigned GetCallerIdentity
                        "aws_identity_token": "auto",
                        "agent_name": self._agent_name,
                        "audience": ["vijil"],
                    },
                )
                response.raise_for_status()
                data = response.json()

            self._jwt_svid = data["jwt_svid"]
            self._spiffe_id = data["spiffe_id"]
            self._attested = True
            logger.info(
                "async JWT-SVID attestation via delegate successful: %s",
                self._spiffe_id,
            )
        except ImportError:
            logger.warning("httpx not installed; delegate attestation unavailable.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Async delegate attestation failed: %s. Agent will run unattested.",
                exc,
            )

    async def attest_async(self) -> None:
        """Async attestation — tries SPIFFE (in thread), then async delegate."""
        if self._attested:
            return

        # SPIFFE attestation is a sync gRPC call — run in a thread
        if _HAS_SPIFFE and os.path.exists(self._spire_socket):
            import asyncio
            await asyncio.to_thread(self._try_spiffe_attestation)
            if self._attested:
                return

        # Delegate attestation — native async
        await self._try_delegate_attestation_async()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spiffe_id(self) -> str | None:
        """The SPIFFE ID from the SVID, or None if not attested."""
        return self._spiffe_id

    @property
    def jwt_svid(self) -> str | None:
        """The JWT-SVID token, or None if not using delegate attestation."""
        return self._jwt_svid

    @property
    def api_key(self) -> str | None:
        """Static API key used when SPIFFE attestation is unavailable."""
        return self._api_key

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def is_attested(self) -> bool:
        """Return True if an SVID (X.509 or JWT) was successfully obtained."""
        return self._attested

    def mtls_context(self) -> ssl.SSLContext:
        """Return an SSL context configured with the SVID for mTLS client auth.

        The context uses the SVID certificate as the client cert and the
        trust bundle as the CA for verifying server certificates.

        Raises:
            RuntimeError: If the agent is not SPIFFE-attested.
        """
        if not self._attested or self._svid is None:
            raise RuntimeError(
                "mTLS context requires SPIFFE attestation. "
                "The agent is currently unattested. "
                "Ensure the SPIRE agent socket is accessible and the spiffe package is installed."
            )

        try:
            from cryptography.hazmat.primitives.serialization import (
                Encoding,
                NoEncryption,
                PrivateFormat,
            )

            leaf = self._svid.leaf  # type: ignore[attr-defined]
            private_key = self._svid.private_key  # type: ignore[attr-defined]

            cert_pem = leaf.public_bytes(Encoding.PEM)
            key_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())

            # Load cert + key in memory via pyOpenSSL (no temp files).
            # Falls back to temp files if pyOpenSSL is not installed.
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            try:
                from OpenSSL.crypto import (  # type: ignore[import-untyped]
                    load_certificate,
                    load_privatekey,
                    FILETYPE_PEM,
                )
                from OpenSSL.SSL import Context, TLSv1_2_METHOD  # type: ignore[import-untyped]

                ossl_ctx = Context(TLSv1_2_METHOD)
                ossl_ctx.use_certificate(load_certificate(FILETYPE_PEM, cert_pem))
                ossl_ctx.use_privatekey(load_privatekey(FILETYPE_PEM, key_pem))

                # Extract the stdlib SSLContext from pyOpenSSL
                ctx = ossl_ctx._context  # type: ignore[attr-defined]
                if not isinstance(ctx, ssl.SSLContext):
                    # pyOpenSSL internals changed — fall back to temp files
                    raise ImportError("Cannot extract stdlib SSLContext from pyOpenSSL")
            except ImportError:
                # pyOpenSSL not available — use temp files with cleanup
                import tempfile
                import os

                cert_path = key_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as cert_f:
                        cert_f.write(cert_pem)
                        cert_path = cert_f.name
                    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as key_f:
                        key_f.write(key_pem)
                        key_path = key_f.name
                    ctx.load_cert_chain(cert_path, key_path)
                finally:
                    if cert_path and os.path.exists(cert_path):
                        os.unlink(cert_path)
                    if key_path and os.path.exists(key_path):
                        os.unlink(key_path)

            # Load trust bundle for verifying peer certificates
            if hasattr(self, "_trust_bundle") and self._trust_bundle is not None:
                for ca_cert in self._trust_bundle:  # type: ignore[attr-defined]
                    ca_pem = ca_cert.public_bytes(Encoding.PEM).decode()
                    ctx.load_verify_locations(cadata=ca_pem)

            return ctx
        except Exception as exc:
            raise RuntimeError(f"Failed to build mTLS context from SVID: {exc}") from exc

    def auth_header(self) -> dict[str, str]:
        """Return the HTTP Authorization header for this identity.

        - API key identity: ``{"authorization": "Bearer <key>"}``
        - JWT-SVID identity: ``{"authorization": "Bearer <jwt>"}``
        - X.509 SVID identity (mTLS): returns empty dict — the TLS
          layer carries the identity; no additional header is needed.
        """
        if self._api_key is not None:
            return {"authorization": f"Bearer {self._api_key}"}
        if self._jwt_svid is not None:
            return {"authorization": f"Bearer {self._jwt_svid}"}
        return {}
