"""TrustRuntime — orchestrates identity, constraints, guards, MAC, and audit."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vijil_dome.trust.attestation import AttestationResult, ToolAttestationStatus
from vijil_dome.trust.audit import AuditEmitter
from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.guard import GuardResult
from vijil_dome.trust.identity import AgentIdentity
from vijil_dome.trust.manifest import ToolManifest
from vijil_dome.trust.policy import ToolCallResult, ToolPolicy

logger = logging.getLogger(__name__)

_HAS_DOME = True  # Dome is always available — trust runtime lives inside vijil-dome.


class TrustRuntime:
    """Core orchestrator composing identity, constraints, guards, MAC, and audit.

    Wires together all trust modules into a single object that agent
    frameworks (LangGraph, CrewAI, etc.) can integrate with.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        agent_id: str,
        constraints: AgentConstraints | dict[str, Any] | None = None,
        manifest: Path | ToolManifest | None = None,
        mode: str = "warn",
        spire_socket: str = "/run/spire/sockets/agent.sock",
    ) -> None:
        self.mode = mode
        self._agent_id = agent_id

        # 1. Resolve identity
        token: str | None = None
        if client is not None:
            token = getattr(getattr(client, "_http", None), "_token", None)
        if token:
            self._identity = AgentIdentity.from_api_key(token)
        else:
            self._identity = AgentIdentity(spire_socket=spire_socket)

        # 2. Resolve constraints: explicit > Console > minimal default
        if isinstance(constraints, AgentConstraints):
            self._constraints = constraints
        elif isinstance(constraints, dict):
            self._constraints = AgentConstraints.model_validate(constraints)
        elif client is not None:
            raw_constraints = client._http.get(f"/agents/{agent_id}/constraints")
            self._constraints = AgentConstraints.model_validate(raw_constraints)
        else:
            # Minimal default: no guards, no tool restrictions, warn mode
            self._constraints = AgentConstraints.model_validate({
                "agent_id": agent_id,
                "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
                "tool_permissions": [],
                "organization": {
                    "required_input_guards": [],
                    "required_output_guards": [],
                    "denied_tools": [],
                },
                "enforcement_mode": mode,
                "updated_at": datetime.now(tz=UTC).isoformat(),
            })

        # 3. Create ToolPolicy — override enforcement_mode to match runtime mode
        constraints_for_policy = self._constraints.model_copy(
            update={"enforcement_mode": mode}
        )
        self._policy = ToolPolicy(constraints_for_policy)

        # 4. Create Dome instance for content guards.
        # Import here to avoid circular import (vijil_dome.__init__ → trust → runtime → vijil_dome).
        from vijil_dome import Dome

        self._dome: Any | None = None
        if _HAS_DOME:
            dome_cfg = self._constraints.dome_config
            if dome_cfg.input_guards or dome_cfg.output_guards:
                try:
                    config: dict[str, Any] = {
                        "input-guards": dome_cfg.input_guards,
                        "output-guards": dome_cfg.output_guards,
                    }
                    config.update(dome_cfg.guards)
                    self._dome = Dome(dome_config=config, enforce=(mode == "enforce"))
                except Exception as exc:
                    logger.warning("Dome initialization failed: %s. Guards disabled.", exc)
            else:
                logger.info("No Dome guards configured; guard passes will be skipped.")
        else:
            logger.info("vijil-dome not installed; guard passes will be skipped.")

        # 5. Load manifest
        self._manifest: ToolManifest | None = None
        if isinstance(manifest, Path):
            self._manifest = ToolManifest.load(manifest)
        elif isinstance(manifest, ToolManifest):
            self._manifest = manifest

        # 6. Create audit emitter
        self._audit = AuditEmitter(agent_id=agent_id)

    # ------------------------------------------------------------------
    # Attestation
    # ------------------------------------------------------------------

    def attest(self) -> AttestationResult:
        """Verify tool identities against the signed manifest."""
        agent_spiffe = self._identity.spiffe_id or self._agent_id

        if self._manifest is None:
            result = AttestationResult(
                agent_identity=agent_spiffe,
                tools=[],
                all_verified=True,
                timestamp=datetime.now(tz=UTC),
            )
            self._audit.emit_attestation(all_verified=True, tool_count=0)
            return result

        if not self._identity.is_attested():
            statuses = [
                ToolAttestationStatus(
                    tool_name=tool.name,
                    expected_identity=tool.identity,
                    verified=False,
                    error="Agent not attested — cannot verify tool identity",
                )
                for tool in self._manifest.tools
            ]
            result = AttestationResult(
                agent_identity=agent_spiffe,
                tools=statuses,
                all_verified=False,
                timestamp=datetime.now(tz=UTC),
            )
            self._audit.emit_attestation(
                all_verified=False, tool_count=len(statuses)
            )
            return result

        statuses = [
            self._verify_tool_identity(tool)
            for tool in self._manifest.tools
        ]
        all_verified = all(s.verified for s in statuses)

        result = AttestationResult(
            agent_identity=agent_spiffe,
            tools=statuses,
            all_verified=all_verified,
            timestamp=datetime.now(tz=UTC),
        )
        self._audit.emit_attestation(
            all_verified=all_verified, tool_count=len(statuses)
        )
        return result

    # ------------------------------------------------------------------
    # Guard passes
    # ------------------------------------------------------------------

    def guard_input(self, message: str) -> GuardResult:
        """Run input through Dome guards, if available."""
        if self._dome is None:
            return GuardResult(
                flagged=False,
                enforced=False,
                score=0.0,
                guarded_response=None,
                exec_time_ms=0.0,
                trace=[],
            )
        scan = self._dome.guard_input(message, agent_id=self._agent_id)
        result = GuardResult.from_scan_result(scan)
        self._audit.emit_guard(
            "input",
            flagged=result.flagged,
            score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    def guard_output(self, response: str) -> GuardResult:
        """Run output through Dome guards, if available."""
        if self._dome is None:
            return GuardResult(
                flagged=False,
                enforced=False,
                score=0.0,
                guarded_response=None,
                exec_time_ms=0.0,
                trace=[],
            )
        scan = self._dome.guard_output(response, agent_id=self._agent_id)
        result = GuardResult.from_scan_result(scan)
        self._audit.emit_guard(
            "output",
            flagged=result.flagged,
            score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    def guard_tool_response(self, tool_name: str, response: str) -> GuardResult:
        """Guard a tool's response through output guards."""
        return self.guard_output(response)

    # ------------------------------------------------------------------
    # MAC enforcement
    # ------------------------------------------------------------------

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> ToolCallResult:
        """Check whether a tool call is permitted by policy."""
        result = self._policy.check(tool_name, args=args)
        self._audit.emit_tool_mac(
            tool_name,
            permitted=result.permitted,
            identity_verified=result.identity_verified,
        )
        return result

    # ------------------------------------------------------------------
    # Tool wrapping
    # ------------------------------------------------------------------

    def wrap_tool(self, tool: Callable[..., Any]) -> Callable[..., Any]:
        """Return a wrapped version of *tool* with MAC and guard enforcement."""
        tool_name = tool.__name__

        @functools.wraps(tool)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            mac_result = self.check_tool_call(tool_name, kwargs)

            if not mac_result.permitted and self.mode == "enforce":
                raise PermissionError(
                    f"Tool '{tool_name}' denied: {mac_result.error}"
                )
            if not mac_result.permitted and self.mode == "warn":
                logger.warning(
                    "Tool '%s' would be denied in enforce mode: %s",
                    tool_name,
                    mac_result.error,
                )

            result = tool(*args, **kwargs)

            if isinstance(result, str) and self._dome is not None:
                guard_result = self.guard_tool_response(tool_name, result)
                if guard_result.flagged and self.mode == "enforce":
                    return guard_result.guarded_response
            return result

        return wrapper

    def wrap_tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Wrap a list of tool callables with MAC and guard enforcement."""
        return [self.wrap_tool(t) for t in tools]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def aguard_input(self, message: str) -> GuardResult:
        """Async input guard — calls Dome's async API directly."""
        if self._dome is None:
            return GuardResult(
                flagged=False, enforced=False, score=0.0,
                guarded_response=None, exec_time_ms=0.0, trace=[],
            )
        scan = await self._dome.async_guard_input(message, agent_id=self._agent_id)
        result = GuardResult.from_scan_result(scan)
        self._audit.emit_guard(
            "input", flagged=result.flagged, score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    async def aguard_output(self, response: str) -> GuardResult:
        """Async output guard — calls Dome's async API directly."""
        if self._dome is None:
            return GuardResult(
                flagged=False, enforced=False, score=0.0,
                guarded_response=None, exec_time_ms=0.0, trace=[],
            )
        scan = await self._dome.async_guard_output(response, agent_id=self._agent_id)
        result = GuardResult.from_scan_result(scan)
        self._audit.emit_guard(
            "output", flagged=result.flagged, score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    def _verify_tool_identity(self, tool: Any) -> ToolAttestationStatus:
        """Verify a single tool's SPIFFE identity via TLS connection.

        Opens a TLS connection to the tool endpoint, extracts the SPIFFE ID
        from the peer certificate's SAN URI field, and compares it to the
        expected identity declared in the tool manifest.

        If the agent is attested (has an X.509 SVID), uses mTLS. Otherwise,
        connects without client cert (server identity only).
        """
        import socket
        import ssl

        endpoint = getattr(tool, "endpoint", "")
        if not endpoint or endpoint == "local":
            # Local tools are not network endpoints — skip verification
            return ToolAttestationStatus(
                tool_name=tool.name,
                expected_identity=tool.identity,
                verified=True,
                error=None,
            )

        # Parse host:port from endpoint (mcp+tls://host:port or https://host:port)
        host, port = self._parse_endpoint(endpoint)
        if not host:
            return ToolAttestationStatus(
                tool_name=tool.name,
                expected_identity=tool.identity,
                verified=False,
                error=f"Cannot parse endpoint: {endpoint}",
            )

        try:
            # Build TLS context — enforce TLS 1.2+ (CodeQL: no insecure versions)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.check_hostname = False  # SPIFFE uses SAN URIs, not hostnames

            # Load agent's mTLS client cert if attested
            if self._identity.is_attested():
                try:
                    agent_ctx = self._identity.mtls_context()
                    ctx = agent_ctx
                    ctx.check_hostname = False
                except RuntimeError:
                    pass  # Fall back to server-only verification

            # Connect and extract peer cert
            raw = socket.create_connection((host, port), timeout=5)
            tls_sock = ctx.wrap_socket(raw, server_hostname=host)

            peer_der = tls_sock.getpeercert(binary_form=True)
            tls_sock.close()

            if peer_der is None:
                return ToolAttestationStatus(
                    tool_name=tool.name,
                    expected_identity=tool.identity,
                    verified=False,
                    error="No peer certificate presented",
                )

            # Extract SPIFFE ID from certificate SAN
            observed_id = self._extract_spiffe_id_from_cert(peer_der)

            if observed_id is None:
                return ToolAttestationStatus(
                    tool_name=tool.name,
                    expected_identity=tool.identity,
                    observed_identity=None,
                    verified=False,
                    error="No SPIFFE ID in peer certificate SANs",
                )

            verified = observed_id == tool.identity
            return ToolAttestationStatus(
                tool_name=tool.name,
                expected_identity=tool.identity,
                observed_identity=observed_id,
                verified=verified,
                error=None if verified else f"Identity mismatch: expected {tool.identity}, got {observed_id}",
            )

        except Exception as exc:
            return ToolAttestationStatus(
                tool_name=tool.name,
                expected_identity=tool.identity,
                verified=False,
                error=f"Verification failed: {exc}",
            )

    async def _verify_tool_identity_async(
        self, tool: Any
    ) -> ToolAttestationStatus:
        """Async variant — runs the blocking TLS connection in a thread."""
        import asyncio
        return await asyncio.to_thread(self._verify_tool_identity, tool)

    async def attest_async(self) -> AttestationResult:
        """Async attestation — runs identity attestation and tool verification."""
        # Ensure identity is attested (async)
        await self._identity.attest_async()

        agent_spiffe = self._identity.spiffe_id or self._agent_id

        if self._manifest is None:
            self._audit.emit_attestation(all_verified=True, tool_count=0)
            return AttestationResult(
                agent_identity=agent_spiffe,
                tools=[],
                all_verified=True,
                timestamp=datetime.now(tz=UTC),
            )

        import asyncio
        statuses = await asyncio.gather(
            *(self._verify_tool_identity_async(tool) for tool in self._manifest.tools)
        )
        all_verified = all(s.verified for s in statuses)
        self._audit.emit_attestation(
            all_verified=all_verified, tool_count=len(statuses)
        )
        return AttestationResult(
            agent_identity=agent_spiffe,
            tools=list(statuses),
            all_verified=all_verified,
            timestamp=datetime.now(tz=UTC),
        )

    @staticmethod
    def _parse_endpoint(endpoint: str) -> tuple[str | None, int]:
        """Parse host and port from a tool endpoint URL."""
        # Strip scheme: mcp+tls://host:port, https://host:port, host:port
        for prefix in ("mcp+tls://", "https://", "http://"):
            if endpoint.startswith(prefix):
                endpoint = endpoint[len(prefix):]
                break

        # Split host:port
        if ":" in endpoint:
            parts = endpoint.rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return None, 0
        return endpoint, 443

    @staticmethod
    def _extract_spiffe_id_from_cert(cert_der: bytes) -> str | None:
        """Extract the SPIFFE ID from a DER-encoded certificate's SAN URIs."""
        try:
            from cryptography import x509
            cert = x509.load_der_x509_certificate(cert_der)
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            uris = san.value.get_values_for_type(x509.UniformResourceIdentifier)
            for uri in uris:
                if uri.startswith("spiffe://"):
                    return uri
        except Exception:
            pass
        return None
