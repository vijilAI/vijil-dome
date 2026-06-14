"""TrustRuntime — orchestrates identity, constraints, guards, MAC, and audit."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import quote

from vijil_dome.controls.models import Control, EvaluationResult
from vijil_dome.trust.attestation import AttestationResult, ToolAttestationStatus
from vijil_dome.trust.audit import (
    AuditEmitter,
    AuditEvent,
    Heartbeat,
    HeartbeatHealth,
)
from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.delta import (
    TrustDelta,
    TrustVector,
    apply_trust_delta,
    extract_trust_deltas,
)
from vijil_dome.trust.guard import EnforcementResult
from vijil_dome.trust.identity import AgentIdentity
from vijil_dome.trust.manifest import ToolManifest
from vijil_dome.trust.policy import ToolCallResult, ToolPolicy
from vijil_dome.trust.signing import BeaconSigner, UnwiredBeaconSigner

logger = logging.getLogger(__name__)

_HAS_DOME = True  # Dome is always available — trust runtime lives inside vijil-dome.

# A benign canned input the reachability probe runs through the guards. It is not
# meant to trip any detector; success (no exception) means the backend responded.
_DETECTOR_PROBE_INPUT = "ping"

# How long a reachability result is reused before the probe runs the real
# detectors again. Defaults at least as long as a typical heartbeat cadence so a
# beacon does not run the backend on every emit. Overridable per instance.
_DEFAULT_DETECTOR_PROBE_TTL_S = 60.0


class _ProbeResult(NamedTuple):
    """A cached detector-reachability result stamped with its monotonic time."""

    at: float
    reachable: bool


# A heartbeat is considered stale (the emitter is not alive) once this many
# intervals have elapsed since the last successful emit — one missed beat is
# tolerated as jitter; two means the loop is dead or stuck.
_HEARTBEAT_STALENESS_FACTOR = 2.0

# How long stop_heartbeat waits for the loop thread to exit before giving up, so
# a hung probe inside the thread cannot make stop block forever.
_HEARTBEAT_JOIN_TIMEOUT_S = 5.0


class TrustRuntime:
    """Core orchestrator composing identity, constraints, guards, MAC, and audit.

    Wires together all trust modules into a single object that agent
    frameworks (LangGraph, CrewAI, etc.) can integrate with. Passing
    ``heartbeat_interval`` auto-starts a daemon thread that emits enforcement
    beacons on that cadence (opt-in; off by default).
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
        audit_sink: Callable[[AuditEvent], None] | None = None,
        heartbeat_interval: float | None = None,
        beacon_signer: BeaconSigner | None = None,
    ) -> None:
        _valid_modes = ("warn", "enforce")
        if mode not in _valid_modes:
            raise ValueError(f"mode must be one of {_valid_modes}, got {mode!r}")
        self.mode = mode
        self._agent_id = agent_id
        self._guards_disabled: bool = False

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
            # A10: an attested agent fetches by its SVID, not the developer-supplied
            # agent_id string — so the developer cannot pick which constraints apply by
            # naming a different agent. The SVID is percent-encoded (it carries ':' and
            # '/'). This branch activates under the pure-mTLS Console client (CON-525):
            # that client carries no API-key token, so identity resolution (step 1) takes
            # the SPIRE path and attests. Transport auth is the mTLS client cert; this code
            # only chooses WHAT to fetch. DEPLOY GATE: CON-525's Console side must bind the
            # served constraints to the mTLS-verified cert SVID (not trust this query param
            # blindly) before this endpoint takes production traffic. An unattested agent
            # keeps today's by-agent_id behavior so existing key-only flows are not bricked.
            svid = self._identity.spiffe_id
            if self._identity.is_attested() and svid:
                raw_constraints = client._http.get(
                    f"/agents/constraints-by-spiffe-id?spiffe_id={quote(svid, safe='')}"
                )
            else:
                # Percent-encode the developer-supplied agent_id too: a '/', '?', or '#'
                # would otherwise change the effective request path (Copilot review).
                raw_constraints = client._http.get(
                    f"/agents/{quote(agent_id, safe='')}/constraints"
                )
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

        # 3. B5: enforce is a FLOOR. A local mode='warn' must not silently downgrade a Console-
        # (or constraint-) mandated 'enforce'. Take the stronger of the two; record any downgrade
        # attempt so it is audited (below) rather than honored. mode is already validated above.
        mandated_mode = self._constraints.enforcement_mode
        effective_mode = "enforce" if "enforce" in (mandated_mode, mode) else "warn"
        self._mode_downgrade_attempted: bool = mandated_mode == "enforce" and mode == "warn"
        self.mode = effective_mode
        constraints_for_policy = self._constraints.model_copy(
            update={"enforcement_mode": effective_mode}
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
                    self._dome = Dome(dome_config=config, enforce=(effective_mode == "enforce"))
                except Exception as exc:
                    # B5: gate on effective_mode, not the raw local mode — else a Dome-init
                    # failure under a mandated-enforce-but-local-warn posture would silently
                    # disable guards (the exact downgrade B5 prevents) instead of raising.
                    if effective_mode == "enforce":
                        raise RuntimeError(
                            f"Dome initialization failed in enforce mode: {exc}"
                        ) from exc
                    logger.warning("Dome initialization failed: %s. Guards disabled.", exc)
                    self._guards_disabled = True
                    self._guards_disabled_error = str(exc)
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
        self._audit = AuditEmitter(agent_id=agent_id, sink=audit_sink)
        if self._mode_downgrade_attempted:
            logger.warning(
                "local mode='warn' cannot downgrade a mandated 'enforce'; using enforce"
            )
            self._audit.emit_mode_downgrade(requested=mode, effective=effective_mode)

        if self._guards_disabled:
            self._audit.emit_guards_disabled(
                error=getattr(self, "_guards_disabled_error", "unknown"),
            )

        if not self._identity.is_attested():
            self._audit.emit_identity_unattested()

        # 7. Trust vector — seeded by a baseline evaluation, not defaulted.
        # ``(control_name, delta)`` pairs arriving before seeding are
        # held here and applied in arrival order when
        # ``seed_trust_vector()`` is called.
        self._trust_vector: TrustVector | None = None
        self._pending_deltas: list[tuple[str, TrustDelta]] = []

        # 8. Detector-reachability probe state. The clock is injectable so the
        # TTL cache can be tested without real time; production uses monotonic.
        self._clock: Callable[[], float] = time.monotonic
        self._detector_probe_ttl_s: float = _DEFAULT_DETECTOR_PROBE_TTL_S
        self._detector_probe_cache: _ProbeResult | None = None

        # 9. Heartbeat scheduler state. The signer defaults to unsigned; the
        # scheduler is opt-in via heartbeat_interval and, when set, auto-starts
        # here as a documented construction side effect (a daemon thread, so it
        # never blocks process exit; one thread per runtime instance).
        self._beacon_signer: BeaconSigner = beacon_signer or UnwiredBeaconSigner()
        self._heartbeat_interval: float | None = heartbeat_interval
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_join_timeout_s: float = _HEARTBEAT_JOIN_TIMEOUT_S
        self._last_emit_at: float | None = None
        if heartbeat_interval is not None:
            self.start_heartbeat(heartbeat_interval)

    # ------------------------------------------------------------------
    # Trust vector
    # ------------------------------------------------------------------

    @property
    def trust_vector(self) -> TrustVector | None:
        """Current measured trust vector, or None if no baseline has been seeded."""
        return self._trust_vector

    def seed_trust_vector(self, vector: TrustVector) -> None:
        """Seed the trust vector from a measured baseline.

        Typically called with the result of a Diamond evaluation. Any
        ``(control_name, delta)`` pairs accumulated before seeding
        (held in ``_pending_deltas``) are applied in arrival order, and
        the queue is cleared.

        Raises ``RuntimeError`` if the trust vector has already been
        seeded — a second seed would erase the runtime adjustments
        accumulated so far, and the "every score point traceable"
        invariant forbids that silently.
        """
        if self._trust_vector is not None:
            raise RuntimeError(
                "TrustRuntime is already seeded; re-seeding would discard "
                "the accumulated trust vector. Create a new TrustRuntime "
                "or expose an explicit reset() if a hard reset is intended."
            )
        seeded = vector
        for control_name, delta in self._pending_deltas:
            before = seeded
            seeded = apply_trust_delta(seeded, delta)
            self._audit.emit_trust_delta(
                control_name=control_name,
                delta=delta,
                before=before,
                after=seeded,
            )
        self._pending_deltas.clear()
        self._trust_vector = seeded

    def apply_evaluation(
        self,
        *,
        controls: list[Control],
        result: EvaluationResult,
    ) -> TrustVector | None:
        """Consume a VijilDome ``EvaluationResult`` for trust-delta side effects.

        Extracts ``vijil.ai/trust-delta`` annotations from triggered
        controls, audits each applied delta, and updates the trust
        vector. Returns the updated vector, or ``None`` if no baseline
        has been seeded yet (the pairs are queued in that case — audit
        emission is deferred to seed-time replay so each logical delta
        produces exactly one audit event).
        """
        pairs = extract_trust_deltas(controls, result)
        if not pairs:
            return self._trust_vector

        if self._trust_vector is None:
            # No baseline yet — queue pairs for application at seed time.
            # Audit emission is deferred to seed_trust_vector's replay so
            # each delta is audited exactly once, with concrete
            # before/after values.
            self._pending_deltas.extend(pairs)
            return None

        # Seeded — apply each delta in turn, auditing the measured before/after.
        current = self._trust_vector
        for control_name, delta in pairs:
            before = current
            current = apply_trust_delta(current, delta)
            self._audit.emit_trust_delta(
                control_name=control_name,
                delta=delta,
                before=before,
                after=current,
            )
        self._trust_vector = current
        return current

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

    def guard_input(self, message: str) -> EnforcementResult:
        """Run input through Dome guards, if available."""
        if self._dome is None:
            return EnforcementResult(
                flagged=False,
                enforced=False,
                score=0.0,
                guarded_response=None,
                exec_time_ms=0.0,
                trace=[],
                guards_disabled=self._guards_disabled,
            )
        scan = self._dome.guard_input(message, agent_id=self._agent_id)
        result = EnforcementResult.from_scan_result(scan)
        self._audit.emit_guard(
            "input",
            flagged=result.flagged,
            score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    def guard_output(self, response: str) -> EnforcementResult:
        """Run output through Dome guards, if available."""
        if self._dome is None:
            return EnforcementResult(
                flagged=False,
                enforced=False,
                score=0.0,
                guarded_response=None,
                exec_time_ms=0.0,
                trace=[],
                guards_disabled=self._guards_disabled,
            )
        scan = self._dome.guard_output(response, agent_id=self._agent_id)
        result = EnforcementResult.from_scan_result(scan)
        self._audit.emit_guard(
            "output",
            flagged=result.flagged,
            score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    def guard_tool_response(self, tool_name: str, response: str) -> EnforcementResult:
        """Guard a tool's response through output guards."""
        return self.guard_output(response)

    # ------------------------------------------------------------------
    # MAC enforcement
    # ------------------------------------------------------------------

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> ToolCallResult:
        """Check whether a tool call is permitted by policy."""
        attested = self._identity.is_attested()
        result = self._policy.check(
            tool_name,
            args=args,
            spiffe_id=self._identity.spiffe_id,
            attested=attested,
        )
        self._audit.emit_tool_mac(
            tool_name,
            permitted=result.permitted,
            identity_verified=result.identity_verified,
            agent_spiffe_id=result.agent_spiffe_id,
        )
        # A3 binding applies to attested callers only. When an SVID-keyed policy is
        # evaluated for an unattested caller, emit a distinct audit event so the residual
        # exposure (an unattested caller hitting an identity-bound policy) is visible — it
        # is not a deny under the default "warn" unattested_tool_policy. Operators close
        # this fully by setting unattested_tool_policy="deny" or awaiting the DOME-166
        # default-flip once attestation is the norm.
        if not attested and self._policy.is_svid_keyed() and result.permitted:
            self._audit.emit_svid_keyed_unattested(
                tool_name,
                policy_subject=self._policy.policy_subject,
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

            if not mac_result.permitted and mac_result.enforced:
                raise PermissionError(
                    f"Tool '{tool_name}' denied: {mac_result.error}"
                )
            if not mac_result.permitted and not mac_result.enforced:
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

    async def aguard_input(self, message: str) -> EnforcementResult:
        """Async input guard — calls Dome's async API directly."""
        if self._dome is None:
            return EnforcementResult(
                flagged=False, enforced=False, score=0.0,
                guarded_response=None, exec_time_ms=0.0, trace=[],
            )
        scan = await self._dome.async_guard_input(message, agent_id=self._agent_id)
        result = EnforcementResult.from_scan_result(scan)
        self._audit.emit_guard(
            "input", flagged=result.flagged, score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    async def aguard_output(self, response: str) -> EnforcementResult:
        """Async output guard — calls Dome's async API directly."""
        if self._dome is None:
            return EnforcementResult(
                flagged=False, enforced=False, score=0.0,
                guarded_response=None, exec_time_ms=0.0, trace=[],
            )
        scan = await self._dome.async_guard_output(response, agent_id=self._agent_id)
        result = EnforcementResult.from_scan_result(scan)
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
                except RuntimeError as exc:
                    if self.mode == "enforce":
                        self._audit.emit_mtls_downgrade(
                            tool.name,
                            error=str(exc),
                        )
                        return ToolAttestationStatus(
                            tool_name=tool.name,
                            expected_identity=tool.identity,
                            verified=False,
                            error=f"mTLS failed in enforce mode: {exc}",
                        )
                    logger.warning("mTLS downgrade for %s: %s", tool.name, exc)
                    self._audit.emit_mtls_downgrade(
                        tool.name,
                        error=str(exc),
                    )

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
        except ImportError:
            logger.debug("cryptography not installed, cannot extract SPIFFE ID")
        except Exception as exc:
            logger.debug("Failed to extract SPIFFE ID from cert: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Enforcement-alive heartbeat (B3)
    # ------------------------------------------------------------------

    def _probe_detector_reachable(self) -> bool:
        """Report whether the detector backend actually responds.

        Distinct from ``guards_constructed``: True only when at least one
        configured guardrail runs a canned input and no probed detector errors —
        proving the backend (local models or a remote inference server) is live,
        not merely that a Dome instance was built. The result is cached for
        ``_detector_probe_ttl_s`` so the real detectors run at most once per
        cadence; a backend that fails after a cached True therefore reads as
        reachable until the entry expires (a bounded, configurable staleness).
        """
        if self._dome is None or self._guards_disabled:
            return False
        now = self._clock()
        cached = self._detector_probe_cache
        if cached is not None and (now - cached.at) < self._detector_probe_ttl_s:
            return cached.reachable
        reachable = self._run_detector_probe(self._dome)
        self._detector_probe_cache = _ProbeResult(at=now, reachable=reachable)
        return reachable

    @staticmethod
    def _run_detector_probe(dome: Any) -> bool:
        """Run the canned input through each configured guardrail; True iff it responds.

        Reachability is read from ``ScanResult.errored_methods``, NOT from a
        raised exception: the guard engine catches a failing detector internally
        and records it in ``errored_methods`` rather than propagating, so a
        ``try/except`` around the call would miss a dead backend entirely. The
        backend is reachable only when at least one guardrail is configured and
        none of the probed detectors errored. The outer ``except`` covers
        unexpected (non-detector) failures, which also mean not-reachable.

        Blocking bound: each detector carries an internal per-call timeout, so a
        guard with N detectors can block up to N×timeout. The heartbeat scheduler
        (DOME-169) runs this off the request path so a slow probe degrades the
        beacon cadence (itself detectable) rather than the agent's own latency.
        """
        probes: list[Callable[[str], Any]] = []
        if dome.input_guardrail is not None:
            probes.append(dome.guard_input)
        if dome.output_guardrail is not None:
            probes.append(dome.guard_output)
        if not probes:
            return False  # Dome built, but no guardrail with detectors to probe.
        try:
            for probe in probes:
                if probe(_DETECTOR_PROBE_INPUT).errored_methods:
                    return False  # a configured detector failed to respond
        except Exception:  # noqa: BLE001 -- any unexpected failure means unreachable
            logger.warning("detector reachability probe failed", exc_info=True)
            return False
        return True

    def emit_heartbeat(self) -> Heartbeat:
        """Emit an enforcement-alive beacon describing the live posture.

        Gathers the configured mode, whether guards were successfully constructed,
        whether the detector backend is reachable, the attestation state, and the
        agent SPIFFE id, then emits an ``enforcement_heartbeat`` audit
        event and returns the ``Heartbeat`` model.

        ``guards_constructed`` is True when a Dome instance was successfully
        built; it proves construction, not that framework callbacks are wired.
        ``detector_reachable`` is a live probe result (see
        ``_probe_detector_reachable``): True only when the backend actually
        responds, False when no guards are configured, guards are disabled, or
        the probe fails. ``attested`` gates the SPIFFE id: ``agent_spiffe_id``
        is only meaningful when ``attested`` is True.

        The beacon is signed by the injected ``BeaconSigner`` (unsigned by
        default); ``start_heartbeat`` drives this method on a cadence. Calling it
        directly is fail-loud — a signer or audit-sink error propagates; the
        scheduler path wraps it (see ``_emit_heartbeat_safely``).
        """
        attested = self._identity.is_attested()
        guards_constructed = self._dome is not None and not self._guards_disabled
        detector_reachable = self._probe_detector_reachable()
        heartbeat = Heartbeat(
            configured_mode=self.mode,
            guards_constructed=guards_constructed,
            detector_reachable=detector_reachable,
            attested=attested,
            agent_spiffe_id=self._identity.spiffe_id if attested else None,
        )
        signature = self._beacon_signer.sign(heartbeat)
        heartbeat = heartbeat.model_copy(update={"signature": signature})
        self._audit.emit_heartbeat(
            configured_mode=heartbeat.configured_mode,
            guards_constructed=heartbeat.guards_constructed,
            detector_reachable=heartbeat.detector_reachable,
            attested=heartbeat.attested,
            agent_spiffe_id=heartbeat.agent_spiffe_id,
            signature=signature,
        )
        return heartbeat

    # ------------------------------------------------------------------
    # Heartbeat scheduler (DOME-169)
    # ------------------------------------------------------------------

    def start_heartbeat(self, interval: float | None = None) -> None:
        """Start the background heartbeat loop if it is not already running.

        Spawns a daemon thread that emits a beacon every ``interval`` seconds.
        Idempotent — a no-op while a loop is already alive. ``interval`` defaults
        to the value given at construction.
        """
        resolved = interval if interval is not None else self._heartbeat_interval
        if resolved is None or resolved <= 0:
            raise ValueError("heartbeat interval must be a positive number of seconds")
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_interval = resolved
        stop = threading.Event()
        thread = threading.Thread(
            target=self._run_heartbeat_loop,
            args=(stop, resolved),
            name=f"dome-heartbeat-{self._agent_id}",
            daemon=True,
        )
        self._heartbeat_stop = stop
        self._heartbeat_thread = thread
        thread.start()

    def stop_heartbeat(self) -> None:
        """Signal the heartbeat loop to stop and wait (bounded) for it to exit.

        Idempotent. The join is bounded by ``_HEARTBEAT_JOIN_TIMEOUT_S`` so a
        probe hung inside the loop cannot make stop block forever.
        """
        stop, thread = self._heartbeat_stop, self._heartbeat_thread
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=self._heartbeat_join_timeout_s)
            if thread.is_alive():
                # The loop did not exit in time — a probe is likely hung. Keep the
                # handle so start_heartbeat refuses to spawn a duplicate; the stop
                # event is set, so the thread exits once the probe returns. Surface
                # it loudly rather than silently orphaning the thread.
                logger.warning(
                    "heartbeat thread %s did not stop within %ss; it will exit when "
                    "its in-flight probe returns",
                    thread.name,
                    self._heartbeat_join_timeout_s,
                )
                return
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def _run_heartbeat_loop(self, stop: threading.Event, interval: float) -> None:
        """Emit a beacon immediately, then once per ``interval`` until stopped.

        ``stop.wait(interval)`` doubles as the sleep and the stop signal, so a
        stop request interrupts the wait at once rather than after a full cycle.
        """
        self._emit_heartbeat_safely()
        while not stop.wait(interval):
            self._emit_heartbeat_safely()

    def _emit_heartbeat_safely(self) -> None:
        """Emit one beacon, advancing ``_last_emit_at`` only on success.

        A failed emit is logged but never propagated, so a single failure cannot
        kill the loop; because the timestamp advances only on success, a run of
        failures grows the staleness that ``heartbeat_health`` surfaces.
        """
        try:
            self.emit_heartbeat()
        except Exception:  # noqa: BLE001 -- a tick failure must not kill the loop
            logger.warning("heartbeat emit failed", exc_info=True)
            return
        self._last_emit_at = self._clock()

    def heartbeat_health(self) -> HeartbeatHealth:
        """Report the heartbeat's in-process liveness (see ``HeartbeatHealth``).

        ``alive`` requires both a running loop AND a recent successful emit, so it
        turns False immediately when the loop is stopped or its thread dies, and
        also when a still-running loop's ticks are failing (staleness) — without
        waiting on the Console staleness sweep.
        """
        thread = self._heartbeat_thread
        running = thread is not None and thread.is_alive()
        last = self._last_emit_at
        age = None if last is None else self._clock() - last
        interval = self._heartbeat_interval
        threshold = (
            interval * _HEARTBEAT_STALENESS_FACTOR if interval is not None else None
        )
        fresh = age is not None and threshold is not None and age <= threshold
        return HeartbeatHealth(running=running, last_emit_age_s=age, alive=running and fresh)
