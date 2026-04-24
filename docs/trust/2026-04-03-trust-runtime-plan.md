# Vijil Trust Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed a trust runtime in the Vijil SDK that wraps LangGraph agents with Dome content guards, tool-level mandatory access control, SPIFFE-based workload identity, and boot-time measured attestation.

**Architecture:** The TrustRuntime composes Dome (content guards), SPIRE (identity), and Console (policy) into a single orchestrator. Framework adapters wrap at two levels: tool callables for MAC + identity, graph entry point for input/output guards. `secure_graph()` replaces `graph.compile()` — one changed line for the developer.

**Tech Stack:** Python 3.12+, Pydantic 2, vijil-dome, py-spiffe, opentelemetry-api, LangGraph, Ed25519 (cryptography lib)

**Design doc:** `docs/plans/2026-04-03-trust-runtime-design.md`

---

## File Structure

### New files (vijil-sdk)

```
src/vijil/trust/
├── __init__.py              # Exports: TrustRuntime, AgentIdentity, ToolManifest, etc.
├── runtime.py               # TrustRuntime orchestrator
├── identity.py              # AgentIdentity — SPIFFE SVID + API key fallback
├── manifest.py              # ToolManifest, ToolEntry — signed build artifact
├── policy.py                # ToolPolicy — MAC enforcement on tool calls
├── guard.py                 # GuardResult, GuardTrace — wraps Dome ScanResult
├── attestation.py           # AttestationResult — boot-time tool verification
├── constraints.py           # AgentConstraints, DomeGuardConfig, OrganizationConstraints
└── audit.py                 # AuditEmitter — OTel span emission

src/vijil/adapters/
├── __init__.py
└── langgraph.py             # secure_graph(), SecureGraph

src/vijil_cli/porcelain/
└── manifest_cmd.py          # vijil manifest sign|verify

tests/unit/trust/
├── __init__.py
├── test_manifest.py         # Manifest signing, loading, verification
├── test_policy.py           # Tool MAC — permit/deny decisions
├── test_guard.py            # GuardResult wrapping ScanResult
├── test_attestation.py      # Boot attestation — verify/reject tools
├── test_identity.py         # SPIFFE identity + fallback
├── test_constraints.py      # AgentConstraints parsing
├── test_runtime.py          # TrustRuntime orchestration
└── test_audit.py            # OTel span emission

tests/unit/adapters/
├── __init__.py
└── test_langgraph.py        # secure_graph, invoke/stream wrapping
```

### Modified files

```
pyproject.toml               # Add [trust] extra: vijil-dome, py-spiffe, cryptography, opentelemetry-api
src/vijil/__init__.py        # Re-export TrustRuntime
src/vijil_cli/main.py        # Register manifest command
```

---

## Dependency Graph

```
Train 1 (Domain models — sequential, foundation):
  1.1 Manifest models → 1.2 Guard models → 1.3 Constraints models → 1.4 Tool policy

Train 2 (Adapters — parallel, after Train 1):
  2.1 Identity (SPIFFE) | 2.2 Constraints fetch | 2.3 Dome adapter | 2.4 Audit emitter

Train 3 (Application — sequential, after Train 2):
  3.1 Attestation logic → 3.2 TrustRuntime orchestrator

Train 4 (Framework + CLI — parallel, after Train 3):
  4.1 LangGraph adapter | 4.2 Manifest CLI command

Train 5 (Package + integration — after Train 4):
  5.1 pyproject.toml extras → 5.2 Integration tests
```

---

## Train 1: Domain Models

### Task 1.1: Tool manifest and entry models

**Files:**
- Create: `src/vijil/trust/__init__.py`
- Create: `src/vijil/trust/manifest.py`
- Test: `tests/unit/trust/__init__.py`
- Test: `tests/unit/trust/test_manifest.py`

- [ ] **Step 1: Write failing tests for ToolEntry and ToolManifest**

```python
# tests/unit/trust/test_manifest.py
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from vijil.trust.manifest import ToolEntry, ToolManifest


class TestToolEntry:
    def test_create_entry(self):
        entry = ToolEntry(
            name="book_flight",
            identity="spiffe://vijil.ai/tools/flights/v2",
            endpoint="mcp+tls://flights.internal:8443",
            version="2.1.0",
        )
        assert entry.name == "book_flight"
        assert entry.identity.startswith("spiffe://")

    def test_entry_without_version(self):
        entry = ToolEntry(
            name="search",
            identity="spiffe://vijil.ai/tools/search/v1",
            endpoint="mcp+tls://search.internal:8443",
        )
        assert entry.version is None


class TestToolManifest:
    def test_create_manifest(self):
        manifest = ToolManifest(
            agent_id="spiffe://vijil.ai/agent/travel-agent",
            tools=[
                ToolEntry(
                    name="book_flight",
                    identity="spiffe://vijil.ai/tools/flights/v2",
                    endpoint="mcp+tls://flights.internal:8443",
                ),
            ],
            compiled_at=datetime.now(timezone.utc),
            signature="unsigned",
        )
        assert len(manifest.tools) == 1
        assert manifest.manifest_version == 1

    def test_canonical_json_is_deterministic(self):
        """Two manifests with same data produce same canonical JSON."""
        kwargs = dict(
            agent_id="spiffe://vijil.ai/agent/test",
            tools=[
                ToolEntry(
                    name="a",
                    identity="spiffe://vijil.ai/tools/a/v1",
                    endpoint="mcp+tls://a:8443",
                ),
            ],
            compiled_at=datetime(2026, 4, 3, tzinfo=timezone.utc),
            signature="",
        )
        m1 = ToolManifest(**kwargs)
        m2 = ToolManifest(**kwargs)
        assert m1.canonical_json() == m2.canonical_json()

    def test_sign_and_verify_roundtrip(self):
        """Sign a manifest, then verify the signature."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        manifest = ToolManifest(
            agent_id="spiffe://vijil.ai/agent/test",
            tools=[],
            compiled_at=datetime.now(timezone.utc),
            signature="",
        )
        signed = manifest.sign(private_key)
        assert signed.signature != ""
        assert signed.verify_signature(public_key)

    def test_tampered_manifest_fails_verification(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        manifest = ToolManifest(
            agent_id="spiffe://vijil.ai/agent/test",
            tools=[],
            compiled_at=datetime.now(timezone.utc),
            signature="",
        )
        signed = manifest.sign(private_key)
        # Tamper with agent_id
        tampered = signed.model_copy(update={"agent_id": "spiffe://evil.ai/agent/bad"})
        assert not tampered.verify_signature(public_key)

    def test_load_from_file(self, tmp_path: Path):
        manifest = ToolManifest(
            agent_id="spiffe://vijil.ai/agent/test",
            tools=[],
            compiled_at=datetime(2026, 4, 3, tzinfo=timezone.utc),
            signature="test-sig",
        )
        path = tmp_path / "manifest.json"
        path.write_text(manifest.model_dump_json(indent=2))
        loaded = ToolManifest.load(path)
        assert loaded.agent_id == manifest.agent_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/trust/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vijil.trust'`

- [ ] **Step 3: Implement ToolEntry, ToolManifest**

```python
# src/vijil/trust/manifest.py
"""Signed tool manifest — build artifact binding tools to SPIFFE identities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import Field

from vijil.models.base import VijilModel


class ToolEntry(VijilModel):
    """A single tool binding in the manifest."""

    name: str
    identity: str
    endpoint: str
    version: str | None = None


class ToolManifest(VijilModel):
    """Signed build artifact declaring authorized tools."""

    manifest_version: int = 1
    agent_id: str
    tools: list[ToolEntry]
    compiled_at: datetime
    signature: str = ""

    def canonical_json(self) -> str:
        """Deterministic JSON for signing (excludes signature field)."""
        data = self.model_dump(exclude={"signature"})
        return json.dumps(data, sort_keys=True, default=str)

    def sign(self, private_key: object) -> ToolManifest:
        """Sign the manifest with an Ed25519 private key. Returns new instance."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        assert isinstance(private_key, Ed25519PrivateKey)
        payload = self.canonical_json().encode()
        sig = private_key.sign(payload)
        return self.model_copy(update={"signature": sig.hex()})

    def verify_signature(self, public_key: object) -> bool:
        """Verify the manifest signature against an Ed25519 public key."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature

        assert isinstance(public_key, Ed25519PublicKey)
        payload = self.canonical_json().encode()
        try:
            public_key.verify(bytes.fromhex(self.signature), payload)
            return True
        except InvalidSignature:
            return False

    @classmethod
    def load(cls, path: Path) -> ToolManifest:
        """Load a manifest from a JSON file."""
        return cls.model_validate_json(path.read_text())
```

```python
# src/vijil/trust/__init__.py
"""Vijil Trust Runtime — measured boot and continuous attestation for AI agents."""

from vijil.trust.manifest import ToolEntry, ToolManifest

__all__ = ["ToolEntry", "ToolManifest"]
```

Also create `tests/unit/trust/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/trust/test_manifest.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/vijil/trust/ tests/unit/trust/
git commit -m "feat(trust): add ToolManifest with Ed25519 signing and verification"
```

---

### Task 1.2: Guard result models (wraps Dome ScanResult)

**Files:**
- Create: `src/vijil/trust/guard.py`
- Test: `tests/unit/trust/test_guard.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/trust/test_guard.py
from vijil.trust.guard import DetectorTrace, GuardTrace, GuardResult


class TestGuardResult:
    def test_unflagged_result(self):
        result = GuardResult(
            flagged=False,
            enforced=False,
            score=0.05,
            guarded_response=None,
            exec_time_ms=12.3,
            trace=[],
        )
        assert not result.flagged
        assert result.score < 0.5

    def test_flagged_enforced_result(self):
        result = GuardResult(
            flagged=True,
            enforced=True,
            score=0.95,
            guarded_response="I cannot help with that.",
            exec_time_ms=45.0,
            trace=[
                GuardTrace(
                    guard_name="security",
                    detectors=[
                        DetectorTrace(
                            detector_name="prompt-injection-deberta",
                            hit=True,
                            score=0.95,
                            exec_time_ms=42.0,
                        )
                    ],
                )
            ],
        )
        assert result.flagged
        assert result.enforced
        assert result.guarded_response is not None
        assert len(result.trace) == 1
        assert result.trace[0].detectors[0].hit

    def test_from_scan_result_unflagged(self):
        """GuardResult.from_scan_result wraps Dome's ScanResult."""
        # Mock a ScanResult-like object
        class MockScanResult:
            flagged = False
            enforced = False
            detection_score = 0.1
            response_string = ""
            exec_time = 0.015
            trace = {}

        result = GuardResult.from_scan_result(MockScanResult())
        assert not result.flagged
        assert result.score == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/trust/test_guard.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement guard models**

```python
# src/vijil/trust/guard.py
"""Guard result models — typed wrapper around Dome's ScanResult."""

from __future__ import annotations

from typing import Any

from vijil.models.base import VijilModel


class DetectorTrace(VijilModel):
    """Trace from a single Dome detector."""

    detector_name: str
    hit: bool
    score: float
    exec_time_ms: float


class GuardTrace(VijilModel):
    """Trace from a single Dome guard (contains multiple detectors)."""

    guard_name: str
    detectors: list[DetectorTrace]


class GuardResult(VijilModel):
    """Result of a Dome guard check. Wraps Dome's ScanResult."""

    flagged: bool
    enforced: bool
    score: float
    guarded_response: str | None
    exec_time_ms: float
    trace: list[GuardTrace]

    @classmethod
    def from_scan_result(cls, scan: Any) -> GuardResult:
        """Convert a Dome ScanResult to a GuardResult."""
        traces: list[GuardTrace] = []
        if hasattr(scan, "trace") and isinstance(scan.trace, dict):
            for guard_name, guard_data in scan.trace.items():
                detectors: list[DetectorTrace] = []
                if isinstance(guard_data, dict):
                    for det_name, det_data in guard_data.items():
                        hit = getattr(det_data, "hit", False)
                        result_data = getattr(det_data, "result", {})
                        det_score = (
                            result_data.get("score", 1.0 if hit else 0.0)
                            if isinstance(result_data, dict)
                            else (1.0 if hit else 0.0)
                        )
                        detectors.append(
                            DetectorTrace(
                                detector_name=det_name,
                                hit=hit,
                                score=det_score,
                                exec_time_ms=getattr(det_data, "exec_time", 0.0) * 1000,
                            )
                        )
                traces.append(GuardTrace(guard_name=guard_name, detectors=detectors))

        return cls(
            flagged=scan.flagged,
            enforced=scan.enforced,
            score=scan.detection_score,
            guarded_response=scan.response_string if scan.flagged else None,
            exec_time_ms=getattr(scan, "exec_time", 0.0) * 1000,
            trace=traces,
        )
```

- [ ] **Step 4: Run tests, update `__init__.py` exports**

Run: `poetry run pytest tests/unit/trust/test_guard.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/vijil/trust/guard.py tests/unit/trust/test_guard.py src/vijil/trust/__init__.py
git commit -m "feat(trust): add GuardResult with ScanResult adapter"
```

---

### Task 1.3: Constraints models + ToolCallResult + AttestationResult

**Files:**
- Create: `src/vijil/trust/constraints.py`
- Create: `src/vijil/trust/attestation.py`
- Test: `tests/unit/trust/test_constraints.py`
- Test: `tests/unit/trust/test_attestation.py`

- [ ] **Step 1: Write failing tests for constraints**

```python
# tests/unit/trust/test_constraints.py
from vijil.trust.constraints import (
    AgentConstraints,
    DomeGuardConfig,
    OrganizationConstraints,
    ToolPermission,
)


class TestAgentConstraints:
    def test_parse_constraints(self):
        raw = {
            "agent_id": "travel-agent",
            "dome_config": {
                "input_guards": ["security"],
                "output_guards": ["moderation", "privacy"],
                "guards": {
                    "security": {"type": "security", "methods": ["encoding-heuristics"]},
                },
            },
            "tool_permissions": [
                {
                    "name": "book_flight",
                    "identity": "spiffe://vijil.ai/tools/flights/v2",
                    "endpoint": "mcp+tls://flights:8443",
                },
            ],
            "organization": {
                "required_input_guards": ["security"],
                "required_output_guards": ["privacy"],
                "denied_tools": ["charge_credit_card"],
            },
            "enforcement_mode": "warn",
            "updated_at": "2026-04-03T00:00:00Z",
        }
        constraints = AgentConstraints.model_validate(raw)
        assert constraints.enforcement_mode == "warn"
        assert len(constraints.tool_permissions) == 1
        assert "charge_credit_card" in constraints.organization.denied_tools

    def test_tool_permitted(self):
        constraints = AgentConstraints.model_validate({
            "agent_id": "test",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [
                {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/flights/v2", "endpoint": "x"},
            ],
            "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": ["bad_tool"]},
            "enforcement_mode": "enforce",
            "updated_at": "2026-04-03T00:00:00Z",
        })
        names = {tp.name for tp in constraints.tool_permissions}
        assert "book_flight" in names
        assert "bad_tool" in constraints.organization.denied_tools
```

- [ ] **Step 2: Write failing tests for attestation**

```python
# tests/unit/trust/test_attestation.py
from datetime import datetime, timezone
from vijil.trust.attestation import AttestationResult, ToolAttestationStatus


class TestAttestationResult:
    def test_all_verified(self):
        result = AttestationResult(
            agent_identity="spiffe://vijil.ai/agent/test",
            tools=[
                ToolAttestationStatus(
                    tool_name="book_flight",
                    expected_identity="spiffe://vijil.ai/tools/flights/v2",
                    observed_identity="spiffe://vijil.ai/tools/flights/v2",
                    verified=True,
                ),
            ],
            all_verified=True,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.all_verified

    def test_mismatch_not_verified(self):
        result = AttestationResult(
            agent_identity="spiffe://vijil.ai/agent/test",
            tools=[
                ToolAttestationStatus(
                    tool_name="book_flight",
                    expected_identity="spiffe://vijil.ai/tools/flights/v2",
                    observed_identity="spiffe://evil.ai/tools/fake",
                    verified=False,
                    error="SPIFFE ID mismatch",
                ),
            ],
            all_verified=False,
            timestamp=datetime.now(timezone.utc),
        )
        assert not result.all_verified
        assert result.tools[0].error is not None

    def test_unreachable_tool(self):
        status = ToolAttestationStatus(
            tool_name="offline_tool",
            expected_identity="spiffe://vijil.ai/tools/offline/v1",
            observed_identity=None,
            verified=False,
            error="Tool endpoint unreachable",
        )
        assert status.observed_identity is None
```

- [ ] **Step 3: Implement models**

```python
# src/vijil/trust/constraints.py
"""Agent constraints — policy fetched from Console."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from vijil.models.base import VijilModel


class ToolPermission(VijilModel):
    name: str
    identity: str
    endpoint: str
    allowed_actions: list[str] | None = None


class DomeGuardConfig(VijilModel):
    input_guards: list[str]
    output_guards: list[str]
    guards: dict[str, dict[str, object]]


class OrganizationConstraints(VijilModel):
    required_input_guards: list[str]
    required_output_guards: list[str]
    denied_tools: list[str]
    max_model_tier: str | None = None


class AgentConstraints(VijilModel):
    agent_id: str
    dome_config: DomeGuardConfig
    tool_permissions: list[ToolPermission]
    organization: OrganizationConstraints
    enforcement_mode: Literal["warn", "enforce"]
    updated_at: datetime
```

```python
# src/vijil/trust/attestation.py
"""Attestation results — boot-time tool identity verification."""

from __future__ import annotations

from datetime import datetime

from vijil.models.base import VijilModel


class ToolAttestationStatus(VijilModel):
    tool_name: str
    expected_identity: str
    observed_identity: str | None = None
    verified: bool
    error: str | None = None


class AttestationResult(VijilModel):
    agent_identity: str
    tools: list[ToolAttestationStatus]
    all_verified: bool
    timestamp: datetime
```

- [ ] **Step 4: Run all trust tests**

Run: `poetry run pytest tests/unit/trust/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/vijil/trust/constraints.py src/vijil/trust/attestation.py \
  tests/unit/trust/test_constraints.py tests/unit/trust/test_attestation.py \
  src/vijil/trust/__init__.py
git commit -m "feat(trust): add AgentConstraints and AttestationResult models"
```

---

### Task 1.4: Tool policy — MAC enforcement logic

**Files:**
- Create: `src/vijil/trust/policy.py`
- Test: `tests/unit/trust/test_policy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/trust/test_policy.py
import pytest
from vijil.trust.policy import ToolPolicy, ToolCallResult
from vijil.trust.constraints import AgentConstraints, ToolPermission, OrganizationConstraints, DomeGuardConfig


def _make_constraints(**overrides):
    defaults = {
        "agent_id": "test",
        "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
        "tool_permissions": [
            {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/flights/v2", "endpoint": "x"},
            {"name": "search_hotels", "identity": "spiffe://vijil.ai/tools/hotels/v1", "endpoint": "x"},
        ],
        "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": ["charge_card"]},
        "enforcement_mode": "enforce",
        "updated_at": "2026-04-03T00:00:00Z",
    }
    defaults.update(overrides)
    return AgentConstraints.model_validate(defaults)


class TestToolPolicy:
    def test_permitted_tool(self):
        policy = ToolPolicy(_make_constraints())
        result = policy.check("book_flight")
        assert result.permitted
        assert result.policy_permitted

    def test_denied_tool_not_in_permissions(self):
        policy = ToolPolicy(_make_constraints())
        result = policy.check("delete_all_records")
        assert not result.permitted
        assert not result.policy_permitted

    def test_org_denied_tool(self):
        policy = ToolPolicy(_make_constraints())
        result = policy.check("charge_card")
        assert not result.permitted
        assert "denied by organization" in (result.error or "").lower()

    def test_warn_mode_still_reports_denied(self):
        policy = ToolPolicy(_make_constraints(enforcement_mode="warn"))
        result = policy.check("charge_card")
        assert not result.permitted
        assert not result.enforced  # warn mode

    def test_enforce_mode_sets_enforced(self):
        policy = ToolPolicy(_make_constraints(enforcement_mode="enforce"))
        result = policy.check("charge_card")
        assert not result.permitted
        assert result.enforced  # enforce mode

    def test_get_permission_returns_entry(self):
        policy = ToolPolicy(_make_constraints())
        perm = policy.get_permission("book_flight")
        assert perm is not None
        assert perm.identity == "spiffe://vijil.ai/tools/flights/v2"

    def test_get_permission_returns_none_for_unknown(self):
        policy = ToolPolicy(_make_constraints())
        assert policy.get_permission("unknown_tool") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/trust/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ToolPolicy**

```python
# src/vijil/trust/policy.py
"""Tool policy — mandatory access control on tool invocations."""

from __future__ import annotations

from vijil.models.base import VijilModel
from vijil.trust.constraints import AgentConstraints, ToolPermission


class ToolCallResult(VijilModel):
    """Result of a tool MAC + identity check."""

    permitted: bool
    tool_name: str
    identity_verified: bool = False
    policy_permitted: bool = False
    enforced: bool = False
    error: str | None = None


class ToolPolicy:
    """Enforces mandatory access control on tool calls."""

    def __init__(self, constraints: AgentConstraints) -> None:
        self._permissions = {tp.name: tp for tp in constraints.tool_permissions}
        self._denied = set(constraints.organization.denied_tools)
        self._enforce = constraints.enforcement_mode == "enforce"

    def check(self, tool_name: str) -> ToolCallResult:
        """Check whether a tool call is permitted."""
        if tool_name in self._denied:
            return ToolCallResult(
                permitted=False,
                tool_name=tool_name,
                policy_permitted=False,
                enforced=self._enforce,
                error=f"Tool '{tool_name}' denied by organization constraints",
            )

        if tool_name not in self._permissions:
            return ToolCallResult(
                permitted=False,
                tool_name=tool_name,
                policy_permitted=False,
                enforced=self._enforce,
                error=f"Tool '{tool_name}' not in agent permissions",
            )

        return ToolCallResult(
            permitted=True,
            tool_name=tool_name,
            policy_permitted=True,
            enforced=False,
        )

    def get_permission(self, tool_name: str) -> ToolPermission | None:
        """Get the permission entry for a tool (for identity verification)."""
        return self._permissions.get(tool_name)
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/unit/trust/test_policy.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/vijil/trust/policy.py tests/unit/trust/test_policy.py src/vijil/trust/__init__.py
git commit -m "feat(trust): add ToolPolicy with MAC enforcement"
```

---

## Train 2: Adapters

### Task 2.1: Agent identity — SPIFFE + API key fallback

**Files:**
- Create: `src/vijil/trust/identity.py`
- Test: `tests/unit/trust/test_identity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/trust/test_identity.py
import pytest
from vijil.trust.identity import AgentIdentity


class TestAgentIdentity:
    def test_from_api_key(self):
        identity = AgentIdentity.from_api_key("vj-test-key")
        assert not identity.is_attested()
        assert identity.spiffe_id is None
        assert identity.api_key == "vj-test-key"

    def test_spire_unavailable_returns_unattested(self):
        identity = AgentIdentity(spire_socket="/nonexistent/socket")
        assert not identity.is_attested()
        assert identity.spiffe_id is None

    def test_mtls_context_without_spire_raises(self):
        identity = AgentIdentity(spire_socket="/nonexistent/socket")
        with pytest.raises(RuntimeError, match="SPIRE"):
            identity.mtls_context()

    def test_auth_header_with_api_key(self):
        identity = AgentIdentity.from_api_key("vj-test-key")
        assert identity.auth_header() == {"authorization": "Bearer vj-test-key"}
```

- [ ] **Step 2: Implement AgentIdentity**

```python
# src/vijil/trust/identity.py
"""Agent identity — SPIFFE-based workload identity with API key fallback."""

from __future__ import annotations

import logging
import ssl
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentIdentity:
    """SPIFFE-based workload identity with API key fallback.

    If SPIRE socket is available, fetches X.509-SVID for mTLS.
    Otherwise, falls back to API key authentication.
    """

    def __init__(self, spire_socket: str = "/run/spire/sockets/agent.sock") -> None:
        self._spire_socket = spire_socket
        self._spiffe_id: str | None = None
        self._api_key: str | None = None
        self._attested = False

        if Path(spire_socket).exists():
            self._init_spire()
        else:
            logger.info("SPIRE socket not found at %s — using API key fallback", spire_socket)

    def _init_spire(self) -> None:
        """Attempt to fetch SVID from SPIRE Workload API."""
        try:
            from pyspiffe.workloadapi import WorkloadApiClient

            client = WorkloadApiClient(f"unix://{self._spire_socket}")
            svid = client.fetch_x509_svid()
            self._spiffe_id = str(svid.spiffe_id)
            self._svid = svid
            self._attested = True
            logger.info("SPIFFE identity: %s", self._spiffe_id)
        except Exception as e:
            logger.warning("Failed to fetch SVID from SPIRE: %s", e)

    @classmethod
    def from_api_key(cls, api_key: str) -> AgentIdentity:
        """Create identity from static API key (development fallback)."""
        instance = cls.__new__(cls)
        instance._spire_socket = ""
        instance._spiffe_id = None
        instance._api_key = api_key
        instance._attested = False
        return instance

    @property
    def spiffe_id(self) -> str | None:
        return self._spiffe_id

    @property
    def api_key(self) -> str | None:
        return self._api_key

    def is_attested(self) -> bool:
        return self._attested

    def mtls_context(self) -> ssl.SSLContext:
        """Create SSL context for outbound mTLS connections."""
        if not self._attested:
            raise RuntimeError(
                "Cannot create mTLS context without SPIRE attestation. "
                "Ensure SPIRE Agent is running and the socket is accessible."
            )
        # py-spiffe provides the SSL context from the SVID
        return self._svid.ssl_context()  # type: ignore[attr-defined]

    def auth_header(self) -> dict[str, str]:
        """Return auth header for HTTP calls (API key fallback)."""
        if self._api_key:
            return {"authorization": f"Bearer {self._api_key}"}
        if self._attested:
            return {}  # mTLS handles auth
        return {}
```

- [ ] **Step 3: Run tests**

Run: `poetry run pytest tests/unit/trust/test_identity.py -v`
Expected: All 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/vijil/trust/identity.py tests/unit/trust/test_identity.py
git commit -m "feat(trust): add AgentIdentity with SPIFFE + API key fallback"
```

---

### Task 2.2: Audit emitter — OTel spans

**Files:**
- Create: `src/vijil/trust/audit.py`
- Test: `tests/unit/trust/test_audit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/trust/test_audit.py
from vijil.trust.audit import AuditEmitter, AuditEvent


class TestAuditEmitter:
    def test_emit_guard_event(self):
        events: list[AuditEvent] = []
        emitter = AuditEmitter(agent_id="test-agent", sink=events.append)
        emitter.emit_guard("input", flagged=False, score=0.1, exec_time_ms=12.0)
        assert len(events) == 1
        assert events[0].event_type == "guard"
        assert events[0].agent_id == "test-agent"

    def test_emit_tool_mac_event(self):
        events: list[AuditEvent] = []
        emitter = AuditEmitter(agent_id="test-agent", sink=events.append)
        emitter.emit_tool_mac("book_flight", permitted=True, identity_verified=True)
        assert events[0].event_type == "tool_mac"

    def test_emit_attestation_event(self):
        events: list[AuditEvent] = []
        emitter = AuditEmitter(agent_id="test-agent", sink=events.append)
        emitter.emit_attestation(all_verified=True, tool_count=3)
        assert events[0].event_type == "attestation"
```

- [ ] **Step 2: Implement AuditEmitter**

A lightweight emitter that accepts a sink callable. Default sink writes OTel spans when the library is available, falls back to logging.

```python
# src/vijil/trust/audit.py
"""Audit emitter — structured events for trust runtime decisions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from vijil.models.base import VijilModel

logger = logging.getLogger(__name__)


class AuditEvent(VijilModel):
    """A single audit event from the trust runtime."""

    event_type: str  # "guard", "tool_mac", "attestation"
    agent_id: str
    timestamp: datetime
    attributes: dict[str, Any]


class AuditEmitter:
    """Emits structured audit events. Pluggable sink (OTel, logging, list)."""

    def __init__(
        self,
        agent_id: str,
        sink: Callable[[AuditEvent], None] | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._sink = sink or self._default_sink

    def _default_sink(self, event: AuditEvent) -> None:
        logger.info("audit: %s %s", event.event_type, event.attributes)

    def _emit(self, event_type: str, **attrs: Any) -> None:
        event = AuditEvent(
            event_type=event_type,
            agent_id=self._agent_id,
            timestamp=datetime.now(timezone.utc),
            attributes=attrs,
        )
        self._sink(event)

    def emit_guard(
        self, direction: str, *, flagged: bool, score: float, exec_time_ms: float
    ) -> None:
        self._emit("guard", direction=direction, flagged=flagged, score=score, exec_time_ms=exec_time_ms)

    def emit_tool_mac(
        self, tool_name: str, *, permitted: bool, identity_verified: bool
    ) -> None:
        self._emit("tool_mac", tool_name=tool_name, permitted=permitted, identity_verified=identity_verified)

    def emit_attestation(self, *, all_verified: bool, tool_count: int) -> None:
        self._emit("attestation", all_verified=all_verified, tool_count=tool_count)
```

- [ ] **Step 3: Run tests**

Run: `poetry run pytest tests/unit/trust/test_audit.py -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/vijil/trust/audit.py tests/unit/trust/test_audit.py
git commit -m "feat(trust): add AuditEmitter with pluggable sink"
```

---

## Train 3: Application

### Task 3.1: TrustRuntime orchestrator

**Files:**
- Create: `src/vijil/trust/runtime.py`
- Test: `tests/unit/trust/test_runtime.py`

This is the core orchestrator. It composes identity, Dome guards, tool policy, attestation, and audit into a single object. Tests mock all external dependencies.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/trust/test_runtime.py
from unittest.mock import MagicMock, patch
import pytest
from vijil.trust.runtime import TrustRuntime
from vijil.trust.guard import GuardResult
from vijil.trust.policy import ToolCallResult


class TestTrustRuntime:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        # Mock the constraints fetch
        client._http.get.return_value = {
            "agent_id": "test-agent",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [
                {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/flights/v2", "endpoint": "x"},
            ],
            "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
            "enforcement_mode": "warn",
            "updated_at": "2026-04-03T00:00:00Z",
        }
        return client

    def test_construction_fetches_constraints(self, mock_client):
        runtime = TrustRuntime(
            client=mock_client,
            agent_id="test-agent",
            mode="warn",
        )
        assert runtime.mode == "warn"
        assert runtime._policy is not None

    def test_check_tool_call_permitted(self, mock_client):
        runtime = TrustRuntime(client=mock_client, agent_id="test-agent", mode="warn")
        result = runtime.check_tool_call("book_flight", {})
        assert result.permitted

    def test_check_tool_call_denied(self, mock_client):
        runtime = TrustRuntime(client=mock_client, agent_id="test-agent", mode="enforce")
        result = runtime.check_tool_call("delete_records", {})
        assert not result.permitted

    def test_guard_input_delegates_to_dome(self, mock_client):
        runtime = TrustRuntime(client=mock_client, agent_id="test-agent", mode="warn")
        # Mock Dome
        mock_scan = MagicMock()
        mock_scan.flagged = False
        mock_scan.enforced = False
        mock_scan.detection_score = 0.05
        mock_scan.response_string = ""
        mock_scan.exec_time = 0.01
        mock_scan.trace = {}
        runtime._dome = MagicMock()
        runtime._dome.guard_input.return_value = mock_scan

        result = runtime.guard_input("hello")
        assert isinstance(result, GuardResult)
        assert not result.flagged

    def test_warn_mode_does_not_enforce(self, mock_client):
        runtime = TrustRuntime(client=mock_client, agent_id="test-agent", mode="warn")
        result = runtime.check_tool_call("unauthorized", {})
        assert not result.permitted
        assert not result.enforced

    def test_enforce_mode_enforces(self, mock_client):
        runtime = TrustRuntime(client=mock_client, agent_id="test-agent", mode="enforce")
        result = runtime.check_tool_call("unauthorized", {})
        assert not result.permitted
        assert result.enforced

    def test_wrap_tool_returns_callable(self, mock_client):
        runtime = TrustRuntime(client=mock_client, agent_id="test-agent", mode="warn")

        def original_tool(query: str) -> str:
            return f"result for {query}"

        original_tool.__name__ = "book_flight"
        wrapped = runtime.wrap_tool(original_tool)
        assert callable(wrapped)

    def test_wrapped_tool_calls_original(self, mock_client):
        runtime = TrustRuntime(client=mock_client, agent_id="test-agent", mode="warn")
        # Mock Dome to not flag
        mock_scan = MagicMock()
        mock_scan.flagged = False
        mock_scan.enforced = False
        mock_scan.detection_score = 0.0
        mock_scan.response_string = ""
        mock_scan.exec_time = 0.0
        mock_scan.trace = {}
        runtime._dome = MagicMock()
        runtime._dome.guard_output.return_value = mock_scan

        def book_flight(dest: str) -> str:
            return f"Booked flight to {dest}"

        wrapped = runtime.wrap_tool(book_flight)
        result = wrapped(dest="Paris")
        assert result == "Booked flight to Paris"
```

- [ ] **Step 2: Implement TrustRuntime**

```python
# src/vijil/trust/runtime.py
"""TrustRuntime — core orchestrator for the trust layer."""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable, Literal

from vijil.trust.attestation import AttestationResult
from vijil.trust.audit import AuditEmitter
from vijil.trust.constraints import AgentConstraints
from vijil.trust.guard import GuardResult
from vijil.trust.identity import AgentIdentity
from vijil.trust.manifest import ToolManifest
from vijil.trust.policy import ToolCallResult, ToolPolicy

logger = logging.getLogger(__name__)


class TrustRuntime:
    """Core orchestrator — composes identity, guards, tool policy, and audit."""

    def __init__(
        self,
        *,
        client: Any,  # Vijil client (Any to avoid circular import)
        agent_id: str,
        manifest: ToolManifest | Path | None = None,
        mode: Literal["warn", "enforce"] = "warn",
        spire_socket: str = "/run/spire/sockets/agent.sock",
    ) -> None:
        self.mode = mode
        self._agent_id = agent_id

        # 1. Resolve identity
        if hasattr(client, "_http") and hasattr(client._http, "_token"):
            self._identity = AgentIdentity.from_api_key(client._http._token)
        else:
            self._identity = AgentIdentity(spire_socket=spire_socket)

        # 2. Fetch constraints from Console
        raw = client._http.get(f"/agents/{agent_id}/constraints")
        self._constraints = AgentConstraints.model_validate(raw)

        # 3. Create tool policy
        self._policy = ToolPolicy(self._constraints)

        # 4. Create Dome instance (try vijil_dome, fall back to None)
        self._dome = self._init_dome()

        # 5. Load manifest
        self._manifest = self._load_manifest(manifest)

        # 6. Create audit emitter
        self._audit = AuditEmitter(agent_id=agent_id)

    def _init_dome(self) -> Any:
        """Initialize Dome from constraints config."""
        try:
            from vijil_dome import Dome

            dome_config = self._constraints.dome_config.model_dump()
            # Reshape to Dome's expected format
            config = {
                "input-guards": dome_config["input_guards"],
                "output-guards": dome_config["output_guards"],
                **dome_config["guards"],
            }
            return Dome(
                dome_config=config,
                enforce=(self.mode == "enforce"),
            )
        except ImportError:
            logger.warning("vijil-dome not installed — content guards disabled")
            return None

    def _load_manifest(
        self, manifest: ToolManifest | Path | None
    ) -> ToolManifest | None:
        if manifest is None:
            return None
        if isinstance(manifest, Path):
            return ToolManifest.load(manifest)
        return manifest

    # -- Boot attestation --

    def attest(self) -> AttestationResult:
        """Verify all tools in the manifest. Call before serving requests."""
        from vijil.trust.attestation import ToolAttestationStatus
        from datetime import datetime, timezone

        if self._manifest is None:
            return AttestationResult(
                agent_identity=self._identity.spiffe_id or self._agent_id,
                tools=[],
                all_verified=True,
                timestamp=datetime.now(timezone.utc),
            )

        statuses: list[ToolAttestationStatus] = []
        for tool in self._manifest.tools:
            status = self._verify_tool_identity(tool)
            statuses.append(status)

        all_ok = all(s.verified for s in statuses)
        result = AttestationResult(
            agent_identity=self._identity.spiffe_id or self._agent_id,
            tools=statuses,
            all_verified=all_ok,
            timestamp=datetime.now(timezone.utc),
        )
        self._audit.emit_attestation(
            all_verified=all_ok, tool_count=len(statuses)
        )
        return result

    def _verify_tool_identity(self, tool: Any) -> Any:
        """Verify a single tool's SPIFFE identity via mTLS."""
        from vijil.trust.attestation import ToolAttestationStatus

        if not self._identity.is_attested():
            return ToolAttestationStatus(
                tool_name=tool.name,
                expected_identity=tool.identity,
                observed_identity=None,
                verified=False,
                error="Agent not attested — cannot verify tool identity",
            )

        # TODO: implement mTLS connection to tool endpoint, extract SVID
        # For now, return unverified (SPIRE integration in Task 2.1)
        return ToolAttestationStatus(
            tool_name=tool.name,
            expected_identity=tool.identity,
            observed_identity=None,
            verified=False,
            error="mTLS tool verification not yet implemented",
        )

    # -- Runtime enforcement --

    def guard_input(self, message: str) -> GuardResult:
        if self._dome is None:
            return GuardResult(
                flagged=False, enforced=False, score=0.0,
                guarded_response=None, exec_time_ms=0.0, trace=[],
            )
        scan = self._dome.guard_input(message, agent_id=self._agent_id)
        result = GuardResult.from_scan_result(scan)
        self._audit.emit_guard(
            "input", flagged=result.flagged, score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    def guard_output(self, response: str) -> GuardResult:
        if self._dome is None:
            return GuardResult(
                flagged=False, enforced=False, score=0.0,
                guarded_response=None, exec_time_ms=0.0, trace=[],
            )
        scan = self._dome.guard_output(response, agent_id=self._agent_id)
        result = GuardResult.from_scan_result(scan)
        self._audit.emit_guard(
            "output", flagged=result.flagged, score=result.score,
            exec_time_ms=result.exec_time_ms,
        )
        return result

    def guard_tool_response(self, tool_name: str, response: str) -> GuardResult:
        return self.guard_output(response)

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> ToolCallResult:
        result = self._policy.check(tool_name)
        self._audit.emit_tool_mac(
            tool_name, permitted=result.permitted,
            identity_verified=result.identity_verified,
        )
        return result

    # -- Tool wrapping --

    def wrap_tool(self, tool: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a tool callable with MAC + response guard."""
        tool_name = getattr(tool, "__name__", str(tool))

        @functools.wraps(tool)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # MAC check
            mac_result = self.check_tool_call(tool_name, kwargs)
            if not mac_result.permitted and self.mode == "enforce":
                raise PermissionError(
                    f"Tool '{tool_name}' denied: {mac_result.error}"
                )
            if not mac_result.permitted:
                logger.warning("Tool '%s' denied (warn mode): %s", tool_name, mac_result.error)

            # Execute tool
            result = tool(*args, **kwargs)

            # Guard tool response
            if isinstance(result, str) and self._dome is not None:
                guard = self.guard_tool_response(tool_name, result)
                if guard.flagged and self.mode == "enforce":
                    return guard.guarded_response
                if guard.flagged:
                    logger.warning(
                        "Tool '%s' response flagged (warn mode, score=%.2f)",
                        tool_name, guard.score,
                    )

            return result

        return wrapped

    def wrap_tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Wrap multiple tool callables."""
        return [self.wrap_tool(t) for t in tools]
```

- [ ] **Step 3: Run all trust tests**

Run: `poetry run pytest tests/unit/trust/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/vijil/trust/runtime.py tests/unit/trust/test_runtime.py
git commit -m "feat(trust): add TrustRuntime orchestrator with guards, MAC, and audit"
```

---

## Train 4: Framework Adapter + CLI

### Task 4.1: LangGraph adapter — secure_graph

**Files:**
- Create: `src/vijil/adapters/__init__.py`
- Create: `src/vijil/adapters/langgraph.py`
- Test: `tests/unit/adapters/__init__.py`
- Test: `tests/unit/adapters/test_langgraph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/adapters/test_langgraph.py
from unittest.mock import MagicMock, patch
import pytest


class TestSecureGraph:
    @pytest.fixture
    def mock_runtime(self):
        from vijil.trust.guard import GuardResult
        from vijil.trust.policy import ToolCallResult

        runtime = MagicMock()
        runtime.mode = "warn"
        runtime.guard_input.return_value = GuardResult(
            flagged=False, enforced=False, score=0.0,
            guarded_response=None, exec_time_ms=0.0, trace=[],
        )
        runtime.guard_output.return_value = GuardResult(
            flagged=False, enforced=False, score=0.0,
            guarded_response=None, exec_time_ms=0.0, trace=[],
        )
        runtime.attest.return_value = MagicMock(all_verified=True)
        runtime.wrap_tools.side_effect = lambda tools: tools
        return runtime

    def test_secure_graph_wraps_invoke(self, mock_runtime):
        from vijil.adapters.langgraph import SecureGraph

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"messages": ["response"]}

        secure = SecureGraph(graph=mock_graph, runtime=mock_runtime)
        result = secure.invoke({"messages": ["hello"]})

        mock_runtime.guard_input.assert_called_once()
        mock_runtime.guard_output.assert_called_once()
        assert result == {"messages": ["response"]}

    def test_invoke_blocked_in_enforce_mode(self, mock_runtime):
        from vijil.adapters.langgraph import SecureGraph
        from vijil.trust.guard import GuardResult

        mock_runtime.mode = "enforce"
        mock_runtime.guard_input.return_value = GuardResult(
            flagged=True, enforced=True, score=0.95,
            guarded_response="Blocked.", exec_time_ms=10.0, trace=[],
        )

        mock_graph = MagicMock()
        secure = SecureGraph(graph=mock_graph, runtime=mock_runtime)
        result = secure.invoke({"messages": ["hack the system"]})

        mock_graph.invoke.assert_not_called()  # blocked before reaching graph
        assert "Blocked" in str(result)
```

- [ ] **Step 2: Implement adapter**

```python
# src/vijil/adapters/langgraph.py
"""LangGraph adapter — secure_graph wraps StateGraph with trust enforcement."""

from __future__ import annotations

import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class SecureGraph:
    """Wrapped LangGraph with trust enforcement at invoke() and tool boundaries."""

    def __init__(self, *, graph: Any, runtime: Any) -> None:
        self._graph = graph
        self._runtime = runtime
        self._attestation = runtime.attest()

    @property
    def runtime(self) -> Any:
        return self._runtime

    @property
    def attestation(self) -> Any:
        return self._attestation

    def invoke(self, input: dict[str, Any], config: Any = None) -> dict[str, Any]:
        """Invoke the graph with trust enforcement."""
        # Extract user message for input guard
        messages = input.get("messages", [])
        last_message = messages[-1] if messages else ""
        message_text = (
            last_message.get("content", str(last_message))
            if isinstance(last_message, dict)
            else str(last_message)
        )

        # Input guard
        input_result = self._runtime.guard_input(message_text)
        if input_result.flagged and self._runtime.mode == "enforce":
            return {"messages": [input_result.guarded_response]}

        # Execute graph
        result = self._graph.invoke(input, config)

        # Output guard
        out_messages = result.get("messages", [])
        last_out = out_messages[-1] if out_messages else ""
        out_text = (
            last_out.get("content", str(last_out))
            if isinstance(last_out, dict)
            else str(last_out)
        )

        output_result = self._runtime.guard_output(out_text)
        if output_result.flagged and self._runtime.mode == "enforce":
            return {"messages": [output_result.guarded_response]}

        return result

    def stream(self, input: dict[str, Any], config: Any = None) -> Iterator[Any]:
        """Stream the graph with trust enforcement on accumulated output."""
        messages = input.get("messages", [])
        last_message = messages[-1] if messages else ""
        message_text = str(last_message) if not isinstance(last_message, dict) else last_message.get("content", "")

        input_result = self._runtime.guard_input(message_text)
        if input_result.flagged and self._runtime.mode == "enforce":
            yield {"messages": [input_result.guarded_response]}
            return

        accumulated: list[Any] = []
        for chunk in self._graph.stream(input, config):
            accumulated.append(chunk)
            yield chunk

        # Guard accumulated output after stream completes
        # (output guard runs but cannot retract already-yielded chunks)
        if accumulated:
            last_chunk = accumulated[-1]
            if isinstance(last_chunk, dict):
                text = str(last_chunk.get("messages", [""]))
                output_result = self._runtime.guard_output(text)
                if output_result.flagged:
                    logger.warning("Stream output flagged (score=%.2f)", output_result.score)


def secure_graph(
    graph: Any,
    *,
    client: Any,
    agent_id: str,
    manifest: Any = None,
    mode: str = "warn",
    **compile_kwargs: Any,
) -> SecureGraph:
    """Wrap a LangGraph StateGraph with trust enforcement.

    Replaces graph.compile(). Wraps tools with MAC + identity,
    then compiles the graph, then wraps invoke/stream with guards.
    """
    from vijil.trust.runtime import TrustRuntime

    runtime = TrustRuntime(
        client=client,
        agent_id=agent_id,
        manifest=manifest,
        mode=mode,  # type: ignore[arg-type]
    )

    # Wrap tools if the graph has them
    if hasattr(graph, "nodes"):
        for node_name, node_data in graph.nodes.items():
            if hasattr(node_data, "tools"):
                node_data.tools = runtime.wrap_tools(node_data.tools)

    # Compile the graph
    compiled = graph.compile(**compile_kwargs) if hasattr(graph, "compile") else graph

    return SecureGraph(graph=compiled, runtime=runtime)
```

```python
# src/vijil/adapters/__init__.py
"""Framework adapters for the Vijil Trust Runtime."""
```

- [ ] **Step 3: Run tests**

Run: `poetry run pytest tests/unit/adapters/test_langgraph.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/vijil/adapters/ tests/unit/adapters/
git commit -m "feat(trust): add LangGraph adapter — secure_graph replaces graph.compile()"
```

---

### Task 4.2: Manifest CLI command

**Files:**
- Create: `src/vijil_cli/porcelain/manifest_cmd.py`
- Modify: `src/vijil_cli/main.py`

- [ ] **Step 1: Implement CLI command**

```python
# src/vijil_cli/porcelain/manifest_cmd.py
"""vijil manifest sign|verify"""

import json
from pathlib import Path

import typer

from vijil_cli import state
from vijil_cli.output import format_output


def register_manifest(app: typer.Typer) -> None:
    manifest_app = typer.Typer(name="manifest", help="Manage tool manifests.", no_args_is_help=True)

    @manifest_app.command("sign")
    def sign(
        input_path: str = typer.Argument(help="Path to unsigned manifest JSON"),
        output_path: str = typer.Option(None, "--output", "-o", help="Output path (default: overwrite input)"),
    ) -> None:
        """Sign a tool manifest via Console."""
        ctx = state.get_ctx()
        manifest_data = json.loads(Path(input_path).read_text())
        result = ctx.client._http.post("/manifests/sign", json=manifest_data)
        manifest_data["signature"] = result["signature"]
        out = Path(output_path or input_path)
        out.write_text(json.dumps(manifest_data, indent=2))
        typer.echo(f"Signed manifest written to {out}")

    @manifest_app.command("verify")
    def verify(
        path: str = typer.Argument(help="Path to signed manifest JSON"),
    ) -> None:
        """Verify a manifest signature."""
        from vijil.trust.manifest import ToolManifest
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        ctx = state.get_ctx()
        manifest = ToolManifest.load(Path(path))
        key_data = ctx.client._http.get("/manifests/public-key")
        public_key = load_pem_public_key(key_data["public_key"].encode())
        if manifest.verify_signature(public_key):
            typer.echo("Manifest signature valid.")
        else:
            typer.echo("Manifest signature INVALID.", err=True)
            raise typer.Exit(code=4)

    app.add_typer(manifest_app)
```

- [ ] **Step 2: Register in main.py**

Add to `src/vijil_cli/main.py`:
```python
from vijil_cli.porcelain.manifest_cmd import register_manifest
register_manifest(app)
```

- [ ] **Step 3: Commit**

```bash
git add src/vijil_cli/porcelain/manifest_cmd.py src/vijil_cli/main.py
git commit -m "feat(cli): add vijil manifest sign|verify commands"
```

---

## Train 5: Package and Integration

### Task 5.1: Add trust extra to pyproject.toml

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/vijil/__init__.py`

- [ ] **Step 1: Add optional dependencies**

Add to `pyproject.toml`:
```toml
[tool.poetry.extras]
trust = ["vijil-dome", "cryptography", "opentelemetry-api"]
spiffe = ["py-spiffe"]
```

Add under `[tool.poetry.dependencies]`:
```toml
vijil-dome = {version = ">=1.5.0", optional = true}
cryptography = {version = ">=43.0.0", optional = true}
opentelemetry-api = {version = ">=1.34.0", optional = true}
py-spiffe = {version = ">=0.3.0", optional = true}
```

- [ ] **Step 2: Export TrustRuntime from top-level**

Add to `src/vijil/__init__.py`:
```python
try:
    from vijil.trust.runtime import TrustRuntime
except ImportError:
    pass  # trust extra not installed
```

- [ ] **Step 3: Run all tests**

Run: `poetry install --all-extras && poetry run pytest tests/unit/ -v`
Expected: All 237+ existing tests + all new trust tests PASS

- [ ] **Step 4: Lint and type check**

Run: `poetry run ruff check src/ tests/ && poetry run mypy src/`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/vijil/__init__.py
git commit -m "feat(sdk): add [trust] and [spiffe] extras to pyproject.toml"
```

---

### Task 5.2: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/sdk-reference.md`

- [ ] **Step 1: Add trust runtime section to README**

Add a section after "Framework integrations" in README.md:

```markdown
## Trust runtime

Embed security enforcement directly in your agent:

\`\`\`bash
pip install vijil-sdk[trust]
\`\`\`

\`\`\`python
from vijil.adapters.langgraph import secure_graph

# Replace graph.compile() with secure_graph()
app = secure_graph(graph, client=Vijil(), agent_id="travel-agent")
result = app.invoke({"messages": [user_input]})
\`\`\`

This adds Dome content Guards on every LLM call, mandatory access control on tool invocations, and SPIFFE-based identity verification. See [Trust Runtime Design](docs/plans/2026-04-03-trust-runtime-design.md) for the full architecture.
```

- [ ] **Step 2: Add trust section to SDK reference**

Add a new section to `docs/sdk-reference.md` documenting TrustRuntime, secure_graph, and the trust models.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/sdk-reference.md
git commit -m "docs(sdk): add trust runtime section to README and SDK reference"
```

---

## Summary

| Train | Tasks | Commits | Estimated lines |
|-------|-------|---------|-----------------|
| 1. Domain models | 4 tasks | 4 commits | ~400 |
| 2. Adapters | 2 tasks | 2 commits | ~200 |
| 3. Application | 1 task | 1 commit | ~250 |
| 4. Framework + CLI | 2 tasks | 2 commits | ~300 |
| 5. Package + docs | 2 tasks | 2 commits | ~100 |
| **Total** | **11 tasks** | **11 commits** | **~1250** |

One branch, one PR: `vin/sdk-trust-runtime`
