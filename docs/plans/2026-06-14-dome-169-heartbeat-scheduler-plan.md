---
title: DOME-169 Enforcement-alive heartbeat — scheduler, real probe, signing seam — Implementation Plan
date: 2026-06-14
persona: Risk Owner (operator who must detect an agent that goes dark)
design: docs/trust/2026-06-04-tamper-evident-identity-mac-plan.md
prfaq: docs/trust/2026-06-04-in-process-control-bypass-threat-model.md
prd: docs/trust/2026-06-04-in-process-control-bypass-threat-model.md
linear: https://linear.app/vijil/issue/DOME-169
branch: ciphr/dome-169-heartbeat-scheduler-signer
status: Draft
graphify_snapshot: 98be0bf445f017f87fb5ac46bf29ab5534053728
---

# DOME-169 Enforcement-alive heartbeat — Implementation Plan

**Goal:** Drive `emit_heartbeat()` on a self-monitoring cadence, report a real detector-reachability signal, and add a signing seam — so a registered agent that goes dark or silently downgrades is detectable in-process, arming B4's liveness reconciler.

**Architecture:** B3 (PR #245) landed the `Heartbeat` event shape and `emit_heartbeat()`; nothing calls it, `detector_reachable` is a proxy, the beacon is unsigned. This plan adds a `BeaconSigner` seam (Domain), a real bounded reachability probe plus a supervised daemon-thread scheduler with an in-process liveness self-check (Application), and forwards an opt-in `heartbeat_interval` through `secure_agent` and all three adapters (Adapter).

**Snapshot note:** `graph.json` was built at `98be0bf`; it lags `main` (`e15b6d4`) by the 167/168/170 merges, none of which touch the heartbeat/identity/adapter symbols below. Every file:line claim was re-verified by direct read against `e15b6d4`.

**Design-divergence flag (vs design B3).** The design (`...mac-plan.md`) specifies the beacon "SVID-signed now." This plan ships the signing *seam* and a real `X509BeaconSigner` exercised with a synthetic key, but the **default is `UnwiredBeaconSigner` (`signed=False`)** because `_svid.private_key`/`_svid.leaf` exist only under live SPIRE attestation (cluster-gated). Wiring the live signer + Console verify is deferred to **DOME-179** (filed). This is a conscious, tracked relaxation of "signed now" to "seam now, live-signed at cluster" — surfaced here, not silent.

## Transitive Infrastructure

| Runtime check | Required infra | Availability |
|---|---|---|
| Unit + integration suite (`poetry run pytest vijil_dome/tests/trust/`) | Poetry env; no network; synthetic EC key + injected clock in-test | Local |
| Signer unit tests | `cryptography` (already a dep, used by `identity.py`) | Local |

No live SPIRE socket, Console, or cluster is required — the live signer is out of scope (DOME-179).

## Dependency Graph

```
Train 1 (Domain):                  1.1
Train 2 (Application, after T1):    2.1 -> 2.2
Train 3 (Adapter+Test, after T2):  3.1 -> 3.2
```

**Layer-order note:** Train 3's Adapter task (3.1) forwards a param into the `TrustRuntime` API that Train 2's Application task (2.2) introduces; the param cannot be forwarded before it exists, so Application precedes Adapter here (dependency-justified deviation from the canonical Adapter→Application order). Trains are commit-groups within one PR (DOME-169); the `UnwiredBeaconSigner`/unset-interval defaults preserve current behavior, so each train is independently revertable.

## Train 1: Signing seam (Domain)

### Task 1.1: BeaconSigner Protocol + signers + Heartbeat.signature

**Layer:** Domain
**Files:** Create `vijil_dome/trust/signing.py`; Modify `vijil_dome/trust/audit.py` (`Heartbeat` + `emit_heartbeat`); Test `vijil_dome/tests/trust/test_beacon_signing.py`
**Linear Issue:** DOME-169
**Commit:** `feat(dome): add BeaconSigner seam + signed heartbeat envelope [DOME-169]`
**Target size:** 280 lines

Define `BeaconSignature` (Pydantic: `alg`, `signature` b64, `cert_chain` PEM list, `signed_subject` SPIFFE id) and `BeaconSigner` Protocol `sign(beacon: Heartbeat) -> BeaconSignature | None`. `UnwiredBeaconSigner.sign` returns `None` (beacon ships `signed=False`) — the honest default for unattested/api-key agents. `X509BeaconSigner.sign` signs canonical JSON of the beacon's identity-bearing fields with the SVID private key (ECDSA/RSA by key type) and attaches the leaf cert. Add `signature: BeaconSignature | None = None` to `Heartbeat`; thread through `AuditEmitter.emit_heartbeat`. Tests: synthetic `ec.generate_private_key` + self-signed cert → sign/verify round-trip; unwired → `None`; canonical-JSON determinism.

**Context-graph audit:**
- `trust_audit_heartbeat` @ `audit.py:L27` confirms `Heartbeat` location ✓ (direct-read verified, e15b6d4)
- `trust_audit_auditemitter_emit_heartbeat` @ `audit.py:L118` confirms emit signature ✓
- `trust_identity_agentidentity` @ `identity.py:L34`; `_svid.private_key`/`.leaf` used by `mtls_context` (L285-286) ✓

## Train 2: Probe + scheduler (Application)

### Task 2.1: Real, bounded detector_reachable probe

**Layer:** Application
**Files:** Modify `vijil_dome/trust/runtime.py` (`emit_heartbeat` + `_probe_detector_reachable`); Test `vijil_dome/tests/trust/test_heartbeat_probe.py`
**Linear Issue:** DOME-169
**Commit:** `feat(dome): bounded detector-reachability probe for the heartbeat [DOME-169]`
**Target size:** 170 lines

Replace the `detector_reachable = self._dome is not None and not self._guards_disabled` proxy (runtime.py:679) with `_probe_detector_reachable()`: `False` when `_dome is None` (not configured) or `_guards_disabled`; else run a fixed benign canned string through `self._dome.guard_input` directly (bypasses the audit-emitting runtime wrapper at runtime.py:337) inside a try — success ⇒ `True`, any exception ⇒ `False` (never propagates). **Cost bound:** cache the result with a TTL (≥ the heartbeat interval) so the real detectors (HF/LLM/OpenAI-moderation can be in the config) run at most once per cadence, not per call; document the per-probe inference cost in the docstring. `guards_constructed` stays `_dome is not None and not _guards_disabled` — the two fields now mean different things. Tests: four states (none / disabled / probe-raises / probe-succeeds) with a stub Dome; TTL cache hit.

**Context-graph audit:**
- `trust_runtime_trustruntime_emit_heartbeat` @ `runtime.py:L660`; proxy at L679; `runtime.guard_input`→`self._dome.guard_input`+`emit_guard` at L337 ✓ (direct-read verified)
- `vijil_dome_dome_dome_guard_input` @ `Dome.py:L354` confirms `Dome.guard_input` (no audit emit of its own) ✓

### Task 2.2: Supervised scheduler + in-process liveness self-check

**Layer:** Application
**Files:** Modify `vijil_dome/trust/runtime.py` (`__init__` params, `start_heartbeat`/`stop_heartbeat`/`heartbeat_health`, watchdog); Test `vijil_dome/tests/trust/test_heartbeat_scheduler.py`
**Linear Issue:** DOME-169
**Commit:** `feat(dome): supervised heartbeat scheduler with in-process liveness check [DOME-169]`
**Target size:** 250 lines

Add `heartbeat_interval: float | None = None`, `beacon_signer: BeaconSigner | None = None` (default `UnwiredBeaconSigner`) to `__init__`. `start_heartbeat()` spawns a `threading.Thread(daemon=True)` looping `emit_heartbeat(); _clock.wait(interval)`; records `_last_emit_at`; per-iteration `try` so one failure never kills the loop, and on an escaped error it loudly logs + sets a `_thread_failed` flag. **In-process liveness (the design's "dead loop is loud"):** `heartbeat_health()` returns `{alive, last_emit_age_s, thread_failed}` and is `alive=False` when `last_emit_age_s > 2×interval` — a host or B4 can poll it WITHOUT depending on the deferred Console sweep. `stop_heartbeat()` signals via `threading.Event` and `join(timeout=...)` (bounded, no hang); idempotent. If `heartbeat_interval` is set, `__init__` auto-starts (documented construction side-effect; daemon, never blocks exit; one thread per runtime instance — documented). Inject a clock seam so tests are deterministic (no real sleeps). `emit_heartbeat` attaches `beacon_signer.sign(...)`. Tests: clock-driven ≥2 beacons; bounded idempotent stop; injected per-iteration error keeps loop alive + `heartbeat_health().alive` stays true; simulated dead thread ⇒ `alive=False`.

**Context-graph audit:**
- `trust_runtime_trustruntime` @ `runtime.py:L33` confirms `__init__` extension point; no thread/timer exists in `trust/` (net-new) ✓ (direct-read verified)

## Train 3: Adapter wiring + integration (Adapter, Test)

### Task 3.1: Forward heartbeat_interval through secure_agent + all three adapters

**Layer:** Adapter
**Files:** Modify `vijil_dome/trust/adapters/auto.py`, `langgraph.py`, `adk.py`, `strands.py`; Test `vijil_dome/tests/trust/adapters/test_secure_agent_heartbeat.py`
**Linear Issue:** DOME-169
**Commit:** `feat(dome): wire heartbeat_interval through secure_agent and adapters [DOME-169]`
**Target size:** 150 lines

Add `heartbeat_interval`/`beacon_signer` to `secure_agent` (auto.py:31) and thread them into **all three** adapters' `TrustRuntime(...)` constructions — `langgraph.py:190`, `adk.py:71`, `strands.py:141`. **Correction:** langgraph does NOT get this free; its `**compile_kwargs` route to `graph.compile()` (langgraph.py:200), never to `TrustRuntime`, so all three need explicit threading. Tests assert each framework's runtime has a live scheduler after `secure_agent(..., heartbeat_interval=...)`.

**Context-graph audit:**
- `adapters_auto_secure_agent` @ `auto.py:L31`; `secure_graph` `TrustRuntime(...)` at langgraph.py:190 takes fixed args only (`**compile_kwargs`→`graph.compile()` L200); `adk.py:71`, `strands.py:141` likewise ✓ (direct-read verified)

### Task 3.2: End-to-end heartbeat integration test

**Layer:** Test
**Files:** Test `vijil_dome/tests/trust/test_heartbeat_e2e.py`
**Linear Issue:** DOME-169
**Commit:** `test(dome): end-to-end heartbeat cadence, probe, liveness, signing [DOME-169]`
**Target size:** 160 lines

Capture beacons via an audit sink. With a stub Dome + synthetic-key `X509BeaconSigner` + injected clock: assert beacons emit on cadence, carry a signature that verifies, `detector_reachable` flips with stubbed backend health, `heartbeat_health().alive` reflects a stalled loop, and the unwired default ships `signed=False`. Drives a real `TrustRuntime`.

**Context-graph audit:**
- `vijil_dome_tests_trust_test_heartbeat_b3_py` confirms canonical trust test dir `vijil_dome/tests/trust/` ✓

## Deferred / Out of scope

| Deferred item | Why | Tracking |
|---|---|---|
| Live `X509BeaconSigner` wired to a real SPIRE SVID (signs every beacon on a real install) | `_svid.private_key` requires live SPIRE attestation (cluster) | **DOME-179** |
| Console-side beacon signature verification + bind to B4 liveness record | Needs Console + cluster; pairs with B4 (DOME-174) and CON-525 mTLS swap | **DOME-179** |
| TEE remote-attestation quote (replaces SVID-sign) | Roadmap (Phala × Vijil) | design doc roadmap |

## Pre-Commit Checklist

- [ ] `poetry run pytest vijil_dome/tests/trust/` green (2 output-detection tests fail locally without `OPENAI_API_KEY` — pre-existing, pass in CI)
- [ ] `make lint` clean (ruff + mypy via mypy.ini, src AND tests)
- [ ] No silent fail-open: probe exceptions ⇒ `detector_reachable=False` (never propagate); a dead scheduler ⇒ `heartbeat_health().alive=False` in-process (not reliant on deferred B4)
- [ ] `stop_heartbeat` join is bounded; scheduler tests use an injected clock (no real-sleep flake)
- [ ] `vijil-steelman` + silent-failure-hunter PASS on the combined diff
- [ ] PR body states the DOME-179 deferred boundary + the design-B3 divergence

## Sizing Summary

| Task | Layer | Lines |
|---|---|---|
| 1.1 BeaconSigner seam + signature | Domain | 280 |
| 2.1 Bounded reachability probe | Application | 170 |
| 2.2 Scheduler + liveness self-check | Application | 250 |
| 3.1 secure_agent + 3-adapter wiring | Adapter | 150 |
| 3.2 E2E integration test | Test | 160 |
| **Total** | | **1010** |
