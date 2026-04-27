# Vijil Trust Runtime — Gap Analysis and Roadmap

**Date:** 2026-04-10
**Status:** Active
**Author:** Vin

---

## Overview

This document maps every component required for a complete Vijil Trust Runtime deployment on both EKS-hosted and managed AgentCore agents. Each component is rated as Built, Verified, Stub, or Missing. The roadmap organizes remaining work into four phases, each delivering incremental value.

---

## Component Status

### SDK Trust Runtime (vijil-sdk)

| Component | Status | Evidence | Gap |
|---|---|---|---|
| TrustRuntime orchestrator | Built | 8 unit tests, PR #1 | No gaps |
| AgentIdentity (SPIFFE) | **Verified live** | SVID fetched on vijil-spire-poc EKS cluster | SVID rotation watcher not implemented |
| AgentIdentity (API key fallback) | Built | 4 unit tests | No gaps |
| AgentIdentity (JWT-SVID for managed runtime) | **Missing** | — | OIDC bridge path not implemented |
| ToolManifest (Ed25519 signing) | Built | 11 unit tests | No gaps |
| ToolPolicy (MAC) | Built | 7 unit tests | No gaps |
| GuardResult (Dome wrapper) | Built | 5 unit tests | No gaps |
| AuditEmitter | Built | 3 unit tests | Console streaming adapter missing |
| LangGraph adapter (secure_graph) | Built | 8 unit tests | Not tested with real LangGraph graph |
| ADK adapter | **Missing** | — | Same TrustRuntime, different wrapper |
| `_verify_tool_identity()` mTLS | **Stub** | mTLS pattern verified live, not wired into method | Must use proven mTLS pattern from PoC |
| Boot attestation | **Stub** | Calls stub `_verify_tool_identity()` | Blocked by above |
| `wrap_tool()` identity re-verification | **Missing** | MAC check works, identity check does not | Must re-verify SVID on each tool call |
| SVID rotation watcher | **Missing** | — | Must watch Workload API for renewed SVIDs |
| Async guard variants | Declared | API exists | Wraps sync — no true async Dome integration |

### Console API (vijil-console)

| Endpoint | Status | Gap |
|---|---|---|
| `GET /agents/{id}/constraints` | **Missing** | Must return AgentConstraints (dome config + tool permissions + org rules + enforcement mode) |
| `POST /manifests/sign` | **Missing** | Must sign manifest JSON with Console's Ed25519 private key |
| `GET /manifests/public-key` | **Missing** | Must return public key for manifest verification |
| `POST /agents/{id}/audit` | **Missing** | Must accept streamed audit events from trust runtime |
| Organization constraints model | **Missing** | No concept of org-level mandatory rules in Console today |
| Dome config in AgentConstraints format | **Partial** | `Dome.create_from_vijil_agent()` exists but format may differ from AgentConstraints schema |

### Build Pipeline

| Component | Status | Gap |
|---|---|---|
| ODRL policy definition in Console | Partial | Console has policies, not the typed template + tool permission format |
| Policy compiler (ODRL → typed template + manifest) | **Missing** | Developer runs `vijil manifest compile` or Darwin generates it |
| `vijil manifest sign` CLI | Built | Calls Console endpoint that does not exist yet |
| `vijil manifest verify` CLI | Built | Calls Console public key endpoint that does not exist yet |
| Darwin evolve → manifest generation | **Missing** | Darwin does not produce tool manifests today |

### Infrastructure (SPIRE + EKS)

| Component | Status | Gap |
|---|---|---|
| SPIRE Server on EKS | **Running** | vijil-spire-poc, emptyDir storage, disk CA |
| SPIRE Agent (DaemonSet) | **Running** | 2 agents attested via k8s_psat |
| CSI Driver (socket mount) | **Running** | Verified — pods receive socket via CSI volume |
| Agent SVID issuance | **Verified** | `spiffe://vijil.ai/ns/default/agent/test-agent` |
| Tool SVID issuance | **Verified** | `spiffe://vijil.ai/ns/default/tool/echo-tool/v1` |
| Agent-to-tool mTLS | **Verified** | Bilateral authentication confirmed |
| Workload registration automation | **Missing** | Manual `spire-server entry create` today |
| OIDC bridge (managed runtime → JWT-SVID) | **Missing** | Required for managed AgentCore |
| Production hardening | **Missing** | Needs PostgreSQL datastore, AWS KMS CA, OIDC/IRSA for EBS CSI |
| SVID TTL tuning | Default (1 hour) | May need shorter TTL for high-security agents |

### LLM Proxy (credential management)

| Component | Status | Gap |
|---|---|---|
| Go proxy binary | **Missing** | Designed in SPIFFE/SPIRE PRD, not built |
| Vault deployment | **Missing** | Vault not deployed, API keys still in env vars |
| Proxy routing config | **Missing** | Agent-to-provider mapping |
| Proxy telemetry (OTel) | **Missing** | Per-agent cost attribution spans |

### Dome Integration

| Component | Status | Gap |
|---|---|---|
| Dome library (vijil-dome) | Exists (v1.5.0) | No changes needed for trust runtime |
| Dome guard config from Console | Exists | `Dome.create_from_vijil_agent()` fetches config |
| Dome inference decoupling | **Missing** | ML models bundled with torch. Spec 2 offloads to vijil-inference. |
| Dome guards on tool responses | Designed | In TrustRuntime but not tested with real Dome |

---

## Roadmap

### Phase 1 — Runtime enforcement without SPIFFE

**What ships:** Content Guards + tool MAC via API key auth. Developer gets `secure_graph()` working with Dome and tool policy from Console.

**Effort:** ~5 days

| Task | Repo | Depends on |
|---|---|---|
| Build `GET /agents/{id}/constraints` endpoint | vijil-console | — |
| Test `secure_graph()` with a real LangGraph agent | vijil-sdk | Constraints endpoint |
| Test Dome guards end-to-end (install vijil-dome in test env) | vijil-sdk | — |
| Deploy a LangGraph agent with trust runtime on EKS | vijil-sdk | Constraints endpoint |

**Delivered value:** A developer replaces `graph.compile()` with `secure_graph()` and gets Dome input/output Guards and tool-level MAC. Policy comes from Console. Works with API key auth — no SPIRE required.

### Phase 2 — Boot attestation with SPIFFE

**What ships:** Agents verify tool identities at boot via mTLS. Agent identity via SPIFFE SVID.

**Effort:** ~5 days

| Task | Repo | Depends on |
|---|---|---|
| Wire `_verify_tool_identity()` with mTLS (pattern proved in PoC) | vijil-sdk | — |
| Add SVID rotation watcher to AgentIdentity | vijil-sdk | — |
| Add runtime tool identity re-verification to `wrap_tool()` | vijil-sdk | Above |
| Build `POST /manifests/sign` and `GET /manifests/public-key` | vijil-console | — |
| Build policy compiler (`vijil manifest compile`) | vijil-sdk | Sign endpoint |
| Automate SPIRE workload registration (controller or CI step) | infra | — |
| Test boot attestation: agent rejects tool with wrong SVID | vijil-sdk | All above |

**Delivered value:** Agent performs measured boot — verifies every tool identity before serving. Signed manifests prevent tool substitution. Enterprise CISOs recognize this as the TPM/secure-boot pattern applied to agents.

### Phase 3 — Managed AgentCore support

**What ships:** Agents running in managed AgentCore microVMs get SPIFFE identity via JWT-SVID.

**Effort:** ~5 days

| Task | Repo | Depends on |
|---|---|---|
| Build OIDC bridge: AWS IAM token → JWT-SVID | infra (new service) | SPIRE Server running |
| Add JWT-SVID path to AgentIdentity | vijil-sdk | OIDC bridge |
| Test managed runtime: agent in microVM gets SPIFFE identity | vijil-sdk | Both above |
| Build `POST /agents/{id}/audit` Console endpoint | vijil-console | — |
| Add Console audit streaming to AuditEmitter | vijil-sdk | Audit endpoint |
| ADK adapter (fast follow — same TrustRuntime) | vijil-sdk | — |

**Delivered value:** Same SPIFFE identity model works in both EKS-hosted and managed AgentCore. Audit events stream to Console. ADK agents supported.

### Phase 4 — Transient credentials (zero static secrets)

**What ships:** LLM Proxy eliminates API keys from agent environments. Vault stores credentials transiently.

**Effort:** ~8 days

| Task | Repo | Depends on |
|---|---|---|
| Build LLM Proxy (Go binary: mTLS + Vault) | New repo | SPIRE running |
| Deploy Vault with SPIFFE auth method | infra | SPIRE running |
| Configure per-agent Vault policies | infra | Vault deployed |
| Proxy routing config (agent → provider mapping) | infra | Proxy built |
| Proxy OTel telemetry (per-agent cost attribution) | proxy repo | Proxy built |
| Production SPIRE hardening (PostgreSQL, AWS KMS CA) | infra | — |
| End-to-end: agent → trust runtime → proxy → Vault → LLM | all | All above |

**Delivered value:** No agent process holds a static API key. All credentials are transient, attestation-gated, and automatically rotated. The LiteLLM-class supply chain attack is no longer possible.

---

## Cost Summary

| Phase | Effort | New infra cost |
|---|---|---|
| Phase 1 | ~5 days | None (uses existing EKS + API key auth) |
| Phase 2 | ~5 days | SPIRE PoC cluster ~$133/mo (already running) |
| Phase 3 | ~5 days | OIDC bridge service (minimal — single container) |
| Phase 4 | ~8 days | Vault (~$50/mo), LLM Proxy (~$30/mo) |

---

## Related

- **Trust Runtime design spec:** `docs/plans/2026-04-03-trust-runtime-design.md`
- **Trust Runtime PRD:** `docs/plans/2026-04-03-trust-runtime-prd.md`
- **Trust Runtime implementation:** PR #1 (branch `vin/sdk-trust-runtime`)
- **SPIFFE/SPIRE PRD:** `~/Downloads/SPIFFE-SPIRE-LLM-Proxy-PRD-Design.docx`
- **SPIRE PoC cluster:** vijil-spire-poc (EKS us-west-2)
- **IAM vs SPIFFE analysis:** `docs/plans/2026-04-10-iam-vs-spiffe-identity.md`
