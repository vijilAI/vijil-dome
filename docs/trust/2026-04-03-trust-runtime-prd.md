# Vijil Trust Runtime — Product Requirements Document

**Date:** 2026-04-03
**Status:** Implementation in progress (PR #1)
**Author:** Vin

---

## 1. Summary

The Vijil Trust Runtime embeds security enforcement inside AI agents at the framework level. Instead of evaluating agents after they are built or protecting them with a network proxy, the runtime wraps every LLM call and tool invocation with mandatory access control, content Guards, and cryptographic identity verification. A developer replaces one line of code — `graph.compile()` becomes `secure_graph()` — and gets measured boot, continuous attestation, and Console-sourced policy enforcement.

## 2. Problem

Agent developers building with LangGraph, ADK, and other frameworks face three security gaps that Vijil's existing products do not close:

**No mandatory access control on tool calls.** When an LLM hallucinates a tool invocation — `charge_credit_card` instead of `search_hotels` — nothing in the framework prevents execution. The LLM decides what to call, and the framework obeys. Tool authorization does not exist.

**No security enforcement during development.** Vijil Diamond evaluates agents after they are built, taking 5-15 minutes per run. Dome protects agents at the network edge as a sidecar. Neither gives the developer in-code, in-IDE feedback at the moment a security violation occurs. Developers discover issues minutes to hours later.

**No workload identity for agents.** Agents authenticate with static API keys stored in environment variables. The LiteLLM supply chain compromise (March 2026) demonstrated that a single malicious dependency can exfiltrate every API key in the organization. Agents have no cryptographic identity, no attestation, and no way to prove they have not been tampered with.

## 3. Solution

The Trust Runtime is a Python module in the Vijil SDK that wraps agent framework invocations at two levels:

- **Tool callables** — each tool function is wrapped with mandatory access control (is this tool permitted?) and identity verification (is this tool endpoint the one declared in the manifest?).
- **Graph entry point** — the agent's `invoke()` and `stream()` methods are wrapped with Dome content Guards (input scanning, output scanning) and audit logging.

The runtime operates in three phases:

**Build time.** The developer compiles an ODRL policy from Console into a signed tool manifest — a list of (tool name, SPIFFE identity, endpoint) tuples signed with Ed25519. The manifest declares exactly which tools the agent is authorized to use, each bound to a cryptographic identity.

**Boot time.** Before the agent serves requests, the runtime verifies every tool in the manifest. Each tool endpoint presents its SPIFFE SVID via mTLS. The runtime checks the SVID against the manifest. If any tool fails verification, the agent refuses to start. This is measured attestation — the same pattern enterprise security teams use for server infrastructure (TPM, UEFI secure boot).

**Runtime.** Every user message passes through Dome input Guards. Every tool call passes through MAC and identity re-verification. Every tool response passes through Dome output Guards (defending against prompt injection via tool responses). Every final response passes through Dome output Guards. All decisions are logged as OpenTelemetry spans tagged with the agent's SPIFFE ID.

## 4. Developer Experience

```python
# Before Vijil (no security):
app = graph.compile()
result = app.invoke({"messages": [user_input]})

# After Vijil (one changed line):
from vijil.adapters.langgraph import secure_graph
app = secure_graph(graph, client=Vijil(), agent_id="travel-agent")
result = app.invoke({"messages": [user_input]})
```

- **Warn mode** (development): policy violations log warnings, execution continues. The developer sees security issues in real time without breaking their workflow.
- **Enforce mode** (production): policy violations block execution. The agent rejects unsafe inputs, denies unauthorized tool calls, and filters unsafe outputs.

Guard configuration and tool permissions come from Console. The security team controls policy centrally. The developer writes code.

## 5. Threat Model

| Threat | Defense | Phase |
|---|---|---|
| Tool substitution — attacker replaces a legitimate tool | SPIFFE identity verification via mTLS at boot and on every call | Boot + Runtime |
| Tool compromise — legitimate tool behaves maliciously | Dome output Guards on tool responses catch adversarial content; typed template constrains action space | Runtime |
| Prompt injection via tool responses — tool returns data designed to hijack agent behavior | Dome Guards scan every tool response before it reaches the LLM | Runtime |
| Supply chain credential theft — malicious dependency exfiltrates API keys | SPIFFE SVIDs replace static API keys; credentials are short-lived and attestation-gated (per SPIFFE/SPIRE PRD) | Boot + Runtime |

## 6. Architecture

```
Agent Process
  ┌──────────────────────────────────────────┐
  │  Agent Code ──→ Trust Runtime            │
  │                  ├── Dome Guards          │
  │                  ├── Tool Policy (MAC)    │
  │                  ├── Identity (SPIFFE)    │
  │                  └── Audit (OTel)         │
  └──────────┬──────────────────┬────────────┘
             │                  │
       LLM Proxy          MCP Tool Servers
       (mTLS + Vault)     (mTLS + SPIFFE)
             │
       LLM Providers
```

**Console** provides AgentConstraints (Guard config + tool permissions + organization rules). **Dome** runs in-process as a library (not a sidecar). **SPIRE** provides workload identity. **Vault** stores transient credentials (per separate SPIFFE/SPIRE PRD).

## 7. Key Decisions

| Decision | Rationale |
|---|---|
| Wrap at invoke() boundary, not framework internals | Survives framework version changes. LangGraph, ADK, CrewAI all have `invoke()`. |
| Dome as in-process library | Two-line developer adoption. No sidecar, no infrastructure. `pip install vijil-sdk[trust]`. |
| Policy from Console, not code | Prevents drift between security team intent and runtime behavior. One source of truth. |
| SPIFFE identity with API key fallback | Developers without SPIRE infrastructure can still use Guards and MAC in development. |
| Signed tool manifests | Prevents tool substitution attacks. Enterprise security teams recognize this as measured boot. |

## 8. Scope

**In scope (Spec 1 — PR #1, implemented):**
- TrustRuntime orchestrator
- Dome Guard integration (input, output, tool response)
- Tool-level mandatory access control
- SPIFFE-based agent identity with API key fallback
- Signed tool manifests with Ed25519
- Boot-time measured attestation
- LangGraph adapter (`secure_graph()`)
- `vijil manifest sign|verify` CLI commands
- Audit event emission (pluggable sink)

**Fast follow:**
- ADK adapter (same runtime, different wrapper)
- Console endpoints: `GET /agents/{id}/constraints`, `POST /manifests/sign`

**Future specs:**
- Behavioral anomaly detection (continuous integrity scoring)
- Dome inference decoupling (offload ML models to vijil-inference)
- SLSA compilation provenance (signed policy compiler)
- Inter-agent attestation (multi-agent graphs)

## 9. Success Criteria

1. A developer adds trust enforcement to a LangGraph agent by changing one line of code.
2. `pip install vijil-sdk[trust]` completes in under 30 seconds (no torch dependency).
3. A tool with a mismatched SPIFFE identity prevents the agent from booting in enforce mode.
4. Every LLM call and tool invocation passes through policy enforcement with under 50ms p99 overhead.
5. A CISO reading this document recognizes measured boot and continuous attestation — patterns already used for server infrastructure.
