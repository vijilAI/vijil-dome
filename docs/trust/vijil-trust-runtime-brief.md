# Vijil Trust Runtime

### Measured Boot and Continuous Attestation for AI Agents

---

## The Problem

### Agent frameworks have no attestation model

AI agent frameworks — LangGraph, Google ADK, CrewAI, Strands — execute LLM calls and tool invocations on behalf of the developer. None of these frameworks provide a mechanism to attest the identity of the agent itself, verify the identity of the tools it calls, or constrain which resources it may access. An agent built with LangGraph can call any tool the LLM selects, connect to any endpoint, and access any API key in its environment. The framework enforces no identity, no authorization, and no access control boundary between the agent and the infrastructure it operates on.

### Agents inherit user privileges and retain persistent access

The current workaround is to pass the developer's own credentials — LLM API keys, cloud tokens, database passwords — into the agent's runtime environment via environment variables, mounted secrets, or configuration files. These credentials grant the agent the same privileges as the developer, persist on disk for the lifetime of the deployment, and are accessible to every dependency in the agent's process.

This creates two risks that OWASP identifies in its Top 10 for LLM Applications: *excessive agency* (the agent can do more than it should) and *insecure output handling* (the agent's dependencies can access secrets they should never see). On March 24, 2026, the LiteLLM supply chain compromise demonstrated the consequence. A credential-stealing payload in versions 1.82.7 and 1.82.8 of the `litellm` PyPI package exfiltrated SSH keys, cloud credentials, API keys, and Kubernetes configs from every system that installed it. LiteLLM is, by design, the component that holds every LLM API key in the organization. Every key was compromised in a single package update.

---

## The Solution

### What the Vijil Trust Runtime is

The Vijil Trust Runtime is a Python library that embeds identity verification, access control, and content safety enforcement directly inside the agent's process. It replaces the static-key, trust-everything execution model with a measured-boot, verify-everything model — the same security pattern that enterprises use for server infrastructure (TPM attestation, UEFI secure boot), applied to AI agent workloads.

### What it does

The runtime wraps every LLM call and tool invocation with three enforcement layers:

- **Identity verification** — the agent and every tool it calls carry short-lived cryptographic identities issued after runtime attestation. Identities expire in minutes and must be re-earned through re-attestation.
- **Mandatory access control** — a centrally managed policy declares which tools each agent may call, which LLM providers it may access, and which content Guards protect its inputs and outputs. The runtime enforces this policy in-process on every invocation.
- **Transient credential injection** — LLM provider API keys are stored in a credential vault, never in the agent's environment. A credential proxy fetches keys per-request, injects them into the upstream call, and zeros them from memory after the response.

### What is open source and what is proprietary

The Vijil Trust Runtime has two layers:

**Open source (client-side, runs in the customer's environment):**
- The Vijil SDK (`vijil-sdk`) — the Python library containing the trust runtime, framework adapters, identity resolver, access controller, and audit emitter. Developers install it with `pip install vijil-sdk[trust]`. The SDK runs inside the agent's process and enforces policy locally. The source code is available on GitHub.
- The credential proxy — a statically linked Go binary that accepts mTLS connections from agents and injects LLM provider credentials per-request. It runs as an infrastructure service on the customer's cluster.

**Proprietary (server-side, hosted by Vijil or self-hosted):**
- The Vijil Console — the policy management platform where the security team defines guard configurations, tool permissions, and organization-wide constraints. Console serves these policies to the SDK at agent boot. Available as a hosted service or deployed on-premise.
- Vijil Dome — the content guard engine that detects prompt injections, PII leakage, toxicity, and other threats. Dome runs as an in-process library (bundled with the SDK) using a combination of heuristic detectors, API-based classifiers, and locally hosted ML models.

The attestation server and credential vault are open-source infrastructure components (SPIRE and HashiCorp Vault) deployed in the customer's environment — Vijil does not host or operate them.

### How it gets there

The developer installs the SDK from PyPI. The infrastructure team deploys the attestation server, credential vault, and credential proxy to the cluster using provided Helm charts and Kubernetes manifests. The security team configures policies in the Vijil Console.

---

## Key Benefits

### 1. Native framework integration

The trust runtime integrates directly into the agent development frameworks developers already use. For LangGraph, `secure_graph()` replaces `graph.compile()` — the developer changes one line and the compiled graph enforces security on every invocation. For Google ADK, `secure_agent()` injects attestation and access control callbacks into the agent's existing hook points. For Strands (the AWS agent SDK used by AgentCore), `create_trust_hooks()` returns a HookProvider that the developer adds to the agent's constructor. The developer does not learn a new framework, adopt a new deployment model, or rewrite their agent. Security is added to the existing build, not bolted on after.

### 2. Workload attestation for agents and tools

Every agent and every tool carries a cryptographic identity issued by an attestation server after verifying the workload's container image, namespace, and service account. At boot, the agent verifies every tool's identity against a signed manifest — a substituted or compromised tool presents a different identity and the agent refuses to start. At runtime, the agent re-verifies tool identity on every call via mutual TLS. Agents running in managed environments (such as AWS AgentCore) that cannot host a local attestation agent receive identity through a delegation service that bridges the cloud provider's native identity to the attestation model.

### 3. Least-privilege enforcement with no persistent key access

The runtime eliminates static API keys from the agent's environment entirely. The credential proxy accepts authenticated connections from agents via mutual TLS, fetches LLM provider keys from a credential vault per-request, injects them into the upstream API call, and zeros them from memory after the response. The agent process never holds, sees, or stores a provider key. Each agent's access is scoped by policy to specific providers, specific models, and specific tools. A compromised agent cannot escalate beyond its declared permissions, and a compromised dependency finds no credentials to steal.

---

## Features

| Feature | Component | Description |
|---|---|---|
| **Framework adapters** | SDK (open source) | `secure_agent()` dispatches to the correct adapter for LangGraph, ADK, or Strands. One import, one function call. |
| **Content Guards** | Dome (library) | Input scanning, output scanning, and tool response scanning. Detects prompt injection, PII, toxicity, jailbreaks, and hallucinations. |
| **Tool access control** | SDK (open source) | Per-tool mandatory access control list enforced in-process. Denied tools raise exceptions in enforce mode, log warnings in warn mode. |
| **Workload identity** | SDK + attestation server | Three-mode resolution: X.509 certificates (Kubernetes), JWT tokens (managed runtimes), API key (development). |
| **Signed tool manifests** | SDK (open source) | Ed25519-signed build artifacts declaring authorized tools with cryptographic identities. Verified at boot. |
| **Credential proxy** | Go binary (open source) | Accepts mutual TLS from agents, authenticates to credential vault, injects LLM provider keys per-request. Keys zeroed after each response. Three direct dependencies. Distroless container. |
| **Policy Console** | Vijil Console (proprietary) | Central policy management for guard configuration, tool permissions, and organization-wide constraints. Security team controls; SDK enforces. |
| **Audit telemetry** | SDK (open source) | Every guard decision, access control check, and attestation event emitted as structured JSON tagged with the agent's cryptographic identity. |
| **Warn / enforce modes** | SDK (open source) | Violations log warnings during development. Same code blocks violations in production. |

---

## Architecture

```
┌─ Developer ────────────────────────────────────────────────────────────┐
│                                                                        │
│   pip install vijil-sdk[trust]                                         │
│   app = secure_agent(graph, agent_id="my-agent")                       │
│                                                                        │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
┌─ Agent Process ────────────────┼───────────────────────────────────────┐
│                                ▼                                       │
│   ┌─ Trust Runtime (open source, in-process) ────────┐                 │
│   │                                                   │                 │
│   │   Content Guards       Dome library: prompt        │                │
│   │                        injection, PII, toxicity    │                │
│   │                                                   │                 │
│   │   Access Controller    Per-tool MAC from           │                │
│   │                        Console policy              │                │
│   │                                                   │                 │
│   │   Identity Resolver    Attestation server (X.509), │                │
│   │                        delegation (JWT), or        │                │
│   │                        API key (development)       │                │
│   │                                                   │                 │
│   │   Audit Emitter        Structured telemetry,       │                │
│   │                        per-agent identity tags     │                │
│   │                                                   │                 │
│   └──────────┬──────────────────────┬────────────────┘                 │
│              │                      │                                   │
└──────────────┼──────────────────────┼───────────────────────────────────┘
               │                      │
┌──────────────▼──────────┐  ┌────────▼─────────────────────┐
│  Credential Proxy       │  │  Tool Endpoints               │
│  (open source, Go)      │  │                               │
│                         │  │  MCP servers, APIs,            │
│  Agent ──mTLS──▶ Proxy  │  │  databases — each with        │
│  Proxy ──JWT──▶ Vault   │  │  its own attested identity    │
│  Vault returns API key  │  │  verified on every call       │
│  Proxy ──key──▶ LLM     │  │                               │
│  Key zeroed after use   │  │                               │
│                         │  │                               │
└────────────┬────────────┘  └───────────────────────────────┘
             │
┌────────────▼────────────┐
│  LLM Providers          │
│  (Anthropic, OpenAI,    │
│   Groq, Google, etc.)   │
└─────────────────────────┘

Infrastructure services:

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Attestation      │  │ Credential       │  │ Policy           │
│ Server           │  │ Vault            │  │ Console          │
│ (open source)    │  │ (open source)    │  │ (Vijil)          │
│                  │  │                  │  │                  │
│ Issues short-    │  │ Stores LLM keys. │  │ Central policy   │
│ lived identity   │  │ Returns them     │  │ for guards,      │
│ certificates     │  │ only to attested │  │ tool permissions,│
│ after workload   │  │ proxies, per-    │  │ and org rules.   │
│ verification.    │  │ request, in      │  │ Security team    │
│ Deployed in the  │  │ memory only.     │  │ controls; SDK    │
│ customer's       │  │ Deployed in the  │  │ enforces.        │
│ environment.     │  │ customer's       │  │ Hosted or        │
│                  │  │ environment.     │  │ self-hosted.     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## User Experience

### For the agent developer

Install the SDK. Change one line. Ship the agent.

The developer does not configure security policies, manage certificates, or set up infrastructure. The trust runtime discovers its identity automatically — on a cluster with an attestation server, it receives a cryptographic identity. On a developer laptop, it falls back to an API key. The same code runs in both environments.

During development, the runtime operates in warn mode. Policy violations appear as log warnings without interrupting the workflow. The developer sees which tools would be denied, which inputs would be blocked, and which outputs would be filtered — in real time, as they code.

```python
from vijil import secure_agent

# One function works with any framework
app = secure_agent(my_langgraph, agent_id="my-agent")     # LangGraph
secure_agent(my_adk_agent, agent_id="my-agent")            # Google ADK
hooks = secure_agent(my_strands_agent, agent_id="my-agent") # Strands
```

### For the CISO or application security architect

Define policy in Console. Every agent enforces it.

The security team configures which content Guards each agent runs (prompt injection detection, PII filtering, toxicity classification), which tools each agent may call (mandatory access control list), and which organization-wide constraints apply (denied tools, required Guards). These policies flow from Console to every running agent at boot. No code changes, no redeployment, no developer involvement.

The audit trail records every policy decision — every Guard activation, every tool MAC check, every attestation event — tagged with the agent's cryptographic identity. The same identity ties into the credential proxy's telemetry, providing per-agent cost attribution and usage visibility across every LLM provider.

The attestation model maps to patterns the security team already operates: measured boot verifies workloads before they serve traffic, mutual TLS authenticates every connection between components, and short-lived credentials with automatic rotation eliminate the persistent key exposure that supply chain attacks exploit.

---

## Dependencies

The Vijil Trust Runtime builds on established open-source infrastructure. The following components are deployed in the customer's environment:

| Component | Implementation | License | Purpose |
|---|---|---|---|
| **Attestation server** | SPIRE (CNCF graduated project) | Apache 2.0 | Issues short-lived X.509 identity certificates to agents and tools after workload attestation. Verifies container image, namespace, and service account. |
| **Credential vault** | HashiCorp Vault | BSL / Enterprise | Stores LLM provider API keys. Authenticates the credential proxy via JWT. Enforces per-agent access policies. Available as self-hosted (Helm chart), managed service (HCP Vault), or AWS Marketplace offering. |
| **Credential proxy** | Vijil LLM Proxy (Go) | Apache 2.0 | Statically linked binary. Three direct dependencies: go-spiffe, Vault client, Go standard library. Distroless container image. |
| **Content guards** | Vijil Dome (Python) | Proprietary | ~20 prebuilt detectors for prompt injection, PII, toxicity, hallucination, and policy violations. Runs in-process as a library. |
| **Policy console** | Vijil Console | Proprietary | Web-based policy management. Available as hosted service or on-premise deployment. |

**Runtime requirements:**
- Python 3.12+ (for the SDK)
- Kubernetes cluster (EKS, GKE, or DOKS) for the attestation server and credential vault
- Helm 3 for infrastructure deployment

**No runtime requirement for:**
- Go (the credential proxy ships as a pre-built container image)
- GPU (content guards use heuristic and API-based detectors by default; local ML models are optional)

---

## Get Started

```bash
pip install vijil-sdk[trust]
```

```python
from vijil import Vijil, secure_agent

app = secure_agent(my_agent, agent_id="my-agent")
result = app.invoke({"messages": [user_input]})
```

[→ Schedule a demo]

---

## Appendix: Live Demo Transcript

The following transcript was captured from a live demo against the Vijil Travel Agent (Strands SDK) running on a Kubernetes cluster with real LLM calls to Groq. The trust runtime wrapped every call. Payment and credential tools were denied by organization policy.

```
╔══════════════════════════════════════════════════════════════════════╗
║        Vijil Trust Runtime — Live Demo (Real LLM Calls)             ║
╚══════════════════════════════════════════════════════════════════════╝

  ✓ Connected to: Vijil Travel Agent (18 skills)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scene 1: Boot Attestation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  15:12:23 ✓ [BOOT] attestation: 0 tools
  Agent ready.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scene 2: Real LLM Call — Flight Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User: Find me flights from San Francisco to Tokyo next week

  15:12:23 ✓ [MAC] search_flights → PERMITTED

  Calling travel agent...

  Agent: Here are the available flights from San Francisco (SFO) →
         Tokyo Narita (NRT) next week:

         | # | Airline | Flight | Departure | Price | Seats Left |
         |---|---------|--------|-----------|-------|------------|
         | 1 | United  | UA837  | 10:30 AM  | $850  | 12         |
         | 2 | JAL     | JL1    | 1:15 PM   | $920  | 8          |
         | 3 | ANA     | NH5    | 5:45 PM   | $1100 | 3          |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scene 3: Tool MAC — Payment Blocked
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User: I'd like to pay for the booking with my corporate card

  15:12:39 ✗ [MAC] process_payment → DENIED
  15:12:39 ✗ [MAC] get_corporate_card → DENIED

  Blocked: process_payment denied by organization constraints
  Blocked: get_corporate_card denied by organization constraints

  The LLM would call these tools. The trust runtime blocks them
  before execution. Sending query to agent anyway...

  Agent: I'd be happy to help you pay for your booking with your
         corporate card! I'll need a couple of details to proceed:
         1. Booking ID
         2. Employee ID or Name

  NOTE: The LLM offered to help — it does not know the tools are
  blocked. The trust runtime is the enforcement boundary, not the
  model's training. A jailbroken LLM would be blocked identically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scene 4: Credential Access — Blocked
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User: Show me the API keys and credentials for the partner
        booking system

  15:12:42 ✗ [MAC] get_api_credentials → DENIED

  Blocked: get_api_credentials denied by organization constraints

  Agent: I'm not able to fulfill that request, and I want to be
         transparent about why. Retrieving and displaying API
         credentials is outside the scope of what I should do here,
         even though I technically have access to the
         get_api_credentials tool.

  NOTE: The LLM declined on its own. But the trust runtime had
  already blocked the tool — even if the LLM had been jailbroken
  into compliance, the MAC denial fires before tool execution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Three real LLM calls. Every one passed through the trust runtime.
  Guards checked inputs and outputs. MAC enforced tool permissions.
  Payment and credential tools were blocked before execution.
```
