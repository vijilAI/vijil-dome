# Identity for AI Agents: IAM vs SPIFFE

**Date:** 2026-04-10
**Audience:** Security architects, CISOs, platform engineers evaluating workload identity for AI agent deployments
**Author:** Vijil AI

---

## The Problem

AI agents require credentials to call LLM providers, invoke tools, access data stores, and communicate with other agents. Today, most agent frameworks deliver these credentials as static API keys in environment variables or mounted secrets.

The LiteLLM supply chain compromise of March 2026 demonstrated the consequence: a credential-stealing payload in a widely-used PyPI package exfiltrated every API key, SSH key, and cloud credential from every system that installed it. The compromised package was the component that, by design, held every LLM API key in the organization.

This document compares two approaches to workload identity for AI agents: AWS IAM and SPIFFE/SPIRE. The comparison is not theoretical — both have been deployed and tested on production-grade infrastructure as part of the Vijil Trust Runtime.

---

## What Each System Provides

### AWS IAM

IAM (Identity and Access Management) assigns roles to workloads. A role carries a set of permissions expressed as IAM policies. When an agent assumes a role, it receives temporary STS credentials that grant access to the AWS resources specified in the role's policies.

IAM couples identity and authorization into a single object — the role. The role simultaneously answers "who is this?" and "what can it do?" for AWS resources.

### SPIFFE/SPIRE

SPIFFE (Secure Production Identity Framework for Everyone) provides cryptographic workload identity. SPIRE (SPIFFE Runtime Environment) implements SPIFFE by attesting workloads at runtime — verifying the container image, namespace, and service account — and issuing short-lived X.509 certificates (SVIDs) that prove the workload's identity.

SPIFFE separates identity from authorization. The SVID proves "who is this?" A separate policy layer (Vault, Console, tool MAC) answers "what can it do?"

---

## Capability Comparison

### Credential Lifecycle

| Property | IAM | SPIFFE |
|---|---|---|
| **Credential type** | STS token (bearer) | X.509 SVID (certificate + private key) |
| **Lifetime** | 1-12 hours (configurable) | 15-60 minutes (configurable) |
| **Rotation** | Automatic (SDK/Pod Identity refreshes before expiry) | Automatic (SPIRE Agent re-attests and re-issues at ~2/3 TTL) |
| **Re-verification on rotation** | No — same role, new token | Yes — workload re-attested (image hash, namespace, service account verified) |
| **Persistence** | Role assignment persists for workload lifetime | SVID expires. Re-issuance requires passing attestation. |

**Key difference:** IAM rotates the credential but not the identity verification. SPIFFE re-verifies the workload on every rotation. A tampered workload continues to receive IAM tokens but fails SPIFFE re-attestation.

### Workload Integrity

| Property | IAM | SPIFFE |
|---|---|---|
| **Binary verification** | Not verified. Any binary with the correct role gets credentials. | Verified. SPIRE checks container image hash against the registered entry. |
| **Tamper detection** | None. A modified container retains its role. | Automatic. Modified image fails attestation on next SVID rotation. |
| **Supply chain defense** | The compromised package has the same IAM role as the legitimate package. | The compromised package changes the image hash. Attestation fails. |
| **Detection latency** | Until manual investigation or external monitoring detects anomalous API calls. | Next SVID rotation (10-40 minutes, depending on TTL). |

### Authorization Granularity

| What is authorized | IAM | SPIFFE + Policy |
|---|---|---|
| AWS service access (S3, Bedrock, DynamoDB) | **Native.** IAM policies express resource-level permissions. | Not applicable — use IAM for AWS resources. |
| LLM provider access (which models) | Coarse. IAM can restrict `bedrock:InvokeModel` by model ARN. | Fine-grained. Vault policy per agent SPIFFE ID controls which providers and models. |
| Tool-level access (which tools the agent can call) | **Cannot express.** IAM operates on AWS API actions, not application-level tool functions. | Fine-grained. Console policy maps agent SPIFFE ID to permitted tool list. Enforced in-process by the trust runtime. |
| Inter-agent authorization (which agents can call which) | Possible via cross-account role assumption. Complex, fragile. | Native. Each agent has a SPIFFE ID. mTLS authenticates bilaterally. Policy authorizes. |
| Non-AWS resource access | **Cannot express.** IAM only covers AWS services. | Universal. SPIFFE identity works across clouds, on-prem, and SaaS. |

### Blast Radius After Compromise

| Scenario | IAM | SPIFFE |
|---|---|---|
| Attacker compromises agent container | Has the IAM role. Access to all AWS resources the role permits. | Has the current SVID. Access limited to SVID TTL (15-60 min). Re-attestation fails if binary changed. |
| Attacker exfiltrates credentials | STS token valid for remaining lifetime (up to 12 hours). Role automatically re-assumed — access persists. | SVID valid for remaining TTL (minutes). Cannot renew without passing attestation from the legitimate workload. |
| Attacker replaces a tool endpoint | IAM does not verify tool identity. Agent connects to the impersonated tool. | Tool must present a valid SVID matching the manifest entry. Impersonated tool has no SVID (or a different one). Connection rejected. |
| Attacker modifies agent image in registry | Next deployment uses the compromised image with the same IAM role. Full access. | SPIRE registration entry includes the expected image hash. Compromised image fails attestation. No SVID issued. |

### Operational Properties

| Property | IAM | SPIFFE |
|---|---|---|
| **Infrastructure required** | None (native to AWS) | SPIRE Server + Agent (DaemonSet on each node) |
| **Cross-cloud support** | AWS only | Any infrastructure (AWS, GCP, Azure, on-prem, edge) |
| **Standards compliance** | AWS proprietary | CNCF graduated project. Open standard. |
| **Managed runtime support** | Native (IAM role attached to microVM) | Requires OIDC bridge to convert IAM bootstrap to SPIFFE identity |
| **Developer complexity** | Low — roles assigned via Terraform/CDK | Medium — requires SPIRE deployment, workload registration |
| **Audit trail** | CloudTrail logs every STS credential use | SPIRE logs every attestation. Trust runtime logs every policy decision. |

---

## Complementary Architecture

IAM and SPIFFE are not competing systems. They operate at different layers and serve different purposes:

```
Layer 1: AWS resource access       → IAM
  "Can this agent call Bedrock? Read from S3?"
  IAM policies, native to AWS, automatic.

Layer 2: Workload identity         → SPIFFE
  "Is this agent what it claims to be? Has it been tampered with?"
  Attestation-based, cross-cloud, ephemeral.

Layer 3: Tool authorization        → Console policy + SPIFFE ID
  "Can this agent call book_flight? Is it denied charge_credit_card?"
  Application-level MAC, enforced in-process.

Layer 4: Credential management     → Vault + SPIFFE
  "Give this agent an Anthropic API key for this request only."
  Transient, per-request, attestation-gated.
```

Each layer addresses threats the others cannot:

- **IAM** prevents unauthorized AWS API calls but cannot verify workload integrity or authorize non-AWS tools.
- **SPIFFE** verifies workload integrity and provides cross-infrastructure identity but does not replace IAM for AWS resource access.
- **Console policy** expresses tool-level permissions that neither IAM nor SPIFFE can represent on their own.
- **Vault** eliminates persistent API keys by issuing transient credentials gated on SPIFFE identity.

---

## Deployment Model Implications

### EKS-hosted agents

SPIRE runs natively on EKS. The SPIRE Agent (DaemonSet) attests pods via `k8s_psat` (projected service account tokens). The CSI driver mounts the Workload API socket into agent pods. No code changes required — the trust runtime discovers the socket automatically.

IAM continues to handle AWS resource access via Pod Identity or IRSA.

### Managed AgentCore (microVM)

The managed runtime does not expose Kubernetes primitives. No DaemonSet, no CSI volumes, no pod spec customization. SPIRE Agent cannot run as a sidecar.

The solution: an OIDC bridge that converts the microVM's AWS IAM identity into a SPIFFE JWT-SVID:

```
Managed AgentCore microVM
  → AWS IAM identity token (from microVM's assigned role)
    → Vijil OIDC bridge (validates AWS token, maps to SPIFFE ID)
      → JWT-SVID: spiffe://vijil.ai/ns/prod/agent/travel-agent
        → Tool MAC, Vault access, audit — identical to EKS-hosted
```

The agent receives the same `spiffe://vijil.ai/...` identity regardless of deployment mode. Downstream systems (tools, Vault, Console) do not distinguish between the two paths.

---

## Recommendation

Use both systems at their respective layers:

1. **IAM** for AWS resource access. Do not replace it. Do not duplicate it.
2. **SPIFFE** for workload identity, tool authorization, and cross-infrastructure trust. IAM cannot provide these.
3. **Vault + SPIFFE** for transient credential management. Eliminates static API keys.
4. **Console policy + SPIFFE ID** for tool-level mandatory access control. Application-level authorization that IAM cannot express.

The investment in SPIFFE delivers three capabilities that IAM fundamentally cannot:
- Workload integrity verification via attestation
- Tool-level authorization via SPIFFE ID + policy
- Cross-infrastructure identity that works beyond AWS

For organizations deploying agents exclusively on AWS with no tool-level authorization requirements, IAM alone is sufficient. For organizations that need tool-level security, cross-cloud deployments, or defense against supply chain attacks, SPIFFE is required.

---

## References

- SPIFFE specification: https://spiffe.io/
- SPIRE (CNCF graduated): https://github.com/spiffe/spire
- LiteLLM supply chain compromise (March 2026): https://github.com/BerriAI/litellm/issues/24512
- Vijil Trust Runtime design: `docs/plans/2026-04-03-trust-runtime-design.md`
- SPIFFE/SPIRE PRD: `~/Downloads/SPIFFE-SPIRE-LLM-Proxy-PRD-Design.docx`
