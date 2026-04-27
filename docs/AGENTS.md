# AGENTS.md

This file provides guidance for AI agents (Claude Code, etc.) working with vijil-dome.

## Project Overview

**vijil-dome** is Vijil's runtime security library for AI agents. It provides two layers of protection:

1. **Content guards** — 20+ detectors for prompt injection, toxicity, PII, hallucination, stereotyping, and prompt harmfulness
2. **Trust runtime** — agent identity (SPIFFE), tool-level MAC enforcement, signed tool manifests, and structured audit

Both layers are pip-installable with no infrastructure dependency. The trust runtime optionally connects to SPIRE (identity) and the Vijil Console (constraints).

## Architecture

```
vijil_dome/
├── Dome.py                    # Content guard entry point
├── __init__.py                # Top-level: Dome, TrustRuntime, secure_agent
├── defaults.py                # Default guard configuration
├── types.py                   # DomePayload, type definitions
│
├── detectors/                 # Content detection methods
│   ├── methods/               # Detector implementations
│   │   ├── pi_hf_deberta.py   # Prompt injection (DeBERTa)
│   │   ├── pi_hf_mbert.py     # Prompt injection (mBERT)
│   │   ├── toxicity_deberta.py # Toxicity detection
│   │   ├── stereotype_eeoc.py # EEOC stereotype detection (ModernBERT)
│   │   ├── prompt_harmfulness.py # Prompt harmfulness (ModernBERT)
│   │   ├── pii_presidio.py    # PII detection (Presidio)
│   │   ├── secret_detector.py # Secret/credential detection
│   │   ├── llm_models.py      # LLM-based detection
│   │   ├── openai_models.py   # OpenAI moderation API
│   │   ├── perspective.py     # Google Perspective API
│   │   ├── hhem_hallucination.py # Hallucination detection
│   │   └── ...
│   └── utils/                 # Detector utilities
│       ├── hf_model.py        # HuggingFace model loading + S3 resolver
│       ├── llm_api_base.py    # LLM API base class
│       └── embeddings_base.py # Embedding utilities
│
├── trust/                     # Trust runtime (ported from vijil-sdk)
│   ├── runtime.py             # TrustRuntime — orchestrates all trust modules
│   ├── identity.py            # AgentIdentity — API key, SPIFFE, mTLS
│   ├── constraints.py         # AgentConstraints, ToolPermission
│   ├── policy.py              # ToolPolicy — MAC enforcement logic
│   ├── manifest.py            # ToolManifest — signed tool inventory
│   ├── guard.py               # GuardResult — structured guard outcomes
│   ├── audit.py               # AuditEmitter — structured event logging
│   ├── attestation.py         # AttestationResult — tool identity verification
│   ├── models.py              # TrustModel — Pydantic base class
│   └── adapters/              # Framework-specific secure_agent() wrappers
│       ├── auto.py            # Unified secure_agent() dispatcher
│       ├── langgraph.py       # LangGraph SecureGraph
│       ├── adk.py             # Google ADK callback injection
│       └── strands.py         # Strands HookProvider
│
├── guardrails/                # Guard configuration parsing
│   ├── config_parser.py       # TOML/dict config parsing
│   └── defaults.py            # Default guardrail configs
│
├── integrations/              # Framework integrations (content guards only)
│   ├── adk/                   # Google ADK DomeCallback
│   ├── langchain/             # LangChain DomeRunnable
│   ├── mcp/                   # MCP tool wrapping
│   ├── strands/               # Strands DomeHookProvider
│   └── vijil/                 # Vijil platform integration
│
├── cli/                       # CLI commands
│   └── manifest_cmd.py        # vijil manifest sign|verify
│
├── deploy/                    # Trust infrastructure artifacts
│   ├── llm-proxy/             # SPIFFE-authenticated LLM credential proxy (Go)
│   ├── identity-delegate/     # JWT-SVID identity delegate
│   ├── spire-*.yaml           # SPIRE Helm values and registrations
│   └── vault-*.yaml           # Vault configuration
│
└── instrumentation/           # Observability
    ├── tracing.py             # OTEL tracing
    └── logging/               # Structured logging
```

## Key Concepts

### Content guards vs Trust runtime

| Feature | `Dome` (content guards) | `TrustRuntime` / `secure_agent()` |
|---------|------------------------|-----------------------------------|
| Input/output scanning | Yes | Yes (delegates to Dome) |
| Tool-level MAC | No | Yes |
| Agent identity | `agent_id` string only | SPIFFE workload identity or API key |
| Tool identity verification | No | Yes (mTLS + signed manifests) |
| Audit logging | Via OTEL integration | Structured `AuditEvent` stream |
| Framework adapters | `DomeCallback`, `DomeRunnable` | `secure_agent()` (LangGraph, ADK, Strands) |
| Infrastructure required | None | None (optional: SPIRE, Console) |

### When to use which

- **`Dome` alone** — you want content guardrails and nothing else
- **`secure_agent()`** — you want full trust enforcement with one function call
- **`TrustRuntime` directly** — you need fine-grained control or use an unsupported framework

## Common Usage

### Content guards

```python
from vijil_dome import Dome

dome = Dome()
input_scan = dome.guard_input("How can I hack a system?")
if not input_scan.is_safe():
    return input_scan.guarded_response()
```

### Trust runtime (one line)

```python
from vijil_dome import secure_agent

app = secure_agent(graph, agent_id="travel-agent", mode="enforce")
```

### Trust runtime (direct)

```python
from vijil_dome import TrustRuntime

runtime = TrustRuntime(agent_id="my-agent", mode="enforce", constraints={...})
runtime.guard_input(query)
mac_result = runtime.check_tool_call("search_flights", {})
```

### Configuration

```python
# Content guards via dict
dome = Dome(dome_config={
    "input-guards": ["prompt-injection", "input-toxicity"],
    "output-guards": ["output-toxicity", "pii-masking"],
})

# Content guards via TOML
dome = Dome.from_toml("guardrails.toml")
```

## Detector Categories

| Category | Detectors | Purpose |
|----------|-----------|---------|
| **Security** | `prompt-injection-deberta`, `prompt-injection-mbert`, `security-llm`, `jailbreak-heuristics` | Detect adversarial inputs |
| **Moderation** | `toxicity-deberta`, `toxicity-mbert`, `stereotype-eeoc-fast`, `prompt-harmfulness-fast`, `moderations-oai-api`, `perspective-api` | Content moderation |
| **Privacy** | `pii-presidio`, `secret-detector` | PII/credential detection |
| **Integrity** | `hhem-hallucination`, `factcheck-roberta` | Factual accuracy |

## Common Commands

```bash
# Install core
pip install vijil-dome

# Install with trust runtime
pip install "vijil-dome[trust,trust-adapters]"

# Install with local models
pip install "vijil-dome[local]"

# CPU-only PyTorch
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

# Run all tests
poetry run pytest

# Run trust runtime tests
poetry run pytest vijil_dome/tests/trust/ --ignore=vijil_dome/tests/trust/adapters

# Run detector tests
poetry run pytest vijil_dome/tests/detectors/ -v

# Lint
poetry run ruff check vijil_dome/ demo/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI moderation API |
| `GROQ_API_KEY` | — | Groq API (stereotype/harmfulness safeguard mode) |
| `GOOGLE_API_KEY` | — | Perspective API |
| `ANTHROPIC_API_KEY` | — | Claude-based detection |
| `HF_TOKEN` | — | Gated HuggingFace models |
| `VIJIL_MODEL_DIR` | `/models` | Local model directory (S3-synced in production) |
| `VIJIL_CONSOLE_URL` | — | Console URL for constraints and manifest signing |
| `VIJIL_API_KEY` | — | Console API key |

## Testing Conventions

- **Detector tests**: `vijil_dome/tests/detectors/`
- **Trust core tests**: `vijil_dome/tests/trust/` (80 tests, no framework deps)
- **Trust adapter tests**: `vijil_dome/tests/trust/adapters/` (need langgraph/adk/strands installed)
- **Fixtures**: Use `@pytest.fixture` for Dome and TrustRuntime instances
- **Model tests**: Skip with `@_skip_no_*_model` when models not available locally

## Development Notes

### Adding a New Detector

1. Create detector class in `detectors/methods/`
2. Inherit from `HFBaseModel`, `LLMAPIBase`, or `DetectionMethod`
3. Register with `@register_method(DetectionCategory, METHOD_NAME)`
4. Add constant to `detectors/__init__.py`
5. Add module to `detectors/methods/__init__.py`
6. Add tests in `tests/detectors/`

### Trust Runtime Integration Patterns

- **`secure_agent()`** — one-call wrapper, detects framework automatically
- **`TrustRuntime`** — direct instantiation for custom integrations
- **`runtime.wrap_tools(tools)`** — wraps a list of callables with MAC + guards
- **`runtime.check_tool_call(name, args)`** — standalone MAC check
- **`runtime.attest()`** — verify tool identities against signed manifest

### Integration Patterns (content guards only)

- **ADK**: `DomeCallback` for before/after model hooks
- **LangChain**: `DomeRunnable` in chains
- **MCP**: `DomeMCPWrapper` for tool wrapping
- **Strands**: `DomeHookProvider` for before/after model hooks

## Relationship to Other Repos

| Repo | Relationship |
|------|-------------|
| **vijil-console** | Serves agent constraints and manifests; receives audit events |
| **vijil-diamond** | Uses Dome detectors for agent evaluation |
| **vijil-sdk** | Trust runtime originated here (ported in PR #181) |
| **vijil-travel-agent** | Demo agent using Dome + trust runtime |

## Internal Docs

| Doc | Purpose |
|-----|---------|
| `docs/trust/2026-04-03-trust-runtime-design.md` | Architecture, security model, data flow |
| `docs/trust/2026-04-03-trust-runtime-plan.md` | Implementation plan (44 commits) |
| `docs/trust/2026-04-03-trust-runtime-prd.md` | Product requirements |
| `docs/trust/2026-04-10-iam-vs-spiffe-identity.md` | Why SPIFFE over IAM |
| `docs/trust/2026-04-10-trust-runtime-gap-analysis.md` | Gap analysis vs enterprise requirements |
| `docs/trust/vijil-trust-runtime-brief.md` | Marketing tech brief |

## External Documentation

Full documentation at [docs.vijil.ai/dome](https://docs.vijil.ai/dome/intro.html)
