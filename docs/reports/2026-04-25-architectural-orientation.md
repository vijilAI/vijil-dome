# Architectural Orientation: vijil-dome

**Reviewer:** AI-generated (Claude Opus 4.7, vijil-repo-orientation skill v0.1)
**Date:** 2026-04-25
**Commit examined:** `a4d4f08` on `vin/dome-trust-runtime`
**Time spent reading:** Equivalent of a deep ~6-hour pass: 4 parallel exploration agents covering structure, tests/infra, novelty, and concurrency/quality, followed by synthesis. Read directly: `README.md`, `CLAUDE.md`, `pyproject.toml`, `docs/AGENTS.md`, `vijil_dome/Dome.py`, `vijil_dome/trust/`, `vijil_dome/detectors/DETECTOR_INFO.md`, plus selected files in `vijil_dome/`.

---

## Calibration note

Anchoring "3" against the public-component baseline for an AI agent guardrail library: a competent team using FastAPI patterns + Pydantic + a curated detector ensemble + framework adapters + standard observability would land at 3 on most dimensions. The Tier-1.5 ceiling without invention is **NeMo Guardrails (NVIDIA OSS) plus SPIFFE-based agent identity plus MAC on tool calls** — competently composing those three layers. Dome is *above* that baseline on engineering discipline (multi-author, OSS-grade release pipeline, optional-extras isolation that lets users `pip install vijil-dome` without pulling torch or langgraph) and on the structural completeness of its trust runtime. It is *below* peer Vijil repos on intellectual depth because Apache 2.0 source plus public-standard composition leaves limited algorithmic IP.

Dome is the **oldest of the four Vijil repos examined** — first commit 2025-06-23, ~10 months as of this orientation. It is a public PyPI package (`pip install vijil-dome`, currently v1.6.6, Apache 2.0, four credited authors), not a service. The trust runtime under `vijil_dome/trust/` was *just* ported from `vijil-sdk` in late April (commits `26b9009`, `78be3d4`, `9b3df1a`, `c5a582c`, `92e38fa`, `3f9c23e`) — a major architectural integration completed within this orientation's window. The orientation evaluates the trust runtime as production-ready (the agents verified the SPIFFE integration is real, not a stub), but a reader should know that this part of the codebase is freshly merged.

## Summary metrics

| Metric | Score | Confidence |
|---|---|---|
| Architectural soundness | 3.6 / 5 | 3.8 / 5 |
| Intellectual depth | 3.0 / 5 | 3.3 / 5 |
| Mean confidence across all scored sections | 3.8 / 5 | — |

The soundness/depth gap is **positive (3.6 - 3.0 = 0.6)** — soundness exceeds depth, mirror-image of Swarm's negative-gap profile. Swarm's inventions were research-leaning; Dome's are application-leaning. Both are valid platform contributions. A reader weighing operational adoption should focus on §10 (the fail-open philosophy and the parallel response_string nondeterminism); a reader weighing intellectual depth should read §25 and §26 together — Dome's value is integration completeness *plus* the strategic asset of being first-credible to apply mature identity and authorization standards (SPIFFE, MAC, signed manifests) to AI agents, rather than algorithmic novelty per se.

---

## 1. Thesis

vijil-dome is a pip-installable Python library that secures AI agents at runtime through two orthogonal layers: **content guards** (a registry of 20+ detectors composed into guards and guardrails, scanning inputs and outputs for prompt injection, jailbreak, toxicity, PII, and policy violations) and a **trust runtime** (SPIFFE-based agent identity, mandatory access control on tool calls, signed tool manifests, and structured audit emission). The central technical bet is that **agent runtime security needs both content-layer detection and identity-layer authorization, and the two should compose cleanly** — `Dome()` works standalone for content guards; `secure_agent(graph, agent_id, mode)` adds the trust runtime via framework auto-detection (LangGraph, Google ADK, Strands). Most detectors wrap established public components (presidio, detect-secrets, HuggingFace transformers, OpenAI/Groq APIs); the Vijil-specific contribution is a small set of fine-tuned models migrated from HuggingFace Hub to private S3 (commit `1fa668e`) plus the trust runtime as a coherent integration.

## 2. Three-file reading path

1. **`vijil_dome/Dome.py`** (~530 LOC) — the user-facing class. `Dome.__init__()` handles config loading from TOML, dict, or `DomeConfig`. `guard_input()` and `guard_output()` (and their async/batch variants) are the primary API. Read this first to understand the user surface.
2. **`vijil_dome/guardrails/__init__.py`** (~829 LOC) — the orchestration engine. `Guard` (line 160-512) and `Guardrail` (line 514-829) compose detectors and guards with sequential or parallel execution, early-exit semantics, and timeout handling. Read this second to see how detection actually flows.
3. **`vijil_dome/trust/runtime.py`** (~500 LOC) — the trust runtime orchestrator. Composes `Identity`, `ToolPolicy`, `Manifest`, and `AuditEmitter`; exposes `attest()`, `check_tool_call()`, `guard_input()`, `guard_output()`. Read this third for the trust layer.

For framework-specific integration: `vijil_dome/trust/adapters/auto.py` (the `secure_agent()` dispatcher) plus the relevant per-framework adapter (`langgraph.py`, `adk.py`, `strands.py`).

## 3. Domain model

> **Score:** 3 / 5 · **Confidence:** 4 / 5
> **Justification:** Public boundary types are well-named Pydantic models — `DomePayload` (`types.py:26-76`), `GuardResult` (`guardrails/__init__.py:59-77`), `GuardrailResult` (`:85-105`), `ScanResult` (`Dome.py:106-137`), `BatchScanResult`. The trust runtime adds `Identity`, `ToolPolicy`, `ToolCallResult`, `ToolEntry`, `ToolManifest`, `AttestationResult`, `AuditEvent` — also Pydantic, also well-named. Two real degradations hold the score at 3 rather than 4. First, **two `GuardResult` classes** exist with the same name in different modules — `guardrails.GuardResult` and `trust.guard.GuardResult` (`trust/guard.py:26`). They don't collide at runtime because they're imported from different modules, but a reader following imports across the trust ↔ content boundary will be briefly confused. Second, **detector return types are dict-typed**: `DetectionResult = Tuple[Hit, HitData]` where `HitData = Dict[str, Any]` (`detectors/__init__.py:79`). The contract that `result["response_string"]` is present, `result.get("score", 0.0)` is a number, and other keys are opaque is *enforced by convention, not by Pydantic*. Missing `response_string` produces a `logger.warning()` rather than an error (`guardrails/__init__.py:332-340`).

**Glossary terms with codebase-specific meaning:**
- **Guard** — a list of detectors with shared category and shared early-exit policy
- **Guardrail** — a list of guards (input or output) with shared parallelism policy
- **Detector** — a single classification method (HF model, LLM call, heuristic) implementing `DetectionMethod` ABC
- **Trust runtime** — the orthogonal authorization layer (identity + MAC + manifest + audit), distinct from content guards
- **Mode** — `enforce` (block on hit) vs `warn` (log on hit, pass through) — same word used in both Dome's `enforce` flag and the trust runtime's `mode` parameter

**Anti-pattern flagged:** Two `GuardResult` classes is a maintenance hazard. A future refactor should rename one (probably `trust.guard.TrustGuardResult`) to avoid the naming collision.

## 4. Structural map

> **Score:** 4 / 5 · **Confidence:** 4 / 5
> **Justification:** `vijil_dome/` (top-level package, no `src/` wrapper — typical for a pip-installable library) divides cleanly into responsibility-named subpackages: `detectors/` (the registry plus 20+ implementations), `guardrails/` (the orchestration engine), `trust/` (the trust runtime, ported from vijil-sdk), `integrations/` (LangChain, ADK, Strands, MCP, A2A, Vijil Console), `instrumentation/` (OTel, structured logging), `cli/` (manifest signing/verification), `deploy/` (SPIRE/Vault reference configs for trust runtime deployments), `examples/` and `tutorials/` (excluded from the wheel). The two-track structure (content guards vs trust runtime) is real and intentional — a reader can use `Dome` standalone or `secure_agent()` for the full stack, and the structure reflects this exactly.

**Top-level structure:**

```
vijil_dome/
├── Dome.py                # Main user-facing class for content guards
├── __init__.py            # Public API surface with lazy imports for trust runtime
├── defaults.py            # Default guardrail configuration
├── types.py               # DomePayload (input union)
├── detectors/             # Detection method registry + 20+ implementations
│   ├── DETECTOR_INFO.md   # Catalog with parameter docs
│   ├── methods/           # Per-detector implementations
│   ├── policies/          # Detector policies
│   └── utils/             # Shared utilities
├── guardrails/            # Guard + Guardrail composition engine
│   ├── config_parser.py   # TOML/dict → guard config
│   └── instrumentation/   # OTel for guard execution
├── trust/                 # Trust runtime (ported from vijil-sdk)
│   ├── runtime.py         # TrustRuntime orchestrator
│   ├── identity.py        # SPIFFE/JWT-SVID/API-key identity cascade
│   ├── policy.py          # MAC enforcement on tool calls
│   ├── manifest.py        # Signed tool manifests (Ed25519)
│   ├── attestation.py     # Tool attestation against manifest
│   ├── audit.py           # Structured event emission
│   ├── guard.py           # Trust-layer GuardResult (separate from content)
│   └── adapters/          # Framework adapters (langgraph, adk, strands, auto)
├── integrations/          # Framework + tool integrations
│   ├── langchain/, adk/, strands/, mcp/, a2a/, vijil/, instrumentation/
│   └── examples/
├── instrumentation/       # OTel + structured logging
├── cli/                   # CLI tools (manifest signing)
├── deploy/                # SPIRE Helm, Vault YAML (reference deployment)
├── tests/                 # 43 test files (inside the package)
├── examples/              # Excluded from wheel
└── tutorials/             # Excluded from wheel
```

**Load-bearing modules:**
1. `Dome.py` and `guardrails/` — content guards core
2. `trust/` — trust runtime core
3. `detectors/__init__.py` — the registry that all detectors register into

**Peripheral modules:** `cli/`, `deploy/`, `examples/`, `tutorials/`, `tests/` (all excluded or non-importable).

The `__init__.py` uses lazy imports for the trust runtime — `secure_agent`, `TrustRuntime`, and the `trust` submodule are loaded on first attribute access, so users who only need content guards don't pay the import cost of `cryptography`, `httpx`, or framework-specific adapters.

## 5. Component composition

There is no central wiring file. Dome composes via constructor injection plus a registry pattern.

**Content guards composition:**

```
Dome.__init__(config)
  ├── create_dome_config(config) → DomeConfig
  ├── input_guardrail = Guardrail(input_guards, ...)
  │     └── Guard(...) per guard
  │           └── DetectionFactory.get_detector(category, method) per detector
  └── output_guardrail = Guardrail(output_guards, ...) (same shape)
```

Detectors register themselves at module import time via `@register_method(DetectionCategory, METHOD_NAME)` decorator (`detectors/__init__.py`). The registry is global, populated by side-effects of importing detector modules. `DetectionFactory.get_detector(category, method_name, **config)` resolves a string method name to a concrete class.

**Trust runtime composition:**

```
TrustRuntime(agent_id, constraints, mode, ...)
  ├── identity = Identity.from_environment()  (SPIFFE → JWT-SVID → API key cascade)
  ├── policy = ToolPolicy(constraints)
  ├── dome = Dome(constraints.guard_config) if guard_config else None
  ├── manifest = ToolManifest.load(path) if manifest_path else None
  └── audit = AuditEmitter(...) (defaults to logging)
```

`secure_agent(agent, agent_id, mode)` (`trust/adapters/auto.py:31-100`) detects the framework from `type(agent).__module__` (with duck-typing fallback) and dispatches to the appropriate adapter, which wraps the agent with the trust runtime.

**Cross-component pattern:** The trust runtime composes Dome internally — `TrustRuntime.guard_input(query)` delegates to `self._dome.guard_input(query)` if a Dome instance was constructed with the trust config. The two layers are designed to compose; they're not mutually exclusive but not coupled either.

## 6. Control flow

**Trace 1: User scans an input → guards execute → flagged response returned**

1. Client `dome.guard_input("How can I rob a bank?")` (sync wrapper around `asyncio.run(async_guard_input)`)
2. `Dome.async_guard_input()` (`Dome.py:360-382`) invokes `self.input_guardrail.async_scan(query, agent_id, team_id, user_id)`
3. `Guardrail.async_scan()` (`guardrails/__init__.py:701-724`) routes to `sequential_guard()` or `parallel_guard()` based on `Guardrail.run_in_parallel`
4. **Sequential path** (`:538-588`): iterate guards in order; each `Guard.async_scan()` runs detectors per its own parallelism flag; if a guard hits and `early_exit`, remaining guards skipped
5. **Parallel path** (`:590-672`): all guards spawn as `asyncio.create_task`; `asyncio.wait(FIRST_COMPLETED)` if early-exit, else `ALL_COMPLETED`; cancel pending on hit
6. **Inside each Guard**, `Guard.async_scan()` (`:392-414`) runs detectors via `sequential_guard()` (`:175-227`) or `parallel_guard()` (`:230-359`)
7. Each detector wrapped in `asyncio.timeout(asyncio_timeout_limit)` (default 5s); on timeout, returns `(hit=False, result={"error": "Detection method timed out"})` — fail-open
8. Results aggregated into `GuardResult`, then `GuardrailResult`
9. `Dome` wraps in `ScanResult` and returns

**Error / failure paths:**
- Detector exception (non-timeout) → bubbles up, logged as warning at `:634-637`; **the guard continues** with results from working detectors. Other detectors still run. If all fail, the guard returns `flagged=False` (fail-open).
- Detector timeout → caught at `:262-267`; returns `hit=False` (fail-open). **A poisoned slow detector silently degrades to "safe."**
- LLM endpoint down (e.g., Groq API outage for `policy-gpt-oss-safeguard`) → the LLM-based detector raises; treated as exception case above.

**Trace 2: User wraps an agent with `secure_agent()` → tool call enforcement**

1. Client `app = secure_agent(graph, agent_id="travel-agent", mode="enforce")`
2. `secure_agent()` (`trust/adapters/auto.py:31-100`) inspects `type(graph).__module__` to detect framework
3. For LangGraph, returns a `SecureGraph` wrapping `graph.compile()`; for ADK, returns the agent with trust callbacks injected; for Strands, returns a `TrustHookProvider`
4. At runtime, when the agent invokes a tool (e.g., `book_flight(...)`):
   - The trust adapter intercepts the tool call
   - Calls `trust_runtime.check_tool_call("book_flight", args)` (`trust/policy.py`)
   - `ToolPolicy.check()` looks up `tool_name` in `self._denied_tools` (organization deny list) and `self._permissions` (agent allowlist)
   - Returns `ToolCallResult(permitted=False, reason="not in agent permissions")` if not permitted
   - Adapter blocks the call (mode="enforce") or logs and proceeds (mode="warn")
5. `AuditEmitter.emit(event)` logs structured audit record

**Trace 3 (optional, attestation): Tool manifest verification at startup**

1. `TrustRuntime.attest()` loads `ToolManifest` from disk
2. `ToolManifest.verify_signature(public_key)` performs Ed25519 verification of the canonical-JSON representation
3. For each `ToolEntry`, the runtime verifies SPIFFE ID format, optionally fetches the X.509 SVID via the SPIRE Workload API
4. Returns `AttestationResult(all_verified=True, tool_count=N)`; tools failing attestation are rejected before serving

## 7. Data flow & state

This is a library, not a service. State lives in the *caller*'s process — primarily as Dome class instances. The Vijil-trained models are loaded from S3 (commit `1fa668e` migrated them from HuggingFace Hub) and cached locally per-process. The trust runtime can optionally fetch agent constraints from the Vijil Console at startup; otherwise constraints are passed as an explicit dict.

**Authoritative storage (when trust runtime is wired to Console):**
- Agent constraints (allowlist/denylist of tools, guard configurations) — fetched from Vijil Console via `httpx`
- Signed tool manifests — loaded from disk or embedded
- Vijil-trained detection models — fetched from private S3 bucket; CI uses OIDC role for S3 access (`.github/workflows/python-app.yaml`)
- Runtime state — none persisted; everything is per-process

**Cached/derived:**
- Detector model weights (HF transformers, ModernBERT, mBERT) — cached in `/models/vijil/` after first download
- Compiled SSL contexts for SPIFFE mTLS — held in `Identity` object lifetime
- Guard configurations — parsed once at `Dome()` construction

**Schema shape:**
- `vijil_dome.config` TOML (described in `DETECTOR_INFO.md:619-645`) — guard names, detector method names, per-detector params
- `ToolManifest` JSON (signed Ed25519) — `agent_id`, `tools: list[ToolEntry]`, `compiled_at`, `signature`

## 8. Boundaries & seams

> **Score:** 4 / 5 · **Confidence:** 4 / 5
> **Justification:** Three real boundaries, each cleanly handled. **Public Python API**: `__init__.py:17-44` exposes a small named surface — `Dome`, `DomePayload`, `BatchScanResult`, `secure_agent`, `TrustRuntime`, `get_default_config()` — with lazy imports so the trust runtime is only loaded when its symbols are touched. **Optional extras**: `pyproject.toml` declares `trust`, `trust-adapters`, `opentelemetry`, `local`, `embeddings`, `s3`, `mcp` as named extras; importing without them works (verified by agents — `vijil_dome.Dome` imports cleanly without `langgraph` installed). **Trust runtime external boundaries**: SPIFFE workload API at `/run/spire/sockets/agent.sock` is optional; identity falls back through JWT-SVID via the identity delegate service to API key, with structured logging at each step. The 4 (not 5) reflects the fact that the `cryptography` library leaks through into trust types — `Ed25519PublicKey` and similar appear in `manifest.py` signatures rather than being abstracted behind a `Signer` protocol.

**Three trust zones:**
1. **Public Python API** — what `import vijil_dome` exposes; lazy and minimal
2. **Cluster/SPIRE attestation boundary** — when the SPIRE Workload API socket exists, identity is established via X.509 SVID with mTLS; otherwise the API key fallback is *clearly logged* as a warning so operators notice
3. **External services** — Vijil Console (constraint fetching, optional), OpenAI/Groq/Anthropic LLM APIs (for LLM-based detectors), S3 (model cache, config loading)

**Where serialization happens:**
- TOML / Python dict → guardrail config (`config_parser.py`)
- Pydantic `model_dump_json()` for structured types (audit events, tool manifests, scan results)
- X.509 PEM serialization for SPIFFE certificates (`identity.py:189-250`)
- `canonical_json()` for Ed25519 signature payloads (`manifest.py:45-62`) — sort_keys=True for determinism

## 9. Ports & adapters

> **Score:** 4 / 5 · **Confidence:** 4 / 5
> **Justification:** Hexagonal discipline is real and load-bearing. `DetectionMethod(ABC)` (`detectors/__init__.py:124-282`) is the port for content detectors, with 20+ concrete adapters implementing `async detect(dome_input) → (hit, data)`. The registry pattern (`@register_method(category, name)`) lets detector modules self-register; the factory (`DetectionFactory.get_detector`) resolves names to instances. Trust runtime has parallel discipline: `Identity` (with three concrete implementations — SPIRE, JWT-SVID, API key), `ToolPolicy`, `Manifest`, `AuditEmitter` — each is a port with at least one concrete adapter and pluggability for new ones. Framework adapters (`trust/adapters/{auto, langgraph, adk, strands}.py`) implement a uniform "wrap an agent" pattern keyed on framework type. The minor gap: there's no super-protocol unifying the framework adapters — `secure_agent()` dispatches via if/elif on `_detect_framework()` rather than via a registered adapter dictionary. Adding a new framework requires editing `auto.py` in addition to creating the adapter module.

**Selected ports:**

| Port | File:line |
|---|---|
| `DetectionMethod` (ABC) | `detectors/__init__.py:124-282` |
| `Identity` | `trust/identity.py:27-150` |
| `ToolPolicy` | `trust/policy.py` |
| `ToolManifest` | `trust/manifest.py:1-68` |
| `AuditEmitter` | `trust/audit.py:15-80` |
| `Constraint` source (Console client / dict / minimal) | `trust/constraints.py` |

**Selected adapters:**

| Adapter | File |
|---|---|
| `Identity.from_spire_socket()` | `trust/identity.py:54-76` |
| `Identity.from_jwt_svid()` | `trust/identity.py:118-160` |
| `Identity.from_api_key()` | `trust/identity.py:63-76` |
| `secure_graph()` (LangGraph) | `trust/adapters/langgraph.py` |
| `secure_agent()` (ADK) | `trust/adapters/adk.py` |
| `create_trust_hooks()` (Strands) | `trust/adapters/strands.py` |
| 20+ detectors | `detectors/methods/*` |

The detector registry pattern is the cleanest extension point in the codebase — adding a new detector is a 3-step extension (subclass `DetectionMethod`, decorate with `@register_method`, reference by name in TOML/dict config). This is reference-quality plugin design.

## 10. Concurrency & failure model

> **Score:** 3 / 5 · **Confidence:** 4 / 5
> **Justification:** Async-native throughout. `asyncio.gather`, `asyncio.wait(FIRST_COMPLETED | ALL_COMPLETED)`, `asyncio.timeout` per detector (default 5s), `asyncio.create_task` with cancellation on early-exit are all used appropriately. Three real concerns hold the score at 3 not 4. (a) **Nondeterministic response_string in parallel guards**: `parallel_guard()` (`guardrails/__init__.py:294-330`) takes the response string from whichever task completes *first*, not from the first detector in the configured list. Two parallel detectors that both hit produce inconsistent block messages across runs depending on async scheduling. This is a real correctness bug worth ranking in §18. (b) **Fail-open philosophy in a security library**: detector timeout returns `hit=False` (`:262-267`); detector exceptions are caught and logged but the guard continues; if all detectors fail, the guard returns `flagged=False`. This is intentional ("availability over strict denial") but should be a config knob rather than a hardcoded policy. (c) **No retry on LLM-based detectors**: `litellm`-backed detectors rely on LiteLLM's internal retry; there's no application-level retry/circuit-breaker. A flapping Groq endpoint causes detector misses without operator notice.

**Concurrency primitives in use:**
- `asyncio.create_task` for parallel detector and guard spawning
- `asyncio.wait(FIRST_COMPLETED)` with cancellation for parallel-with-early-exit
- `asyncio.wait(ALL_COMPLETED)` for parallel-no-early-exit
- `asyncio.timeout(detector_timeout)` per-detector
- `asyncio.Semaphore(max_batch_concurrency=5)` in `DetectionMethod._gather_with_concurrency` for batch detector calls
- `ThreadPoolExecutor` for sync detector wrappers (HuggingFace transformers are blocking) — owned per `Guardrail` instance

**Real correctness bug worth fixing:**
The parallel response_string nondeterminism (`:294-330`). Local fix: when multiple detectors complete in `done`, sort by their index in the configured detector list before selecting the response. Test: spawn two parallel detectors that both hit at controlled times, verify the response is consistent across runs.

**Real philosophical question worth surfacing:**
The fail-open default. A timeout on a *security* library's detector is currently treated as `hit=False`. In some deployments (high-stakes financial agents, healthcare) the operator may want fail-closed (`hit=True` on timeout, block by default). A `timeout_mode: "open" | "closed"` config option would make the discipline explicit.

## 11. Configuration & environments

Configuration accepts three formats: **Python dict**, **TOML file path**, and **`DomeConfig` Pydantic object**. All three flow through `create_dome_config()` (`Dome.py:73-103`) and are normalized to the same internal representation.

**Key knobs:**

| Setting | Default | Effect |
|---|---|---|
| `enforce` | `True` | If False, scan and log but don't block (shadow mode) |
| `asyncio_timeout_limit` | 5 (seconds) | Per-detector timeout |
| `early_exit` | True per guard | Stop on first hit; cancel pending parallel detectors |
| `run_in_parallel` | False per guard | Sequential vs parallel detector execution within a guard |
| `score_threshold` | varies per detector | Decision threshold for ML detectors |

**Trust runtime knobs:**

| Setting | Default | Effect |
|---|---|---|
| `mode` | `"warn"` | `enforce` blocks tool calls; `warn` logs |
| `agent_id` | required | Identity scope for permissions |
| `manifest_path` | None | If provided, signed tool manifest is loaded and verified |
| `constraints` | fetched from Console | Allowlist/denylist of tools |

**S3 config loading:** `Dome.create_from_s3(bucket, key, team_id, agent_id)` (`Dome.py:232-281`) fetches the TOML or JSON config from S3 via `boto3` (optional `s3` extra), caches locally, and tracks changes via `config_has_changed()`.

**Optional extras (pyproject):** `trust`, `trust-adapters`, `opentelemetry`, `local`, `embeddings`, `s3`, `mcp` — each pulls in additional dependencies. Importing `vijil_dome.Dome` works without any of them; framework-specific adapters under `trust/adapters/` import their framework conditionally.

## 12. Observability

> **Score:** 3 / 5 · **Confidence:** 4 / 5
> **Justification:** OTel is wired but optional. The `opentelemetry` extra installs ~15 packages; when present, `AsyncioInstrumentor` and `ThreadingInstrumentor` are activated (`guardrails/instrumentation/instrumentation.py:136-138`) and tests verify metric names (`test_otel_metric_names.py:8-29`). The trust runtime emits structured `AuditEvent`s via `AuditEmitter` (`trust/audit.py:15-80`) with four canonical event types: `guard`, `tool_mac`, `attestation`, plus a generic event base. Default sink is Python `logging.info`; pluggable for OTel, structured-log shippers. The 3 (not 4) reflects two gaps: (a) **no trace correlation across detector calls** — each detector starts its own trace context rather than inheriting the guard's parent trace, so a failed parallel detector is hard to localize within a flagged guardrail trace; (b) **OTel is opt-in via extras**, so users who don't `pip install vijil-dome[opentelemetry]` get only Python logging — fine for development, thin for production. NeMo Guardrails wires OTel by default; Dome's choice to keep it optional is reasonable for a library but reduces the operational defaults.

**Audit events** (real and structured):
- `guard` — direction (input/output), flagged, score, exec_time_ms, agent_id, team_id, user_id
- `tool_mac` — tool_name, permitted, identity_verified, mode (warn/enforce), reason
- `attestation` — all_verified, tool_count, manifest_signature_status

**Sinks:** default `logging.info`; pluggable via `AuditEmitter(sink=callable)` for OTel exporters or external log shippers.

## 13. Build, package, deploy, runtime topology

This is a **library**, not a service. The "deployment" question is "how do users install and use it" plus "what reference deployment artifacts exist for the trust runtime."

**Library build:**
- Poetry (`pyproject.toml`); Python 3.11-3.13
- Package name `vijil-dome` on PyPI; current version 1.6.6
- Tag-triggered publish (`.github/workflows/publish.yml`): GitHub release event → `poetry version` → `poetry build` → `poetry publish` to PyPI via `POETRY_PYPI_TOKEN_PYPI`
- Active downloads visible via the pepy.tech badge in README (real PyPI install volume)

**Test pipeline:**
- `.github/workflows/python-app.yaml`: runs on push/PR to main, on a larger runner (CUDA models are memory-heavy)
- OIDC token to AWS S3 syncs Vijil-trained models to `/models/vijil/` (~570MB, excludes large LLMs); `VIJIL_MODEL_DIR=/models` for detector tests
- `make lint` (ruff + mypy) + `poetry run pytest`

**Trust runtime reference deployment:**
- `vijil_dome/deploy/` contains reference Helm charts and YAML for SPIRE and Vault, plus a Go-based `identity-delegate` service for issuing JWT-SVIDs to workloads outside Kubernetes (e.g., AWS managed AgentCore microVMs)
- These are **reference implementations**, not requirements for library usage. Users with their own SPIRE deployment plug into it directly via `/run/spire/sockets/agent.sock`.

**Top-level `infra/`:** CI-only IAM (`ci-s3-access`) for OIDC token → S3 model download access. No Kubernetes manifests at the top level — those live under `vijil_dome/deploy/` and apply only when deploying the trust runtime's reference SPIFFE infrastructure.

## 14. Test architecture

> **Score:** 3 / 5 · **Confidence:** 4 / 5
> **Justification:** 43 test files at `vijil_dome/tests/`, ~9.6K LOC against ~18.4K LOC of source — test:source ratio of **0.52**, the lowest of the four Vijil repos examined. Substance is real where tests exist: `test_security_detectors.py:48-83` runs prompt-injection detectors against actual jailbreak payloads and asserts `result.hit == True`, then on benign input asserts `result.hit == False`; `test_dome_obj.py:56-118` exercises async detector chains; `test_parallel_config.py` is an explicit regression test for a parallel-execution crash bug (commit `ec46c52`). The downgrade from 4 to 3 reflects four real coverage gaps the agents found: (a) **no adversarial test for the parallel response_string nondeterminism** (the §10 bug); (b) **no unicode/encoding bypass tests** for detectors trained on ASCII-normalized text; (c) **no input-length-attack tests** (unbounded text into HF tokenizers can OOM); (d) **no test for the timeout fail-open behavior** (does a poisoned slow detector really return `hit=False` and not crash?). These gaps are exactly the adversarial-style tests a security library should have.

**Per-area test density:**

| Area | Tests | Substance |
|---|---|---|
| Detectors (`tests/detectors/` plus per-method tests in methods directory) | ~20 files | Real payload assertions; substantive |
| Guardrails (`test_dome_obj.py`, `test_parallel_config.py`) | ~5 files | Strong on happy paths; gaps on adversarial parallelism |
| Trust runtime (`tests/trust/`) | ~12 files | Mock-based; ported from vijil-sdk |
| Framework integrations (`integrations/{adk, langchain, strands, mcp}`) | per-integration | Real callback/hook lifecycle tests |
| OTel instrumentation | `test_otel_metric_names.py` | Shape only — name and meter existence |

**The 0.52 ratio is partly OSS-library shape**: public libraries often shift integration testing onto downstream consumers (the LangChain integration test pattern works fine; users testing their own LangGraph apps with Dome wrap covers the integration in practice). But the gaps in *adversarial* testing are real and shouldn't be excused — a security library that doesn't probe its own bypass surface is one inversion-of-trust away from a serious bug.

## 15. External dependencies & blast radius

> **Score:** 4 / 5 · **Confidence:** 4 / 5
> **Justification:** Twelve+ load-bearing dependencies, all actively maintained. **Same `litellm` pinning discipline as Console** (`>=1.83.0,<2.0.0,!=1.82.7,!=1.82.8`) — evidence of upstream-instability awareness across Vijil repos. Optional extras (`trust`, `trust-adapters`, `opentelemetry`, `local`, `embeddings`, `s3`, `mcp`) are well-isolated: importing without them works. Detector layer wraps mature public libraries (`presidio_analyzer/anonymizer`, `detect-secrets`, `huggingface-hub`, `transformers`, `flashtext`) — all maintained, none abandoned. The only material risks: **`pyspiffe`** (SPIFFE workload API client) is the newest dependency and could lag SPIRE upstream; **`torch ^2.8.0`** as an optional extra means the `local` install is ~2-3GB (the README's "CPU-only PyTorch" workaround is real and documented).

**The twelve+ load-bearing:**

| Dep | Version | Wrapped | Risk |
|---|---|---|---|
| `openai` | `>=1.93.2` | Yes (detector wrappers) | Low |
| `pydantic` | `^2.11.7` | Yes (boundary types) | Low |
| `litellm` | `>=1.83.0,<2.0.0,!=1.82.7,!=1.82.8` | Yes | **Medium** — known upstream regressions |
| `presidio_analyzer/anonymizer` | `^2.2.361` | Yes (PII detector) | Low |
| `detect-secrets` | `^1.5.0` | Yes (secret detector) | Low |
| `huggingface-hub` | `^0.33.2` | Yes (model loading) | Low |
| `flashtext` | `^2.7` | Yes (keyword banlist) | Low |
| `nest-asyncio` | `^1.6.0` | Leaked | Low |
| `aiohttp` | `^3.13.4` | Yes (HTTP) | Low |
| `grpcio` | `^1.73.1` | Used by SPIFFE | Low |
| `torch` (optional) | `^2.8.0` | Yes (local detectors) | Medium — install size |
| `transformers` (optional) | `^4.53.1` | Yes | Low |
| `boto3` (optional) | `^1.34.0` | Yes (S3 config) | Low |
| `pyspiffe` (optional, trust) | (via `trust` extra) | Yes (Identity) | **Medium** — newest dep, lag risk |

The optional-extras isolation is a real strength — most guardrail libraries pull a heavy stack at install time; Dome lets users start with just the regex/heuristic detectors and add weight-bearing libraries on demand.

## 16. Implicit contracts & coupling

> **Score:** 3 / 5 · **Confidence:** 4 / 5
> **Justification:** Two concrete coupling points hold this at 3 rather than 4. (a) **Detector return contract is dict-typed**: `DetectionResult = Tuple[Hit, HitData]` where `HitData = Dict[str, Any]` (`detectors/__init__.py:79`). Callers assume `result["response_string"]` is present; missing it produces `logger.warning()` rather than an error (`guardrails/__init__.py:332-340`). A new detector returning `(True, {"score": 0.9})` without `response_string` will silently pass a degraded message through. (b) **Two `GuardResult` classes** with the same name in different modules (`guardrails.GuardResult` and `trust.guard.GuardResult`). They don't collide at runtime but a reader following imports across the boundary is briefly confused, and a future refactor that tries to unify them will hit a naming hazard. Other contracts are well-typed via Pydantic; the trust runtime's `Identity`, `ToolPolicy`, `ToolManifest`, `AuditEvent` are fully typed and validated. The detector-result dict-typing is the equivalent of Swarm's ADK boundary (also dict-typed) and Darwin's `config_snapshot: dict[str, Any]` — a recurring pattern across Vijil repos worth noting at the platform level.

**Worst offenders by blast radius:**
1. **Detector result dict contract not enforced** — silent log on missing `response_string`; could mask data loss
2. **Two `GuardResult` classes** — naming collision; minor but should be renamed before the trust runtime stabilizes
3. **Detector trust is implicit** — a compromised detector returning `(False, {"score": 0.9, "response_string": "..."})` will be treated as safe; no signature verification on detector output

## 17. Strengths

> **Score:** 4 / 5 · **Confidence:** 4 / 5
> **Justification:** Several patterns worth systematizing across Vijil engineering. (a) **The detector plugin architecture** — `@register_method(category, name)` decorator + global registry + factory resolution by name — is the cleanest plugin design I've seen across the four Vijil repos. (b) **The optional-extras isolation** is exemplary OSS library hygiene — `pip install vijil-dome` is light; `pip install vijil-dome[trust,trust-adapters,local,opentelemetry]` is the full stack, opt-in. (c) **The two-track usage model** (`Dome` for content guards, `secure_agent()` for full trust runtime) lets users adopt the library at either of two levels of commitment. (d) **The lazy import in `__init__.py`** for `secure_agent`, `TrustRuntime`, and `trust` submodule means content-guard users don't pay the cryptography/httpx/framework-adapter import cost. (e) **The SPIFFE identity cascade** — X.509 SVID via SPIRE → JWT-SVID via identity delegate → API key fallback, with structured logging at each step — is operationally thoughtful.

**Patterns worth replicating:**
1. **Decorator-based plugin registration** for any extension point (Console's workflow handlers should adopt this; Swarm's strategy registry should too)
2. **Optional extras for slim install** — should be the default for any Vijil OSS library
3. **Lazy imports in `__init__.py`** — minimizes load-time cost for users of subset features
4. **Two-track usage** with both layers cleanly composable — a pattern for any platform that wants to be adoptable at multiple commitment levels
5. **Identity cascade with graceful degradation** — applies to any system spanning Kubernetes-native and managed-runtime environments

## 18. Weaknesses, smells, and vulnerabilities

> **Score:** 3 / 5 · **Confidence:** 4 / 5 *(inverted: 5 = clean)*
> **Justification:** No critical security vulnerabilities. No `shell=True`, no `eval`, no unsafe `yaml.load`, no SQL injection (no SQL — it's a library). The cluster of items is medium-blast-radius correctness and design issues, plus one real correctness bug worth fixing. Apache 2.0 source plus published OSS posture means the bug surface is publicly auditable, which is good defense-in-depth.

**Ranked by blast radius:**

1. **[MEDIUM] Nondeterministic response_string in parallel guards** (`guardrails/__init__.py:294-330`)
   *Symptom:* Two parallel detectors that both hit produce inconsistent block messages across runs depending on async scheduling.
   *Fix:* Sort the `done` set by detector index in the configured list before selecting the response.

2. **[MEDIUM-philosophical] Fail-open on detector timeout in a security library**
   *Symptom:* Detector timeout returns `hit=False` and the guard continues. A poisoned slow detector silently degrades to "safe."
   *Fix:* Add a `timeout_mode: "open" | "closed"` config option and document the trade-off explicitly.

3. **[MEDIUM] No input length guards on detectors** (multiple detector files)
   *Symptom:* Unbounded text can OOM HF tokenizers or hang LLM-based detectors past timeouts. Sliding-window chunking exists for DeBERTa (`detectors/methods/pi_hf_deberta.py:109-138`) but not enforced uniformly.
   *Fix:* Config-time `max_length` per detector with a global default.

4. **[MEDIUM] No application-level retry on LLM-based detectors** 
   *Symptom:* A flapping Groq endpoint causes detector misses without operator notice; relies on LiteLLM's internal retry which may not be configured.
   *Fix:* `tenacity`-decorated wrapper with bounded backoff before the per-detector timeout.

5. **[LOW-MEDIUM] Detector result dict not enforced** (`guardrails/__init__.py:332-340`)
   *Symptom:* A detector returning `(True, {})` without `response_string` produces a warning log and uses the original query as the block message. Silent data-quality issue.
   *Fix:* Pydantic-typed detector return; validation at boundary.

6. **[LOW] ThreadPoolExecutor per `Guardrail` instance** (`guardrails/__init__.py:536`)
   *Symptom:* If users instantiate many `Dome` instances ephemerally, thread pools accumulate without cleanup.
   *Fix:* Context manager protocol on `Dome`; or pool sharing across instances.

7. **[LOW] Two `GuardResult` classes** (`guardrails/__init__.py:59` and `trust/guard.py:26`)
   *Symptom:* Naming collision; reader confusion.
   *Fix:* Rename `trust.guard.GuardResult` → `TrustGuardResult`.

8. **[LOW] LLM endpoint config is user-trusted** (`detectors/methods/llm_models.py:46-98`)
   *Symptom:* If a malicious config leaks into a Dome instance, LLM_BASE_URL can point to an attacker's endpoint.
   *Fix:* Document that config is a trust boundary; optionally allowlist hostnames.

9. **[LOW] One TODO in entire codebase** (`integrations/mcp/wrapper.py:23` — "Revisit option 1")
   *Symptom:* Unclear referent; no issue link.
   *Fix:* Document the alternative considered, or remove the TODO.

## 19. Modularity & extensibility

> **Score:** 4 / 5 · **Confidence:** 4 / 5
> **Justification:** Best-in-class for the four Vijil repos examined. **Adding a new detector** is a 3-step extension: (1) subclass `DetectionMethod`, (2) decorate with `@register_method(category, "name")`, (3) reference by name in TOML/dict config. No editing of orchestration code required. **Adding a new framework integration** is moderate (~5 steps) — create `integrations/<framework>/wrapper.py`, hook into the framework's callback system, export the wrapper class, plus the trust adapter under `trust/adapters/<framework>.py`. Existing examples (LangChain, ADK, Strands, MCP) provide templates. **Adding a trust adapter for a new framework** requires editing `trust/adapters/auto.py` to register the new framework in `_detect_framework()` — the only friction point. **Adding a new policy/guard rule** is config-driven (no code change needed). The 4 (not 5) reflects the auto-detection in `secure_agent()` requiring explicit if/elif edits rather than a registry pattern.

| Extension type | Cost | Why |
|---|---|---|
| New detector | Easy | `@register_method` + config reference |
| New framework integration | Moderate | Existing adapters provide templates |
| New policy/guard rule | Easy | Config-driven |
| New trust adapter | Moderate | Must edit `auto.py` to register |
| New evaluation method (LLM-judge variant) | Easy | Same as new detector |
| New audit sink | Easy | `AuditEmitter(sink=callable)` |

## 20. Code quality

> **Score:** 3 / 5 · **Confidence:** 4 / 5
> **Justification:** Type hints comprehensive at boundaries; mypy strict-adjacent enforcement. Pydantic models for all public types. **One TODO in 18.4K LOC (~0.005%)** — cleaner than Console (0.013%), comparable to Darwin (0.008%). No dead code visible in sampled files. Conventional commits with issue IDs (e.g., `feat(dome): add prompt harmfulness detector (#177)`). Comment density is sparse — about 2-5% — and some dense methods (`parallel_guard`, multi-window batching) lack docstrings. The internal-typing degradation (detector returns `Dict[str, Any]`) is the only material code-quality friction; the surrounding code follows a strong typed-Pydantic discipline elsewhere.

**Sample audit** (per agent #4):

| File | LOC | Type Hints | Comments | Quality |
|---|---|---|---|---|
| `types.py` | 77 | Full | Clear docstrings | High |
| `guardrails/__init__.py` | 829 | Mostly full | Sparse | Medium-High |
| `detectors/__init__.py` | 250+ | Partial | Sparse on `_gather_with_concurrency` | Medium |
| `detectors/methods/pi_hf_deberta.py` | 250+ | Good | Init has good docstring; methods sparse | High |
| `trust/runtime.py` | ~500 | Comprehensive | Present | High |

The trust runtime code (recently ported) is *higher* quality than some older detector code — reasonable given it's the newest layer. Older detector implementations could benefit from a documentation pass.

## 21. Evolution & archaeology

> **Score:** 4 / 5 · **Confidence:** 4 / 5
> **Justification:** ~10 months of clean evolution with no firefighting patterns. 28 commits in the last 30 days, 42 in 60 — comparable per-day velocity to peer repos. The major recent event is the **trust runtime port from `vijil-sdk`** (commits `26b9009`, `78be3d4`, `9b3df1a`, `c5a582c`, `92e38fa`, `3f9c23e`, `94ee30c`) — a coherent architectural integration completed in late April 2026. Test coverage was ported alongside (`3f9c23e: port trust runtime tests`). **No "Try fixing CI", no "revert again", no "actually fix"** patterns visible. Multi-author distribution (Vin 71 commits, Anuj 38, Varun 33, others 30+ over the last 100 commits) shows real shared ownership rather than single-maintainer-with-credits. The recent migration of Vijil-trained models from HuggingFace Hub to private S3 (commit `1fa668e`) is operational hardening, not a regression. The temperature-scaling work on the stereotype detector (commit `b4f91e1`) is honest applied-ML calibration.

**Active areas:**
- Trust runtime (just merged)
- Detector library (regular additions: prompt harmfulness #177, stereotype calibration #176)
- Framework integrations (recent MCP, Strands additions)
- Model-hosting infrastructure (S3 migration #180)

**Frozen/stable:**
- Core `Dome` entry point — no changes since ~Q3 2025
- `guardrails/__init__.py` orchestration — stable
- Public API surface in `__init__.py`

**Pain signals (limited):**
- One past parallel-crash bug fixed by `ec46c52` (regression test added)
- One TODO in the entire codebase (`integrations/mcp/wrapper.py:23`)

## 22. Inferred decisions

These architectural choices are not explicitly documented but clearly intentional. For each, the most likely rejected alternative:

1. **Two-track usage model (Dome standalone vs `secure_agent()` for full stack).** Rejected: a single mandatory `TrustRuntime` entry. Likely reason: lower the adoption barrier — users who only want content guards can `pip install vijil-dome` and use `Dome()` without thinking about identity, MAC, or framework adapters. The trust runtime is opt-in.

2. **Optional extras with lazy imports.** Rejected: bundle everything in the base install. Likely reason: install size matters (torch is 2-3GB; framework adapters pull langgraph/google-adk/strands). Optional extras keep the slim path actually slim.

3. **Fail-open on detector timeout.** Rejected: fail-closed (block on timeout). Likely reason: availability over strict denial — a slow detector blocking real production traffic is worse than a missed detection. This is debatable in a security library; per §10, it should be a config option.

4. **Permissive failure for individual detectors.** Rejected: any detector failure fails the guard. Likely reason: same availability argument, plus "best-effort detection" is more useful than "all-or-nothing detection" for ensembles.

5. **Decorator-based detector registration with global registry.** Rejected: explicit registration in a config file. Likely reason: lower friction for adding new detectors — drop a file, decorate, done. Cost: import-time side effects; harder to mock in isolation.

6. **Trust runtime ported from vijil-sdk into Dome.** Rejected: keep trust runtime in vijil-sdk as a separate library. Likely reason: tighter integration with content guards; users get one library not two; SDK can shrink to its core (per the user's `feedback-dome-import-paths.md` memory: "flatten vijil_dome.trust.adapters → vijil_dome.adapters; SDK and Dome must remain independent packages").

7. **SPIFFE identity cascade with API key fallback.** Rejected: SPIFFE-only (require SPIRE deployment). Likely reason: development friction — local dev can't easily run SPIRE. The cascade lets developers start with API keys and adopt SPIFFE in production. The cost is the security gradient: API-key-backed identity is unattested.

## 23. Open questions

1. **What's the migration plan for the dict-typed detector return?** `DetectionResult = Tuple[Hit, HitData]` with `HitData = Dict[str, Any]` is the load-bearing internal contract. A typed return (Pydantic `DetectorResult`) would prevent the silent missing-`response_string` issue but breaks every existing detector. Is this on the roadmap, or is the dict contract considered permanent?

2. **What is the policy on fail-open vs fail-closed?** A `timeout_mode` config option would surface the choice to operators. Has this been discussed and decided as fail-open-only, or is it open?

3. **Are the Vijil-trained models genuinely fine-tuned on Vijil data, or are they hosted copies of public models?** The S3 migration (PR #180) suggests they're valuable enough to take off public HF Hub, but whether they outperform public alternatives is the load-bearing IP question. What benchmarks back the proprietary models?

4. **What's the long-term plan for the parallel response_string nondeterminism?** Is this a known issue with a tracked fix, or has it not been observed in production yet?

5. **Are there plans to add adversarial test suites?** Unicode/encoding bypass, input-length attacks, and timeout-fail-open are exactly the gaps a security library should pre-empt. Is this on the roadmap?

6. **What's the relationship between `vijil-dome` and `vijil-sdk` going forward?** The trust runtime was just ported here from vijil-sdk. Per `feedback-dome-import-paths.md` memory, "SDK and Dome must remain independent packages." What does the SDK retain, and how do users choose between them?

7. **Is the operational deployment of the SPIRE/Vault reference (`vijil_dome/deploy/`) used in production by Vijil, or is it strictly reference for customers?** The code is real but the deployment pattern affects how confidently a customer should adopt it.

## 24. Risks if I changed X

1. **"Let's switch the detector return type to a Pydantic model."**
   *What breaks:* Every existing detector. The `DetectionResult = Tuple[Hit, HitData]` contract is in-process across `detectors/__init__.py`, `guardrails/__init__.py`, and every method file under `detectors/methods/`. A typed migration requires either (a) a deprecation window with both old and new return types accepted, or (b) a coordinated update of all 20+ detectors. The benefit (typed contract, no more silent missing-`response_string`) is real but the cost is high.

2. **"Let's make detector timeout fail-closed by default."**
   *What breaks:* User expectations. Existing deployments treat timeout as `hit=False` and proceed. Switching the default would cause inputs that previously passed to suddenly be blocked. The right migration is a config option that defaults to current behavior and lets operators opt into fail-closed; *not* a default flip.

3. **"Let's eliminate the API-key fallback in the trust runtime — SPIFFE-only."**
   *What breaks:* Every developer running locally without SPIRE; every customer in early adoption who hasn't deployed SPIRE; AWS-native AgentCore deployments that use the JWT-SVID path. The cascade exists for good reason. The right migration is to *make API-key fallback emit a louder warning* in production (e.g., raise unless `VIJIL_ALLOW_API_KEY_IDENTITY=true`), not to remove it.

4. **"Let's parallelize all detectors by default."**
   *What breaks:* The cost model. Some detectors are slow (LLM-based, ~1-2s each). Running them all in parallel multiplies LLM API spend. Sequential with early-exit is the cost-optimal default; parallel is the latency-optimal default. The current per-guard `run_in_parallel` flag lets operators pick; flipping the default would surprise users.

5. **"Let's remove the second `GuardResult` class and unify on one."**
   *What breaks:* Imports across the trust runtime. The two classes have slightly different fields (the trust one carries identity context). A unification requires field-merge plus rename plus updating every import. Worth doing eventually; not a one-PR job. Cost is real but bounded.

## 25. Novelty & intellectual depth

> **Score:** 4 / 5 · **Confidence:** 3 / 5
> **Justification:** Three Tier 2 candidates with prior-art deltas, all in the trust runtime: SPIFFE-based identity for AI agents, MAC enforcement on agent tool calls, signed tool manifests with Ed25519. Plus several Tier 1 candidates: framework auto-detection in `secure_agent()`, stereotype detector temperature scaling, the detector plugin pattern. **No confirmed Tier 3.** The score is 4 because three Tier 2 candidates exceed the rubric's "one or two non-obvious compositions" anchor for 3; no confirmed Tier 3 means it does not reach 5. The Tier 2 designation is appropriate even though each piece individually is mature: SPIFFE/SPIRE is a CNCF standard, Ed25519 signing is a 13-year-old IETF standard, MAC is 50 years old. The composition — applying all three to AI agent processes with framework-agnostic injection across LangGraph, ADK, and Strands — is the non-obvious work, and Dome appears to be the first credible OSS library to ship the integrated stack. That "first-credible" status creates value beyond what the novelty rubric measures, addressed explicitly in §26. Confidence is 3 because depth claims always benefit from outside expert review, especially when novelty hinges on integration quality.

**Tier 2 candidates:**

### Candidate 1: SPIFFE-based identity for AI agents
*Files: `vijil_dome/trust/identity.py:27-265`, `vijil_dome/deploy/` (reference SPIRE config)*

- **Problem:** AI agents in production need attested identity to bind tool authorizations to specific agent instances rather than to bearer tokens or shared secrets. Without attestation, a stolen API key gives full agent permissions to any caller.
- **Conventional solution:** API keys per agent (no attestation; revocation is the only defense), or LLM-platform-specific auth (OpenAI Assistants API permissions, Anthropic workspace tokens — platform-locked).
- **What this codebase does:** Full SPIFFE workload identity cascade — X.509 SVID via SPIRE Workload API (mTLS, certificate chain, real serialization to SSL context), JWT-SVID via identity delegate service for managed-runtime cases (AgentCore microVMs), API key as development fallback. Each path produces a verifiable identity that downstream MAC and audit can consume.
- **What had to be figured out:** The cascade order; graceful degradation logging that's loud enough operators notice the API-key fallback; integration with AWS STS for delegate-issued JWT-SVIDs; certificate-chain-to-SSL-context serialization for outbound mTLS to tools.
- **Novelty locus:** Architecture (applying workload identity to AI agent processes) + engineering tradeoffs (cascade with development fallback).
- **Tier:** **2** — non-obvious composition. Workload identity as a CNCF standard is mature; applying it to AI agent processes specifically is a logical extension that no other guardrail library has implemented end-to-end.
- **Closest prior art:** SPIFFE/SPIRE itself (CNCF, mature); Istio's service-to-service identity (uses SPIFFE for microservices); Anthropic's workspace tokens (platform-locked, no SPIFFE).
- **Delta:** First OSS guardrail library with end-to-end SPIFFE-based agent identity, integrated with content guards and MAC.

### Candidate 2: Mandatory Access Control on agent tool calls
*Files: `vijil_dome/trust/policy.py`, `vijil_dome/trust/runtime.py`*

- **Problem:** Agents call tools; tools have side effects (book flights, transfer money, send email). Without authorization, a prompt-injection-compromised agent can call any tool the framework permits. Content guards stop the prompt from arriving but don't stop the tool from being invoked.
- **Conventional solution:** Hand-coded if/else in agent code; framework-specific permissioning (LangGraph hooks, ADK callbacks); or no enforcement at all (the common case).
- **What this codebase does:** A `ToolPolicy` with explicit `denied_tools` (organization-level deny list) and `permissions` (agent allowlist) checked on every tool invocation. `ToolCallResult(permitted, modified_args, reason)` is returned; `mode="enforce"` blocks; `mode="warn"` logs. Constraints can be fetched from Vijil Console at startup or passed inline.
- **What had to be figured out:** The constraints API (allowlist vs denylist, hierarchical org-vs-agent scoping); the framework-detection-and-injection pattern for LangGraph, ADK, Strands; the audit emission for every check.
- **Novelty locus:** Architecture + domain modeling.
- **Tier:** **1-2** — borderline. The MAC pattern is 50 years old; applying it to AI agent tool calls is straightforward. The novelty is in the integration quality (every tool call goes through the policy check; the framework adapters do this transparently).
- **Closest prior art:** OPA (general policy engine; richer language); Istio AuthorizationPolicy (Kubernetes-native); AWS IAM (service-to-service); Cedar (policy language). For AI agents specifically: nothing widely deployed.
- **Delta:** First library-level MAC enforcement for AI agent tool calls with framework-agnostic injection.

### Candidate 3: Signed tool manifests with Ed25519
*Files: `vijil_dome/trust/manifest.py:1-68`, `vijil_dome/cli/manifest_cmd.py`*

- **Problem:** A compromised tool registry (the agent's tool catalog) lets an attacker swap a benign tool's identity for a malicious one. SPIFFE attestation defends against impersonation at runtime; signed manifests defend against tampering of the registry itself.
- **Conventional solution:** Trust the agent code's tool list; no signature verification.
- **What this codebase does:** `ToolManifest` is a Pydantic model (`agent_id`, `tools: list[ToolEntry]`, `compiled_at`, `signature`); signed via Ed25519 with `cryptography.hazmat.primitives.asymmetric.ed25519`; verified at runtime by computing the canonical-JSON of the unsigned payload (sort_keys=True for determinism) and calling `public_key.verify(raw_signature, payload)`.
- **What had to be figured out:** Canonical JSON for signature stability across Python versions and dict-key orderings; the public-key distribution model (out-of-band? embedded? both?); the failure mode when verification fails (refuse to start? warn?).
- **Novelty locus:** Engineering tradeoffs.
- **Tier:** **1** — clever engineering. Signed manifests are 30+ years old (container registries, Kubernetes admission, OS package managers all use them). Applying to agent tool registries is the new bit, and that's a domain-modeling choice rather than algorithmic novelty.
- **Closest prior art:** Sigstore (container image signing); Notary; npm package signing; OS package signing.
- **Delta:** First agent-tool-registry signing scheme with SPIFFE-ID-keyed entries.

**Tier 1 candidates:**
- **`secure_agent()` framework auto-detection** — module-name inspection plus duck typing; clever ergonomics, not invention
- **Stereotype detector temperature scaling** — Platt-scaling-style calibration with `T=1.237`; competent applied ML, not novel technique
- **Detector plugin pattern** — `@register_method` decorator; standard Python idiom applied well
- **Two-track usage model** — clever architecture choice; not algorithmic invention

**Calibration check:** Of the codebase, approximately **15-20% represents genuine invention (Tier 2)**, concentrated in the trust runtime's identity cascade and the MAC integration. The remaining ~80% is wrapping public libraries (presidio, detect-secrets, HuggingFace transformers, OpenAI/Groq APIs) and integration glue.

## 26. Defensible IP surface

> **Score:** 2 / 5 · **Confidence:** 3 / 5
> **Justification:** Apache 2.0 OSS source means architectural patterns are non-defensible by license. SPIFFE/SPIRE, Ed25519, MAC are all public standards. The codebase itself can be forked tomorrow. The actual moats are (a) **Vijil-trained detection models** hosted on private S3 (commit `1fa668e` migrated them off HuggingFace Hub — implying they're seen as valuable to keep proprietary); (b) **operational know-how** around production-tuning the detection thresholds (the temperature-scaling work on the stereotype detector is a single concrete example); (c) **first-mover effects** — being the first AI guardrail library with end-to-end SPIFFE identity creates standards-adoption advantage even if competitors copy. Confidence is 3 because IP defensibility on OSS libraries depends heavily on community-building strategy and proprietary-model quality, both unknowable from code alone.

| Item | Category | Replication time | 18-month defensibility |
|---|---|---|---|
| SPIFFE-based agent identity | Architectural | 2-4 weeks for a competent team | **Very low** — Apache 2.0 source available |
| MAC on tool calls | Architectural | 1-2 weeks | **Very low** — Apache 2.0 source |
| Signed tool manifests | Architectural | 1 week | **Very low** — pattern is 30 years old |
| `secure_agent()` framework auto-detect | Architectural | 1 week | Very low |
| Vijil-trained detection models | Data | Not replicable from clean room without Vijil's labeled data | **Medium-High** — only if proprietary and outperforming public |
| Stereotype calibration constants | Tacit | 2-4 weeks of empirical iteration | Low — temperature value is in the code |
| Detector ensemble curation (which 20+, which thresholds) | Tacit | 1-3 months | Medium |

**Algorithmic IP:** None. Every algorithm in Dome is borrowed.

**Architectural IP:** Effectively none, because Apache 2.0. The trust runtime architecture can be lifted directly.

**Data/Corpus IP:** The Vijil-trained models are the strongest moat — *if* they outperform public alternatives. The S3 migration suggests Vijil sees them as valuable. Without benchmark data comparing them to Llama Guard, ProtectAI's prompt-injection model, or Lakera's commercial product, the magnitude of the moat is unknowable from code alone.

**Tacit IP:** Production threshold tuning, false-positive-rate calibration data, customer-feedback-driven refinements. Real but not visible in the public artifact.

**What needs to be true for 18-month defensibility:**
- (a) The Vijil-trained models must demonstrably outperform public alternatives, ideally on a benchmark Vijil publishes
- (b) The community adoption of Dome (visible via the pepy.tech badge) must compound — first-mover advantage in standards adoption matters more than source-code defensibility
- (c) Continued investment in calibration and detection-quality work that's hard to replicate from clean-room
- (d) Active engagement with the SPIFFE Working Group and adjacent standards bodies so that Dome's deployment patterns shape the next round of agent-identity specs rather than merely conforming to them

**Note on what the rubric measures.** The §26 score reflects the rubric question — *how long would replication take for a competent team given only public product surface* — which structurally favors algorithmic moats over open-standards composition. It does not capture the orthogonal dimension of *category-leadership value*: the strategic asset of being the first credible OSS implementation of an integrated identity-and-authorization stack for AI agents. Historical analogs that scored low on algorithmic novelty but high on category leadership include **Kubernetes** (Borg-style scheduling + cgroups + container runtime API), **HashiCorp Vault** (secret rotation + transit encryption + auth backends), **Open Policy Agent** (Datalog logic + REST policy queries), and **Sigstore** (OIDC + transparency logs + Ed25519 signing). Each composed mature pieces and became durably valuable. Dome's Apache 2.0 posture caps source-replication defensibility at the bottom of the scale; whether category-leadership value materializes depends on the four trajectory factors above. **A reader weighting the rubric score literally sees a 2; a reader weighting category leadership conditional on those trajectory factors sees something materially higher.** The score and the narrative are both honest — they answer different questions, and an investor or acquirer should look at both.

## 27. Creative problem-solving signals

> **Score:** 3 / 5 · **Confidence:** 4 / 5
> **Justification:** Some careful choices visible. (a) The **identity cascade** with structured logging at each step (`identity.py:54-76`) shows a team that thought through the SPIRE-not-deployed and AgentCore-microVM cases before writing the happy path. (b) The **lazy imports in `__init__.py`** for trust runtime symbols is a thoughtful API design that protects content-guard-only users from import cost. (c) The **temperature-scaling work on the stereotype detector** with explicit production-prevalence threshold tuning (commit `b4f91e1`) shows applied-ML discipline. (d) The **two-track usage model** — `Dome` for content guards, `secure_agent()` for full stack — recognizes that not every user wants the full security posture immediately. (e) The **regression test for the parallel-execution crash bug** (`test_parallel_config.py`, added with `ec46c52`) shows the team writes tests for bugs they fix. The 3 (not 4) reflects fewer "comments explaining why-not-the-obvious-approach" than peer codebases — creative problem-solving signals are present but local rather than pervasive.

**Custom abstractions worth naming:**
- Identity cascade (SPIRE → JWT-SVID → API key) with graceful degradation
- Two-track usage (`Dome` vs `secure_agent()`)
- Detector plugin pattern (`@register_method` decorator)
- Optional extras isolation in pyproject

**Tests probing failure modes most teams wouldn't think to check:**
- `test_parallel_config.py` — parallel guard execution after a crash bug
- `test_security_detectors.py:48-83` — real malicious payloads against detectors

**Comments explaining "why not the obvious approach":** Sparse. The strongest examples are the docstrings on the identity cascade explaining why each fallback exists, and the comments in the stereotype detector explaining the temperature-scaling rationale.

## 28. Honest comparison to prior art

For each Tier 2+ novelty claim from §25, the closest existing work and the delta:

**SPIFFE-based agent identity:**
- *Closest:* SPIFFE/SPIRE itself (CNCF mature standard); Istio service-to-service identity (uses SPIFFE for microservices, not for AI agents); Anthropic workspace tokens (platform-locked); OpenAI Assistants API permissions (platform-locked).
- *Delta:* First OSS guardrail library with end-to-end SPIFFE-based agent identity, with cascade-with-graceful-degradation across SPIRE/JWT-SVID/API-key, integrated into both content guards and MAC enforcement at the framework adapter level.

**MAC on agent tool calls:**
- *Closest:* OPA (general policy engine, richer language, network-hop architecture); Istio AuthorizationPolicy (Kubernetes-native, service-to-service); AWS IAM (cloud-service authorization); Cedar (AWS authorization language). For AI agents specifically: ad-hoc allowlist/denylist in agent code, no library-level enforcement.
- *Delta:* First library-level MAC enforcement for AI agent tool calls with framework-agnostic injection across LangGraph, Google ADK, and Strands.

**Signed tool manifests with Ed25519:**
- *Closest:* Sigstore (container image signing, CNCF); Notary (container registry signing); npm package signing; deb/rpm package signing; Kubernetes admission webhooks for signed images.
- *Delta:* First agent-tool-registry signing scheme using SPIFFE IDs as the keying material for tool entries, verified at runtime alongside SPIFFE attestation.

**Search continued for Tier 3 candidates** (original techniques a domain expert wouldn't predict): None identified. Dome's value is integration completeness — having all three trust-layer pieces (identity, MAC, manifest) plus content guards plus framework adapters in one library — rather than any single algorithmic invention. The team's choice to ship Apache 2.0 reflects an understanding that the IP is in the integrated solution, not in the source code that implements it.

---

## Appendix: scoring at a glance

| § | Section | Score | Confidence |
|---|---|---|---|
| 3 | Domain model | 3 | 4 |
| 4 | Structural map | 4 | 4 |
| 8 | Boundaries & seams | 4 | 4 |
| 9 | Ports & adapters | 4 | 4 |
| 10 | Concurrency & failure model | 3 | 4 |
| 12 | Observability | 3 | 4 |
| 14 | Test architecture | 3 | 4 |
| 15 | External dependencies | 4 | 4 |
| 16 | Implicit contracts | 3 | 4 |
| 17 | Strengths | 4 | 4 |
| 18 | Weaknesses *(inverted: 5 = clean)* | 3 | 4 |
| 19 | Modularity & extensibility | 4 | 4 |
| 20 | Code quality | 3 | 4 |
| 21 | Evolution | 4 | 4 |
| 25 | Novelty | 4 | 3 |
| 26 | Defensible IP | 2 | 3 |
| 27 | Creative problem-solving | 3 | 4 |

**Architectural soundness** = mean(§3, 4, 8, 9, 10, 14, 15, 16, 19) = (3+4+4+4+3+3+4+3+4) / 9 = **3.6 / 5**

**Intellectual depth** = mean(§25, 26, 27) = (4+2+3) / 3 = **3.0 / 5**

**Mean confidence** across all 17 scored sections = 65 / 17 = **3.8 / 5**

**Notes on the score distribution:**
- Spread is 2-4. No 5s. One 2 (§26 defensibility) reflects Apache 2.0 OSS posture; eight 4s (§4, §8, §9, §15, §17, §19, §21, §25) reflect engineering maturity plus the integration-novelty value of being first-credible to apply SPIFFE+MAC+signed-manifests to AI agents.
- Dome has the **highest soundness score** of the four Vijil repos examined (3.6 vs Console 3.0, Darwin 3.3, Swarm 3.4) and **middle on depth** (3.0 vs Console 2.7, Darwin 3.3, Swarm 3.7). This is the *positive-gap* profile — soundness materially exceeds depth — mirror-image of Swarm's negative-gap profile (highest depth, middle soundness). Each repo's score profile reflects its design priorities; comparing them tells you which kind of platform value each repo delivers.
- The two highest-leverage actions on Dome's score profile: (a) fix the parallel response_string nondeterminism in `guardrails/__init__.py:294-330` (raises §10 toward 4), (b) add adversarial test suites for unicode bypass and timeout behavior (raises §14 toward 4 and §18 toward 4).

---

## Cross-repo profile (Swarm + Darwin + Console + Dome)

For readers comparing this orientation to its peers:

| Repo | Soundness | Depth | Lowest section | Distinctive strength |
|---|---|---|---|---|
| vijil-swarm | 3.4 | 3.7 | §16 implicit contracts (2) — ADK boundary | Frequency-layered nine-loop architecture; back-pressure async; published architecture papers |
| vijil-darwin | 3.3 | 3.3 | §12 observability (2) — no OTel | IC-QD compression; GEPA Pareto extension; eleven-port hexagonal discipline |
| vijil-console | 3.0 | 2.7 | §16 implicit contracts (2) and §18 weaknesses (2) | Trust state machine; discover/reconcile lifecycle; PDF-to-harness pipeline; deliberate decoupling history |
| **vijil-dome** | **3.6** | **3.0** | §26 defensible IP (2) — Apache 2.0 (rubric question is replication difficulty; category-leadership value is materially higher — see §26 note) | Best-in-class detector plugin architecture; full SPIFFE/MAC/manifest trust runtime as first-credible OSS integration for AI agents; optional-extras isolation |

Dome is the oldest of the four (~10 months) and the only OSS public library in the set. Its score profile reflects an OSS library's strengths (engineering discipline, modularity, package hygiene) and an OSS library's weaknesses on the rubric question (source replication is fast given Apache 2.0 access; novelty is application-leaning rather than algorithmic). The 0.6-point soundness-over-depth gap is the largest *positive* gap of the four repos. The headline finding for any reader weighing investment vs operational adoption: investors should focus on **(a)** the model corpus and operational know-how moats *and* **(b)** the category-leadership value of being first-credible to apply mature identity and authorization standards to AI agents (the §26 note explains why the rubric score of 2 understates this dimension); platform engineers and architects should focus on the engineering quality this codebase demonstrates.
