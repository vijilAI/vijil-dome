# Dome Thin Client + Inference Server

**Date:** 2026-04-29
**Status:** Design
**Author:** Vin

---

## Summary

| Section | Content |
|---------|---------|
| **Overview** | Refactor Dome from a heavyweight library (2 GB with torch, transformers, presidio, litellm) into a thin client (~5 MB) that calls a unified inference server for all model-based detection. Locally executed detectors (regex, keyword, heuristic) stay in-process. Touches vijil-dome (client refactor) and vijil-inference (Ray Serve deployment). |
| **Architecture decisions** | Split detectors into local (4) and remote (11). Replace litellm with direct httpx calls. Host ML classifiers and Python detectors on Ray Serve alongside vLLM for LLM models. Single `/v1/detect` API contract for all remote detectors. |
| **Data flow** | Agent → Dome client → local detectors (in-process) + remote detectors (httpx POST to inference server) → aggregate verdicts → block/pass decision. |
| **Success criteria** | `pip install vijil-dome[lite]` has no torch/transformers/presidio/litellm. All 18 detectors produce identical verdicts. Payments agent code-mode bundle < 10 MB. |

---

## Problem

Dome's Python package bundles all detection models and their dependencies in-process. An agent importing `vijil-dome` inherits ~2 GB of transitive dependencies: torch, transformers, sentence-transformers, presidio, litellm, scipy, numpy, pandas. This makes code-mode AgentCore deploys impractical (5-10 minute cold starts from pip install), bloats container images, and forces every agent node to carry GPU-class dependencies even when detection runs on a shared inference server.

The inference server already hosts most of these models (pi-mbert, toxicity-mbert, stereotype-eeoc, prompt-harmfulness). But Dome doesn't call them — it loads its own copies locally.

## Objective

A Dome thin client that an agent can import with `pip install vijil-dome[lite]` in under 10 seconds, with all model-based detection delegated to the inference server. Local detectors (regex, keywords, secret scanning) stay in-process. The inference server runs all 11 model-based detectors behind a unified API.

## Design Principles

1. **Detection is a service, not a library.** Models run on shared GPU/CPU infrastructure, not in every agent process.
2. **Identical verdicts.** The thin client must produce the same block/pass decisions as the current monolithic Dome for the same inputs and config.
3. **Local for lightweight, remote for heavy.** Pure Python detectors (regex, keyword matching, secret patterns) stay in-process — they're fast and tiny. Anything that loads a model or calls an LLM goes remote.
4. **One API contract.** All remote detectors share the same request/response schema regardless of whether they're LLM-based, transformer classifiers, or Python analyzers.
5. **Graceful degradation.** If the inference server is unreachable, the thin client logs a warning and returns the local-only verdict. Detection is defense-in-depth, not a single point of failure.

## Architecture

```
┌──────────────────────────────────────────────┐
│  Agent Process                               │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  vijil-dome[lite] (~5 MB)             │  │
│  │                                        │  │
│  │  Local detectors (in-process):         │  │
│  │    encoding_heuristics  (stdlib)       │  │
│  │    flashtext_kw_banlist (flashtext)    │  │
│  │    secret_detector      (detect-secrets)│ │
│  │    policy_sections      (stdlib)       │  │
│  │                                        │  │
│  │  Remote dispatcher:                    │  │
│  │    httpx POST → inference server       │  │
│  │    batch multiple detectors per call   │  │
│  └──────────────┬─────────────────────────┘  │
└─────────────────┼────────────────────────────┘
                  │ POST /v1/detect
                  ▼
┌──────────────────────────────────────────────┐
│  Inference Server (EKS)                      │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  Gateway (FastAPI or Ray Serve HTTP) │    │
│  │  POST /v1/detect                     │    │
│  │  Routes by detector_name to backend  │    │
│  └──────────┬───────────────────────────┘    │
│             │                                │
│  ┌──────────▼──────────┐ ┌────────────────┐  │
│  │  vLLM (GPU)         │ │ Ray Serve (CPU)│  │
│  │                     │ │                │  │
│  │  vijil-default      │ │ pi-mbert       │  │
│  │  vijil-adversarial  │ │ toxicity-mbert │  │
│  │  gpt-oss-120b       │ │ stereotype-eeoc│  │
│  │                     │ │ prompt-harmful │  │
│  │  LLM detectors:     │ │ factcheck-rob  │  │
│  │  gpt_oss_safeguard  │ │ jb-perplexity  │  │
│  │  llm_models (gen)   │ │ pii-presidio   │  │
│  │  openai_models      │ │ hhem-halluc    │  │
│  │                     │ │ embeddings     │  │
│  └─────────────────────┘ └────────────────┘  │
└──────────────────────────────────────────────┘
```

## Detection API Contract

### Request: `POST /v1/detect`

```python
class DetectRequest(BaseModel):
    """Batch detection request — one or more detectors on the same input."""
    detectors: list[DetectorInvocation]
    input_text: str
    context_text: str | None = None  # prior conversation for context

class DetectorInvocation(BaseModel):
    """Single detector to run."""
    detector_name: str          # e.g., "pi_mbert", "gpt_oss_safeguard"
    config: dict[str, Any] = {} # detector-specific params (threshold, hub_name, etc.)
```

### Response: `DetectResponse`

```python
class DetectResponse(BaseModel):
    results: list[DetectorResult]

class DetectorResult(BaseModel):
    detector_name: str
    is_flagged: bool
    score: float               # 0.0 = safe, 1.0 = maximum risk
    category: str              # "prompt_injection", "toxicity", "pii", etc.
    details: dict[str, Any] = {}  # detector-specific metadata
    latency_ms: float
    error: str | None = None   # non-None if detector failed (graceful degradation)
```

### Design decisions for the API:

| Decision | Rationale | Revisit When |
|----------|-----------|--------------|
| Batch multiple detectors in one request | Reduces HTTP round-trips. Dome typically runs 3-5 detectors per guard pass. One POST instead of five. | If detectors have wildly different latencies and batching adds tail-latency |
| `config` dict per invocation, not global | Different detectors need different params (threshold for classifiers, hub_name for LLM, entity_types for PII). Per-invocation config is the minimal contract. | If config becomes complex enough to warrant typed models per detector |
| `score` is 0-1 float, not binary | Dome applies thresholds client-side from dome.yaml config. The server returns raw scores. This keeps threshold logic in the client where config mutations happen. | Stable |
| `error` field for graceful degradation | A failing detector returns `error` instead of crashing the entire request. Dome client can proceed with partial results. | Stable |

## Detector Classification

### Local (stay in vijil-dome client)

These detectors have no model dependencies and run in < 1ms:

| Detector | Deps | Why local |
|----------|------|-----------|
| encoding_heuristics | stdlib | Pure regex/codec checks |
| flashtext_kw_banlist | flashtext (~100 KB) | Keyword trie matching |
| secret_detector | detect-secrets (~500 KB) | Regex pattern library |
| policy_sections_detector | stdlib | Text section parsing |

### Remote (move to inference server)

| Detector | Current deps | Inference backend | Model in models.yaml |
|----------|-------------|-------------------|---------------------|
| pi_hf_mbert | torch, transformers | Ray Serve | pi-mbert (deployed) |
| pi_hf_deberta | torch, transformers | Ray Serve | pi-mbert (same model) |
| toxicity_mbert | torch, transformers | Ray Serve | toxicity-mbert (deployed) |
| toxicity_deberta | torch, transformers | Ray Serve | toxicity-scorer (deployed) |
| stereotype_eeoc | torch, transformers | Ray Serve | stereotype-eeoc-detector (deploying) |
| factcheck_roberta | torch | Ray Serve | needs new model entry |
| hhem_hallucination | torch | Ray Serve | needs new model entry |
| jb_perplexity_heuristics | torch, transformers (GPT-2) | Ray Serve | needs new model entry |
| pii_presidio | presidio, spacy | Ray Serve | new: presidio as service |
| gpt_oss_safeguard_policy | litellm | vLLM | vijil-default or gpt-oss-120b |
| llm_models | litellm | vLLM | vijil-default |
| openai_models | litellm | vLLM (or proxy) | vijil-default |
| embedding_models | sentence-transformers | Ray Serve | new: embeddings service |

## Data Flow

### Guard input (current)

```
Dome.guard_input(payload)
  → for each detector in config.input_guards:
      → load model (torch, transformers)    ← THIS IS THE PROBLEM
      → run inference locally
      → apply threshold from config
  → aggregate verdicts
  → return DomeScanResult
```

### Guard input (target)

```
Dome.guard_input(payload)
  → partition detectors into local vs. remote
  → run local detectors in-process (fast, no models)
  → batch remote detectors into one POST /v1/detect
  → httpx.post(inference_server_url, json=batch_request)
  → apply thresholds from config (client-side)
  → aggregate all verdicts
  → return DomeScanResult
```

## Implementation Plan

### Train 1: Detection API + Ray Serve (vijil-inference)

#### Task 1.1: Detection API contract
**Layer:** Domain
**Files:** Create `vijil_dome/detection_api.py` (shared Pydantic models)
**Size:** ~50 lines

Define `DetectRequest`, `DetectorInvocation`, `DetectResponse`, `DetectorResult`. These types are shared between the Dome client and the inference server. Publish as part of vijil-dome so both sides import the same models.

#### Task 1.2: Ray Serve detector deployments
**Layer:** Infra
**Files:** Create `vijil-inference/ray-serve/` with one deployment per detector
**Size:** ~400 lines (boilerplate per detector, each is ~30 lines)

Each detector is a Ray Serve deployment that:
1. Loads the model at startup (from S3 or HuggingFace)
2. Exposes a `detect(text, config) → DetectorResult` method
3. Runs on CPU (classifiers) or GPU (LLM detectors via vLLM)

The gateway aggregates deployments behind `POST /v1/detect`.

Special cases:
- **pii_presidio**: Runs presidio AnalyzerEngine + AnonymizerEngine as a Ray Serve actor
- **LLM detectors** (gpt_oss_safeguard, llm_models, openai_models): Route to vLLM via internal HTTP, format prompt, parse response into DetectorResult
- **embedding_models**: Serve sentence-transformers model, return vectors

#### Task 1.3: Deploy Ray Serve to EKS
**Layer:** Infra
**Files:** Helm chart or K8s manifests in `vijil-inference/ray-serve/`
**Size:** ~200 lines (manifests)

Deploy alongside existing vLLM pods. Ray Serve gateway on port 8000, vLLM on port 8080. Single Kubernetes Service exposes `/v1/detect` (Ray Serve) and `/v1/completions` (vLLM).

### Train 2: Dome thin client (vijil-dome)

#### Task 2.1: Remote detector dispatcher
**Layer:** Adapter
**Files:** Create `vijil_dome/detectors/remote_dispatcher.py`
**Size:** ~100 lines

`RemoteDetectorDispatcher` takes a list of detector invocations, sends a single `POST /v1/detect` to the inference server, and returns `list[DetectorResult]`. Uses httpx with configurable timeout and retry. Falls back to empty results (with logged warning) on connection failure.

Configured via `DOME_INFERENCE_URL` env var (e.g., `http://inference-server:8000`).

#### Task 2.2: Refactor detector loading to local/remote split
**Layer:** Application
**Files:** Modify `vijil_dome/dome.py` (or the guard pipeline)
**Size:** ~150 lines

The guard pipeline currently iterates detectors and runs each locally. Refactor to:
1. Classify each detector as local or remote (based on a registry)
2. Run local detectors in-process (unchanged)
3. Batch remote detectors into one `RemoteDetectorDispatcher` call
4. Merge results and apply thresholds

#### Task 2.3: Optional dependency groups in pyproject.toml
**Layer:** Infra
**Files:** Modify `pyproject.toml`
**Size:** ~20 lines

```toml
[project]
dependencies = [
    "httpx>=0.27.0",
    "pydantic>=2.0",
    "flashtext>=2.7",
    "detect-secrets>=1.4",
    # That's it for lite mode
]

[project.optional-dependencies]
full = [
    "torch>=2.0",
    "transformers>=4.40",
    "sentence-transformers>=2.0",
    "presidio-analyzer>=2.2",
    "presidio-anonymizer>=2.2",
    "litellm>=1.74",
    # ... all current deps
]
```

`pip install vijil-dome` = thin client (~5 MB)
`pip install vijil-dome[full]` = current behavior (local models, ~2 GB)

### Train 3: Integration + validation (vijil-dome + vijil-inference)

#### Task 3.1: Verdict equivalence tests
**Layer:** Test
**Files:** Create `tests/integration/test_verdict_equivalence.py`
**Size:** ~200 lines

For each of the 11 remote detectors, run the same input through:
1. The local detector (current code, `vijil-dome[full]`)
2. The remote detector (via `/v1/detect`)

Assert that `is_flagged` and `score` match within tolerance (scores may differ slightly due to float precision, but binary verdicts must match).

#### Task 3.2: Payments agent thin deploy
**Layer:** Test
**Files:** Update `vijil-sample-agents/agents/payments/` requirements
**Size:** ~20 lines

Build payments agent code-mode bundle with `vijil-dome` (lite), deploy to AgentCore, run red-swarm smoke engagement. Verify trust scores are non-zero and Dome guard decisions match expectations.

## Dependency Graph

```
Train 1 (sequential):
  1.1 API contract → 1.2 Ray Serve detectors → 1.3 EKS deploy

Train 2 (parallel with Train 1, after 1.1):
  2.1 Remote dispatcher (needs API types from 1.1)
    → 2.2 Local/remote split
    → 2.3 Optional deps

Train 3 (after Train 1 + Train 2):
  3.1 Verdict equivalence tests
  3.2 Payments agent thin deploy
```

## Key Decisions

| Decision | Rationale | Revisit When |
|----------|-----------|--------------|
| Ray Serve for ML classifiers, vLLM for LLMs | Ray Serve handles Python-native workloads (presidio, transformers pipelines) with per-model autoscaling. vLLM is optimized for autoregressive generation. Mixing them under one gateway gives best-of-both. | If Ray Serve introduces operational complexity we can't justify |
| Thresholds applied client-side | Dome config (dome.yaml) controls thresholds. The inference server returns raw scores. This keeps the mutation surface in the agent's config where Darwin can reach it. | Stable — this is fundamental to the ablation architecture |
| `vijil-dome` default = lite, opt-in = full | New installs get the thin client. Existing deployments add `[full]` to their requirements to keep current behavior during migration. | After all production agents migrate to thin client |
| Graceful degradation on server unreachable | Defense-in-depth: local detectors still run, remote results are empty with warnings. Better than crashing the agent on inference server downtime. | Stable |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Verdict divergence between local and remote | Trust score inconsistency, false sense of security | Verdict equivalence test suite (Task 3.1) runs in CI |
| Inference server latency adds to agent response time | User-visible delay on every guarded request | Batch detectors into one HTTP call; set aggressive timeout (2s); async with agent processing where possible |
| Ray Serve operational complexity | New infrastructure to maintain alongside vLLM | Start with 3 detectors (pi-mbert, toxicity-mbert, pii-presidio), validate ops, then migrate the rest |

## Out of Scope

- Streaming detection (real-time token-level guardrails) — requires a different protocol
- Custom detector plugin API for customer-defined detectors — future feature
- Model fine-tuning pipeline for detector models — handled by existing pipeline/ in vijil-inference
- Dome sidecar/proxy mode — the thin client is a library, not a service

## Success Criteria

1. `pip install vijil-dome` completes in < 10 seconds with no torch/transformers/presidio
2. All 18 detectors produce identical verdicts (equivalence test passes)
3. Payments agent code-mode ZIP < 10 MB, cold start < 60 seconds
4. Ablation study runs 38 variants with Dome enforcement active

## Related

- **Ablation study:** `vijil-darwin/docs/plans/2026-04-21-gene-sensitivity-ablation-plan.md`
- **Payments agent:** `vijil-sample-agents/agents/payments/` (vijilAI/vijil-sample-agents#18)
- **Inference server:** `vijil-inference/models.yaml`
- **Dome repo:** `vijil-dome/`
