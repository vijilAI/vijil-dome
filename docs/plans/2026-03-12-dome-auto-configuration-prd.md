# Dome Auto-Configuration — PRD

**Date:** 2026-03-12 | **Status:** In Progress | **Branch:** `vin/con-dome-auto-config-v2`

---

## Problem

The TESTED → HARDENED trust stage transition has no automated path. Today:

1. **Dome configuration is manual.** After Diamond evaluates an agent, there is no way to translate pillar scores into a Dome guard configuration. Users must understand Dome's guard taxonomy and manually author JSON configs.
2. **No runtime protection for grey/black-box agents.** White-box agents can integrate Dome via SDK, but grey/black-box agents (accessed only via API) have no code-level integration path. They need a transparent reverse proxy.
3. **Dome is not factored into trust stages.** An undefended agent can reach TRUSTED. The trust stage computation does not check whether Dome protection is active.
4. **No versioning or rollback.** Dome configs are mutable — overwriting a config is destructive with no history.

## Solution

**Manual-first, auto optional.** User reviews Diamond evaluation results, sees pillar scores, clicks "Harden with Dome" → Console recommends a config, stores it versioned, publishes to S3, and reloads the running Dome instance. Auto-hardening on eval completion is opt-in per agent (Phase 4 future work).

## Success Criteria

- A developer can go from evaluation scores to a running Dome config in one API call
- Config rollback restores previous Dome behavior within seconds
- Trust Dashboard correctly shows HARDENED stage for Dome-protected agents
- DomeLivenessDown alert fires when the Dome proxy is unreachable

## Personas

| Persona | Goal |
|---------|------|
| **Agent Developer** (grey/black-box) | One-click hardening without source code changes |
| **Risk Owner** | See trust stage reflect Dome protection status |
| **Platform Operator** | Get alerted when Dome proxy goes down |

---

## Phases & User Stories

### Phase 1: Recommend

*"Given evaluation scores, tell me what guards I need."*

| # | User Story | Status |
|---|-----------|--------|
| 1.1 | As a developer, I can submit pillar scores and get a recommended Dome config | ✅ Done |
| 1.2 | Low security score (< 0.5) produces full security suite (3 detectors) | ✅ Done |
| 1.3 | Low safety score produces moderation guard on input + output | ✅ Done |
| 1.4 | High scores produce minimal/no guards (don't over-protect) | ✅ Done |
| 1.5 | Response includes rationale explaining why each guard was chosen | ✅ Done |

**Implementation:** `POST /recommend-dome-config` (stateless preview), `src/domains/dome/recommender.py` (pure function), 7 unit tests.

---

### Phase 2: Store & Version

*"Save my config, let me roll back if something breaks."*

| # | User Story | Status |
|---|-----------|--------|
| 2.1 | As a developer, I can save a Dome config for my agent | ✅ Done |
| 2.2 | Each save creates a new version (append-only, never overwrites) | ✅ Done |
| 2.3 | I can list all versions for an agent | ✅ Done |
| 2.4 | I can rollback to any previous version | ✅ Done |
| 2.5 | Rollback creates a new version (history preserved) | ✅ Done |
| 2.6 | Config tracks its source: manual, recommended, or rollback | ✅ Done |

**Implementation:** `DomeConfigService.create_versioned_config()`, `rollback_dome_config()`, versioned repository, 3 API endpoints, 12+ unit tests.

---

### Phase 3: Publish & Reload

*"Push my config to a running Dome instance."*

| # | User Story | Status |
|---|-----------|--------|
| 3.1 | After storing config, publish to S3 for Dome to consume | ⚠️ Code exists, not wired |
| 3.2 | After publishing, trigger Dome proxy to reload config | ⚠️ Code exists, not wired |
| 3.3 | S3/reload failures are best-effort (config still saved to DB) | ✅ Done |
| 3.4 | DomeHttpClient can check Dome proxy health | ✅ Done |

**What "not wired" means:** `DomeRecommendationService` (in `src/service_dome/recommendation_service.py`) has the full pipeline — S3 publish at line 90, Dome reload at line 102 — and passes 8 unit tests. But no DI provider constructs it and no API endpoint calls it. Wiring requires: (a) a `get_dome_recommendation_service()` dependency provider, and (b) an endpoint or trigger that invokes `recommend_and_save()`.

**Implementation:** `DomeRecommendationService`, `DomeHttpClient` adapter, 15 unit tests total.

---

### Phase 4: Harden Agent (One-Click)

*"Rewrite my agent to route through Dome — no code changes."*

| # | User Story | Status |
|---|-----------|--------|
| 4.1 | `POST /agents/{id}/harden` — one API call to harden my agent | ❌ Not started |
| 4.2 | Saves original `hub_url` before rewriting to Dome proxy URL | ⚠️ Field exists, no service logic |
| 4.3 | Tracks `system_prompt_hardened` flag | ⚠️ Field exists, no service logic |
| 4.4 | Sets `protection_status`: UNPROTECTED → CONFIGURED → DOMED | ⚠️ Enum + DB column exist, no transitions |
| 4.5 | Full pipeline: fetch eval scores → recommend → store → deploy → rewrite | ❌ Not started |

**What exists:** DB columns and Pydantic model fields for `original_hub_url`, `system_prompt_hardened`, and `protection_status` (3-state enum). No service logic populates or transitions these fields.

**What's needed:** A harden-agent service that orchestrates the full flow, and an API endpoint to trigger it. This is the "one-click" user experience.

---

### Phase 5: Deploy Dome Proxy (K8s)

*"Spin up a Dome proxy pod for my agent automatically."*

| # | User Story | Status |
|---|-----------|--------|
| 5.1 | Console creates a K8s Deployment + Service for the Dome proxy | ❌ Port defined, no adapter |
| 5.2 | Proxy pod becomes Ready → agent status transitions to DOMED | ❌ Not started |
| 5.3 | Tearing down protection removes the proxy pod | ❌ Not started |
| 5.4 | Dome proxy is OpenAI-compatible reverse proxy | N/A (vijil-dome repo) |

**What exists:** `DomeOrchestrationPort` protocol in `src/domains/dome/orchestration_ports.py` defines the contract (`deploy_dome_proxy`, `update_dome_config`, `get_dome_status`, `teardown_dome_proxy`). No concrete adapter implements it.

---

### Phase 6: Trust Stage Integration

*"My dashboard reflects that my agent is hardened."*

| # | User Story | Status |
|---|-----------|--------|
| 6.1 | HARDENED stage requires active Dome + hardened system prompt | ✅ Done |
| 6.2 | TRUSTED stage requires active Dome protection | ✅ Done |
| 6.3 | Trust Dashboard shows correct stage for Dome-protected agents | ✅ Done |

**Implementation:** Updated `stage_computation.py`, 6 dome-specific trust stage tests.

---

### Phase 7: Observability

*"Alert me if Dome goes down."*

| # | User Story | Status |
|---|-----------|--------|
| 7.1 | PrometheusRule fires DomeLivenessDown when proxy is unreachable | ✅ Done |
| 7.2 | Alertmanager webhook receives firing/resolved alerts | ✅ Done |
| 7.3 | Firing DomeLivenessDown logs at WARNING level | ✅ Done |
| 7.4 | Webhook is unauthenticated (internal k8s) | ✅ Done |

**Implementation:** `k8s/mimir-rules/dome-liveness.yaml`, `POST /webhooks/alerts` endpoint, 4 unit tests.

---

### Phase 8: End-to-End Demo

*"Show me the full flow on a Kind cluster."*

| # | User Story | Status |
|---|-----------|--------|
| 8.1 | Evaluate agent → get pillar scores | N/A (existing Diamond flow) |
| 8.2 | Recommend → store → deploy → rewrite (one click) | ❌ Blocked by Phases 4+5 |
| 8.3 | Re-evaluate → improved scores | ❌ Blocked by deployed Dome proxy |
| 8.4 | Dashboard shows HARDENED stage | ⚠️ Computation works, needs real agent transition |

---

## Summary

| Phase | Stories | ✅ Done | ⚠️ Partial | ❌ Not Started |
|-------|:-------:|:------:|:---------:|:-------------:|
| 1. Recommend | 5 | 5 | — | — |
| 2. Store & Version | 6 | 6 | — | — |
| 3. Publish & Reload | 4 | 2 | 2 | — |
| 4. Harden Agent | 5 | — | 3 | 2 |
| 5. Deploy Proxy | 4 | — | — | 3 (+1 external) |
| 6. Trust Stage | 3 | 3 | — | — |
| 7. Observability | 4 | 4 | — | — |
| 8. E2E Demo | 4 | — | 1 | 3 |
| **Total** | **35** | **20** | **6** | **8** (+1 ext) |

**Phases 1, 2, 6, 7 are complete** — the foundation is solid with 55 unit tests.

**Phase 3 needs wiring** — code exists and is tested, just needs DI + trigger. Smallest lift to close.

**Phases 4, 5 are the gap** — the orchestration glue that turns building blocks into the one-click experience. Phase 5 (K8s proxy deployment) is the largest piece of new work.

**Phase 8 blocked on 4+5.**

---

## Adjacent Work (Other Branches)

| Branch | What | Relation |
|--------|------|----------|
| `vin/con-278-dome-codegen-service-orchestration` | LLM generates Dome integration code + GitHub PRs | Complementary — white-box path (code-level) vs our runtime proxy path |
| `vin/vijil-586` through `vin/vijil-597` | DomeGenes for Darwin — genome model, calibration, mutation | Builds on top of our versioned config model |
| `vin/con-273-instruction-hierarchy-hardening` | System prompt hardening | Related to our `system_prompt_hardened` field |

No conflicting work. The DomeGenes/Darwin branch will eventually consume our `DomeConfig` model and `DomeConfigService`.

---

## Out of Scope

- MCP server call interception (future phase)
- Streaming support for proxy endpoint (follow-up)
- Automatic system prompt hardening (manual or Darwin-driven for now)
- Production deployment to AWS (Kind demo only)
- Multi-provider proxy (only OpenAI-compatible `/v1/chat/completions`)
- Auto-hardening on evaluation completion (opt-in, after manual path is proven)
