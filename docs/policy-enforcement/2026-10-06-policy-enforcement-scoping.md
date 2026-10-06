# Policy Enforcement via LLM Judges — Scoping

Status: **Scoping complete, ready for implementation tickets**
Branch: `design/odrl-policy-enforcement`

## 1. Goal

Console lets a team upload a policy document (privacy policy, acceptable-use policy,
compliance doc, etc.) and extracts it into a set of structured, ODRL-inspired
`PolicyRule` rows (permission / prohibition / obligation / recommendation, each with an
`action`, `target`, `constraints`, and a `consequence` of `block` / `warn` / `log` /
`flag` / `escalate` at some `severity`). Rules go through a human review workflow
(`draft` → `approved` / `rejected` / `modified`).

Today those approved rules are only consumed **offline**: a custom harness can be
generated from a policy's approved rules and evaluated after the fact. There is no
**online** path — nothing stops an agent, in production, from violating a rule the
moment it's approved.

This doc scopes what it would take to close that gap: let Dome fetch a policy's
approved rules from Console and enforce them on live traffic, in real time, using an
LLM as the compliance judge (since the rules are natural-language and not mechanically
checkable), wired into Dome's existing `AgentControl` pipeline so the same control
surface used for guardrails today (deny / steer / observe) also covers "did this turn
violate rule PRIV-003."

## 2. Why an LLM judge, not Cedar/Rego transpilation

Console's `PolicyRule` docstring already says the ODRL-inspired schema is meant "to
serve as an intermediate representation for downstream transpilation to Cedar/Rego for
runtime policy evaluation" (`src/domains/policies/models.py` in vijil-console). We
looked for that compiler and it doesn't exist — and for good reason. Rules are
extracted by an LLM from free-text policy documents (`natural_language` is the only
field guaranteed to be faithful; `action`/`target`/`constraints` are the LLM's
best-effort structuring of that text). Compiling "don't share a user's health
information with third parties without consent" into a sound Cedar/Rego constraint is
either lossy or requires a human to hand-author the formal version — which is exactly
what Console's *separate* `dome_spec` feature is for (see §3.3). Trying to resurrect an
ODRL→Cedar/Rego compiler for LLM-extracted rules was already looked at and
deprioritized elsewhere in this repo as "lossy ... shouldn't gate the demo"
(`docs/trust/2026-06-04-tamper-evident-identity-mac-plan.md:25`).

The pragmatic path is the one Dome already has infrastructure for: ask an LLM "does
this turn violate this rule" per rule, with the rule's natural-language text and
structured metadata as context, and treat the LLM's verdict as the sensor reading.

## 3. What already exists (confirmed by reading both repos, not assumed)

### 3.1 Console: policy rules

- `PolicyRule` (`src/domains/policies/models.py`): `rule_id` (human string like
  `PRIV-001`), `rule_type` (permission/prohibition/obligation/recommendation),
  `action`, `target`, `assignee`, `constraints` + `constraint_logic`, `consequence`
  (`{action: block|warn|log|flag|escalate, severity: info|low|medium|high|critical,
  message}`), `natural_language`, `category`, `status`.
- `GET /v1/policies/{policy_id}/rules?status=approved&limit=&offset=`
  (`src/service_agent_environment/api/policies.py:464`) — exactly the endpoint a Dome
  fetcher would poll. Already consumed client-side by `vijil-sdk`'s
  `policies.rules()`.
- `GET/PATCH/DELETE /v1/rules/{rule_id}`, `POST /v1/rules/{rule_id}/{approve,reject}`
  (`src/service_agent_environment/api/rules.py`) for single-rule lifecycle.
- Service-to-service auth: `(client_id, client_secret)` → `POST /v1/auth/token`
  (`src/service_teams/api/auth.py:511`) — the same client-credentials exchange already
  used elsewhere for machine callers (SDK, CI, and Console's own `dome_spec` →
  Dome-side fetch design).

### 3.2 Dome: AgentControl spec — the thing we'd plug into

- `Control` / `ConditionNode` / `EvaluatorRef` / `ControlAction` (`vijil_dome/controls/models.py`):
  a `Control` has a `scope` (which steps it applies to), a `condition` (a boolean tree
  of evaluator leaves), and an `action` (`decision: deny|steer|observe`,
  `on_error: fail_open|fail_closed`).
- Evaluators are pluggable: `@register_evaluator("name")` on an `Evaluator` subclass
  (`async def evaluate(value, config) -> EvaluatorResult`), resolved by
  `controls/evaluators/__init__.py`. Critically, **any existing Dome `DetectionMethod`
  is already exposed to this layer for free** via the `dome:` prefix bridge
  (`DomeBridgeEvaluator`, `controls/evaluators/dome_bridge.py`) — register something as
  a Dome detector and the control schema can reference it as `dome:<method-name>`
  immediately, no changes to `controls/models.py` or the engine required.

### 3.3 Dome: `trust/` module — related but not this

`vijil_dome/trust/` (policy.py, constraints.py, manifest.py, runtime.py) is a
**separate, already-shipping system**: deterministic MAC-style tool-call permissions
keyed on SPIFFE identity + Ed25519-signed manifests. It calls into Dome's existing
`guard_input`/`guard_output` for *content* guards via `AgentConstraints.dome_config`,
but has no rule-engine or LLM-judge concept of its own — it's a consumer of Dome's
guard system, not an alternative to it. Relevant because it already shows the pattern
for wiring a new Dome detector into a higher-level "constraints for this agent" object
(`dome_config`), which we'd want this new policy-judge control to also be reachable
from, but it is not where this work lives.

Also worth noting: Console has a **separate, already-built `dome_spec` feature**
(`src/domains/dome_spec/` in vijil-console, currently uncommitted on
`fix/frameworks-trust-report-type`) for uploading a hand-authored, pre-compiled Dome
control YAML (controls + sensors + norms, with Cedar/Rego inlined) and attaching it
1:1 to a Policy. That's the path for policies that already have a formal,
machine-checkable representation. This scoping doc is about the *other* case — the
common one — where the policy only exists as LLM-extracted natural-language
`PolicyRule` rows with no formal representation, and the "compiler" is an LLM judge
instead of Cedar/Rego.

### 3.4 Dome: an LLM-judge-over-free-text-policy detector basically already exists

- `PolicyGptOssSafeguard` (`vijil_dome/detectors/methods/gpt_oss_safeguard_policy.py`),
  registered as Dome `DetectionMethod` `"policy-gpt-oss-safeguard"`. Takes free-text
  policy content, builds a hardened classifier system prompt, calls
  `openai/gpt-oss-safeguard-20b` or `-120b` via litellm (default hub **groq** — the same
  model family Console already defaults to for rule extraction,
  `VIJIL_RULES_MODEL=groq/openai/gpt-oss-120b`), at `temperature=0`. Has three output
  modes, the richest of which (`with_rationale`) already returns
  `{"violation", "policy_category", "rule_ids", "confidence", "rationale"}` — i.e. the
  shape we want, just not yet tied to Console's actual `rule_id`s.
- `PolicySectionsDetector` (`vijil_dome/detectors/methods/policy_sections_detector.py`)
  chunks a long policy into sections, runs one judge per section in parallel with
  fast-fail on first violation, with optional FAISS top-k retrieval to only check
  relevant sections against the current turn.
- `LlmBaseDetector` (`vijil_dome/detectors/utils/llm_api_base.py`) is the reusable
  provider abstraction (groq/openai/together hubs via litellm) both of the above sit
  on.

This means the net-new judge detector is a **specialization**, not new infrastructure:
swap "one blob of policy text" for "N structured `PolicyRule` objects fetched from
Console," and swap the section-chunking batcher for a rule-batcher.

### 3.5 Judge model: Jev / TypeSafe System One

User-specified model, via OpenRouter: `typesafe/jev-1.13`
(https://openrouter.ai/typesafe/jev-1.13), docs at
https://docs.typesafe.ai/concepts/system-one. `OPENROUTER_API_KEY` goes in Dome's
`.env`, same convention as `GROQ_API_KEY`/`TOGETHERAI_API_KEY` today.

This is **not** a general chat model — it's TypeSafe's "System One" decision layer,
which returns typed, calibrated decisions instead of generated text, via three
primitives: **Choice** (pick from options), **Score** (numeric scale), **Noul**
(boolean probability). The intended pattern is: build a "state" with context, pose
multiple independent questions against it, get back typed answers with calibrated
confidence. That maps onto our problem almost exactly — "did this turn violate rule
PRIV-003" is a **Noul** question, one per approved rule, posed together against the
same turn-as-state in a single call, with the rule's `severity`/confidence handled as
a **Score**. That's a cleaner fit than `PolicyGptOssSafeguard`'s current approach
(prompt-engineered JSON + regex-fallback parsing of a chat completion) for exactly the
"N structured rules → N verdicts" shape this feature needs.

**Wire format — confirmed, not a chat-completions call**:

- TypeSafe's native API: `POST https://api.typesafe.ai/v1/systemone`,
  `Authorization: Bearer <TYPESAFE_API_KEY>`, body
  `{"state": "<turn text>", "model": "jev-latest", "questions": {"<rule_id>":
  {"type": "noul", "instructions": "<rule.natural_language, phrased as a yes/no
  violation check>"}}}`, response
  `{"model": "jev-1.13.0", "answers": {"<rule_id>": {"type": "noul", "confidence":
  0.78, "probabilities": {...}}}, "usage": {...}}` — i.e. the per-rule verdict keying
  we want falls out of the request/response shape for free (`questions`/`answers`
  are keyed dicts, so `rule_id` IS the question key, no extra mapping needed).
- OpenRouter hosts this **as the same typed API**, not chat-completions-normalized:
  `POST https://openrouter.ai/api/v1/systemone` (TypeSafe-SDK-compatible — "switch
  the base URL only") or `POST https://openrouter.ai/api/alpha/decisions` (OpenRouter's
  own wrapper), both billed to the OpenRouter account, model id `typesafe/jev-1.13`
  (or `~typesafe/jev-latest`). Context window 32k tokens; output tokens are free
  (~$0.042/M input tokens) — a real constraint on how many rules fit in one batched
  call, same batching-by-rule-count concern noted for `PolicySectionsDetector`.
- Confirms the recommendation to **not** force this through `LlmBaseDetector`'s
  litellm/chat-completions path (`supported_hubs = [None, "openai", "together",
  "groq"]` — no concept of typed primitives, and "openrouter" isn't even in that list).
  A small dedicated `SystemOneClient` (one `httpx` POST per batch, `OPENROUTER_API_KEY`
  from `.env`) replaces `LlmBaseDetector` for this one detector — no JSON-parsing
  fallback regex needed at all, unlike `PolicyGptOssSafeguard`.

### 3.6 Console's existing agent-scoped `DomeConfig` — the real reuse target

A steer from review: don't invent a new hosted artifact at all — Console already has
a per-agent config/delivery system (`src/domains/dome/` — distinct from `dome_spec`,
§3.7) that's a much better fit to build on. Confirmed by reading it directly:

- `DomeConfig` (`src/domains/dome/models.py:25-60`): `id`, `team_id`, `agent_id`
  (nullable — a config can be unbound or bound to one agent), `status`
  (`pending`/`active`/`archived`), `config_body: dict[str, Any]` (**untyped** — no
  Pydantic schema enforced server-side today), `enforcement_mode` (`warn`/`enforce`).
- **Lifecycle already matches what we need**: create writes `config_pending.json` to
  S3; a separate `apply()` call (`src/service_dome/api/dome_config_router.py:286-326`)
  promotes pending → active (`config.json`), archiving whatever was active before.
  This is *exactly* "a proposed spec a human can review and override before it's
  live" — no new workflow concept needed, just a new way to populate the pending file.
- **Delivery is S3-native, not HTTP** — and already has freshness-checking built in,
  which resolves the "hash check" ask more completely than my first pass's idea of a
  new HTTP ETag endpoint did: the running Dome process (`Dome.create_from_s3` in
  `vijil_dome/Dome.py:250-295`, via `vijil_dome/utils/config_loader.py`) reads
  `teams/{team_id}/agents/{agent_id}/dome/config.json` straight from S3 with local-disk
  TTL caching (default 300s) *and* an S3-object-ETag-based `config_has_changed()`
  check (`Dome.py:300-321`) before re-parsing. Console's `/dome-configs` REST API is
  only used by Console's own UI, never called by the runtime.
- `config_body`'s current shape is the **older plain guardrail format**
  (`src/domains/dome/default_config.py`: `input-guards`/`output-guards` lists of
  `{type, methods}` blocks) — parsed by `vijil_dome/guardrails/config_parser.py` into
  the *old* `Dome` class (`Dome.py`), not the newer Controls-first `VijilDome` class
  (`vijil_dome/core.py`) this feature wires `PolicyRuleJudge` through (§3.2). These are
  two different runtime classes in vijil-dome today. `VijilDome(policy=...)` currently
  only accepts a `list[dict | Control]`, a file path, or `None` — **no S3-aware
  constructor exists yet** (unlike the old `Dome` class).

### 3.7 Three separate systems, and the links that don't exist yet

Confirmed: `dome` (agent-scoped runtime config, §3.6), `dome_spec` (policy-scoped
hand-authored Cedar/Rego, §3.3), and now this feature's compiled judge-controls are
three genuinely independent things today — zero cross-references between `dome` and
`dome_spec` in either direction (no shared FK, no shared code path). Two concrete gaps
that block "an agent enforces a policy":

- **No agent↔policy link exists.** `src/domains/agents/models.py` has
  `dome_config_id` (which `DomeConfig` to use) but no `policy_id`/`dome_spec_id`
  field anywhere. Something needs to record "config X was generated from policy Y" —
  simplest fix is a new field on `DomeConfig` itself (e.g.
  `source_policy_ids: list[UUID] | None`), not a general agent↔policy relationship.
- **Evaluation data doesn't speak the same vocabulary as policy rules.** Diamond's raw
  results are untyped (`EvaluationResults.harness_results: dict[str, Any]`); the
  structured version lives in the *reports* domain (`src/domains/reports/models.py`:
  `ProbeResult`, `Finding` — keyed by `harness_id`/`code`/`severity`). `PolicyRule` is
  keyed by `category`/`action`/`target`. **No mapping between these two vocabularies
  exists anywhere in the codebase.** True "which approved rules does this agent's
  evaluation history show it's already failing on" gap analysis needs that mapping
  built first — it's a real, separate piece of work, not a wiring task. See §5A for
  how this doc proposes sequencing around that gap.

## 4. Proposed architecture

Revised again from a second steer: generate the proposed judge-controls as part of
`DomeConfig.config_body` (§3.6) instead of a brand-new hosted artifact. This reuses
Console's existing per-agent pending→apply→active review workflow *and* the S3-native
TTL/ETag freshness check already built into the Dome client — both pieces of
machinery this doc's first two passes were about to re-invent.

```mermaid
flowchart LR
    subgraph Console
        PR[PolicyRule rows<br/>status=approved]
        Eval[Agent's evaluation/report findings<br/>optional input, see 5A]
        Compiler["PolicyGapCompiler (new)<br/>agent_id + policy_id [+ findings]<br/>-&gt; proposed controls list"]
        Pending["DomeConfig.config_body (existing)<br/>config_pending.json in S3<br/>new 'controls' key alongside<br/>legacy input-guards/output-guards"]
        PR --> Compiler
        Eval -.optional.-> Compiler
        Compiler -->|writes pending, existing create() path| Pending
        Pending -->|human reviews/edits, then apply&#40;&#41;| Active["config.json (active)"]
    end

    subgraph Dome
        Loader["VijilDome S3-aware constructor (new)<br/>reuses config_loader.py's existing<br/>TTL + S3-ETag check, unchanged"]
        Judge["PolicyRuleJudge (Jev/System One)<br/>new DetectionMethod"]
        Controls["Controls parsed from config_body['controls']<br/>bucketed by consequence.action,<br/>each condition: dome:policy-rule-judge"]
        Engine[ControlEngine]
        Active -->|S3 read, cached| Loader
        Loader -->|parsed controls incl. embedded rule data| Controls
        Controls -->|evaluator calls| Judge
        Controls --> Engine
    end

    Turn[Agent turn<br/>input/output/tool-call] --> Engine
    Engine -->|deny / steer / observe| Decision[ControlAction]
    Decision --> Audit[dome-control OTel span: rule_id, policy_id, severity]
```

## 5. Net-new work

### A. Console: `PolicyGapCompiler` — proposes controls into the existing `DomeConfig` pending slot (new)

A new Console-side service, **run on explicit trigger** (a button on the
policy/agent page — not an automatic recompile on every rule change; keeps the
"stale pending" risk at zero since nothing touches `active` until a human calls
`apply()` anyway). Takes `(agent_id, policy_id)`, reads the policy's approved
`PolicyRule` rows, and writes a proposed `controls` list into that agent's
`DomeConfig.config_body` via the **existing** create/pending flow (§3.6) — no new
artifact, no new HTTP endpoint for Dome to poll, no new ETag scheme. One `Control`
per `consequence.action` bucket (§5D), each with the relevant rules'
`rule_id`/`rule_type`/`natural_language`/`action`/`target` embedded directly in the
evaluator config, condition pointing at `dome:policy-rule-judge`. The policy owner
then edits/approves it through whatever surface already drives `DomeConfig`
create/`apply()` today, and `apply()` promotes it to active exactly like any other
dome config change — "propose, let a human adjust and override" falls out of the
existing pending/active lifecycle for free.

**Mandatory vs. adjustable**: confirmed not worth building — every proposed control
is just a plain, editable entry in the list; no locked/non-removable UI treatment.
"Mandatory" only matters as "what gets proposed in the first place" (see below), not
as a constraint on what a human can later change.

**One agent enforcing multiple policies — confirmed, not hypothetical.** `controls`
is a flat list, so merging is additive, but *re-running* the compiler needs to only
touch what it generated before, not a human's edits or another policy's controls.
Each generated `Control` should carry `annotations.source_policy_id` /
`annotations.source_rule_id` (`Control.annotations` already exists and allows extra
fields, §3.2) so `PolicyGapCompiler`, on trigger for policy X, can replace-in-place
only the controls tagged `source_policy_id == X` within `config_body.controls`,
leaving controls from other policies and anything a human added by hand alone.
`DomeConfig` itself needs `source_policy_ids: list[UUID]` (not a single id, per §3.7)
to track which policies an agent's config was ever compiled from.

**"Mandatory" and the evaluation-gap question (§3.7)**: true gap analysis —
cross-referencing an agent's actual evaluation/report findings against which rule
categories are and aren't already mitigated — needs a vocabulary mapping between
Diamond/reports' `harness_id`/`code` taxonomy and `PolicyRule`'s
`category`/`action`/`target` taxonomy that **does not exist yet** (§3.7). Rather than
block this feature on building that mapping, recommend an MVP simplification: every
approved rule gets proposed on trigger, regardless of `consequence.action` — the
optional `[+ findings]` input in §4's diagram (using evaluation history to narrow
*which* rules actually need a control, rather than proposing all of them) is real
value but is its own project and should be a fast-follow, not part of this feature's
first cut.

**Schema note**: `config_body` is an untyped dict server-side today (no Pydantic
model validates it), so adding a `controls` key alongside the existing
`input-guards`/`output-guards` keys is additive and doesn't require a migration —
just confirm nothing downstream assumes `config_body`'s key set is exactly the
legacy guard-list shape.

### B. Dome: a `VijilDome` constructor that reads from S3 (new)

`vijil_dome/core.py`'s `VijilDome.__init__` only accepts a `list[dict | Control]`, a
file path, or `None` (§3.6) — there's no S3-aware path the way the *old* `Dome` class
has (`Dome.create_from_s3`). Add one (e.g. `VijilDome.create_from_s3(...)` or a
`policy_source` option that accepts the same `(team_id, agent_id)` S3 key), reusing
`vijil_dome/utils/config_loader.py`'s existing fetch/TTL/ETag logic as-is — it already
returns a parsed dict; this constructor just needs to additionally read that dict's
`controls` key (parallel to how the old `Dome` class reads `input-guards`/
`output-guards` from the same dict) and hand it to `ControlEngine.load_controls(...)`.
No new freshness mechanism, no new Console-side endpoint — the existing 300s-TTL +
S3-ETag check Console's own `dome_configs` system already relies on just gets a
second consumer.

### C. `PolicyRuleJudge` — the judge detector (new)

New Dome `DetectionMethod`, e.g. `"policy-rule-judge"`, backed by Jev/System One
(§3.5) rather than `LlmBaseDetector`'s generic chat-completions path. Takes the rules
embedded in whichever bucketed `Control` invoked it (no separate Console call needed
at judge time — the `config_body` the agent already loaded carries everything),
poses one **Noul**
question per rule ("did this turn violate rule `{rule_id}`: `{natural_language}`?")
batched into a single System One call, gets back calibrated per-rule violation
probabilities directly (no JSON-parsing fallback regex needed, unlike
`PolicyGptOssSafeguard`). Batches rules the same way `PolicySectionsDetector`
batches sections if a policy's approved-rule count threatens context/request limits.

### D. Wiring into `AgentControl` and deciding what happens on a violation

Because the `dome:` bridge already exposes any `DetectionMethod` to the control
schema for free, this needs **no changes to `controls/models.py` or
`controls/engine.py`**. The proposed `controls` list (§5A) is already bucketed by
`consequence.action` at generation time, so each `Control`'s `action.decision` is
fixed before a human ever reviews it:

- `block` **and** `escalate` → `decision: deny`, `on_error: fail_closed` (per the
  user: escalate is treated as deny for enforcement purposes; it still gets a
  higher-severity audit tag so downstream alerting/reporting can distinguish "routine
  block" from "escalate-worthy block" even though the live decision is the same).
- `warn` / `flag` → `decision: steer` or `observe`, with the rule's
  `message`/rationale as `steering_context`.
- `log` → `observe` only, routed to audit.

This keeps the "one evaluator call → one decision" invariant the engine already
assumes; running the judge once per bucket-control active for a turn is fine as long
as the judge result is cached per-turn across buckets (single Jev call, multiple
controls reading the cached result).

### E. Traceability — via OTel, not a new write-back API

Per a steer from review: violations should trigger *within* Dome and ride Dome's
existing OTel instrumentation back to wherever it's configured to export to, rather
than Dome calling a new Console "report a violation" endpoint. This is much less new
work than it sounds, because the pipe already exists end-to-end:

- `helm_charts/vijil-dome/values/secrets/example.yaml` (in vijil-console) already
  defines exactly this convention for a standalone Dome deployment:
  `ENABLE_DOME_INSTRUMENTATION`, `DOME_{METRICS,TRACES,LOGS}_COLLECTOR_ENDPOINT` +
  `_TOKEN` (pointed at Console's own OTel ingestion, e.g.
  `https://otel.dev05.vijil.ai/v1/traces`), and `AGENT_ID` / `USER_ID` / `TEAM_ID` as
  resource attributes. Console's telemetry query layer
  (`src/domains/telemetry/trace_models.py`) already does tenancy-filtered trace
  search keyed on a verified **`team.id`** resource attribute (see its `CON-433`
  comment) — i.e. "Dome telemetry shows up in Console, scoped to the right team" is
  an already-solved, already-shipping problem for Dome's existing detectors.
- Dome's `_add_darwin_detection_spans` (`vijil_dome/integrations/instrumentation/otel_instrumentation.py`)
  already wraps every `Guardrail.scan`/`async_scan` call in a `dome-detection` span
  with `team.id`/`agent.id`/`user.id` + `detection.label`/`score`/`method` attributes,
  specifically so "Darwin's `TelemetryDetectionAdapter` can query them from Tempo."
  Any detector plugged into an input/output `Guardrail` gets this for free today.

**What's actually missing** for policy violations specifically:

1. `GuardResult`/`GuardrailResult` (`vijil_dome/guardrails/__init__.py`) have a fixed
   schema — `flagged`, `detection_score`, `triggered_methods`, no generic metadata
   dict — so there's nowhere to carry `rule_id`/`policy_id`/`consequence.action` /
   `severity` through `_set_darwin_span_attributes` today. And since we're wiring
   `PolicyRuleJudge` through the **AgentControl** `Control`/`ControlEngine` path (§5D)
   rather than a plain `Guardrail`, to get the deny/steer/observe decision semantics —
   that path currently has **zero OTel span emission at all** (grepped
   `controls/engine.py`/`controls/decorator.py` for `span`/`tracer`/`otel`: nothing).
2. So this needs one small, net-new piece: a `dome-control` span (same shape as
   `dome-detection`, same `_safe_set_attribute` helper from
   `instrumentation/tracing.py`) emitted from `ControlEngine.evaluate()` or the
   `@control` decorator, carrying `control.name`, `control.decision`
   (deny/steer/observe), and — specific to this feature — `policy.id`, `rule.id`,
   `rule.type`, `consequence.action`, `consequence.severity`, plus the existing
   `team.id`/`agent.id`/`user.id`.
3. `POLICY_ID` should join `AGENT_ID`/`TEAM_ID`/`USER_ID` as a resource attribute in
   the same Dome deployment config, so "show me every enforcement decision for this
   policy" is a plain Tempo/trace-search filter, the same way `team.id` already
   enables per-team filtering.

With those two additions, "feed back into Console" is just "point `DOME_TRACES_COLLECTOR_ENDPOINT` at Console's collector, same as any other Dome deployment already does" — no new Console-side ingestion endpoint, no new write-back API, and it's consistent with how every other Dome detection already surfaces in Console today.

### F. Evaluation-side linkage

The user's ask also mentioned linking policies to evaluations, not just enforcement.
That's largely **already covered**: Console's existing rule-driven custom-harness
generation (approved rules → generated test harness) is the evaluation-side story.
Nothing new is needed there for this scoping pass beyond making sure the same
`PolicyRuleJudge` detector (or at least its Noul-question-construction logic) is
reusable as a scoring function inside a harness run, not just as a live `Control`, so
"did this transcript violate rule X" is computed identically whether it's asked live
or after the fact.

## 6. Open questions

All resolved in review:

- Judge model → Jev/System One via OpenRouter, confirmed typed (not
  chat-completions) wire format (§3.5).
- Rule freshness → reuse `DomeConfig`'s existing S3 TTL/ETag check, no new HTTP/hash
  endpoint (§3.6, §5B).
- `escalate` → `deny` (§5D).
- Console credentials → supplied per-deployment, scoped to `(team_id, agent_id)` the
  same way `DomeConfig` already is, not shared (§5B).
- Violation reporting → ride Dome's existing OTel instrumentation back to Console's
  already-configured collector, not a new write-back API (§5E).
- Delivery artifact → reuse `DomeConfig.config_body`'s existing
  pending→apply→active lifecycle instead of a new hosted spec (§3.6, §4, §5A).
- Recompile trigger → **on explicit trigger only**, no automatic recompile (§5A).
- Multi-policy → **confirmed, one agent can enforce several policies at once**;
  `PolicyGapCompiler` merges via `annotations.source_policy_id`-tagged
  replace-in-place per policy, `DomeConfig.source_policy_ids` is a list (§5A).
- Mandatory-vs-adjustable UI distinction → **not worth building**; every proposed
  control is a plain, editable list entry (§5A).

None outstanding — this doc is ready to break into implementation tickets.

## 7. Suggested phasing

- **MVP**: Console's `PolicyGapCompiler` (§5A — run on trigger, proposes every
  approved rule from one or more policies, merged by `source_policy_id` tagging so
  re-running it for one policy doesn't clobber another's controls or a human's
  manual edits) writing into the existing `DomeConfig` pending slot + a `VijilDome`
  S3-aware constructor (§5B) + `PolicyRuleJudge` on Jev/System One (§5C) +
  `block`/`escalate`→deny and `warn`/`flag`→steer buckets (§5D) + the new
  `dome-control` OTel span with `policy.id`/`rule.id`/`consequence.*` attributes,
  wired to whichever `DOME_TRACES_COLLECTOR_ENDPOINT` the deployment already points
  at Console with (§5E) — no new Console ingestion endpoint, no new hosted artifact,
  no new ETag scheme.
- **Fast follow**: `log` bucket, rule-count-aware batching
  (`PolicySectionsDetector`-style fast-fail) for policies with many rules, a Console
  UI surface for querying enforcement spans by `policy.id`.
- **Later**: the harness/probe → policy-rule-category taxonomy mapping (§3.7) that
  would let `PolicyGapCompiler` actually use an agent's evaluation history to decide
  what's a gap, rather than proposing every approved rule; reuse the same judge as an
  evaluation-time scorer, unifying the enforcement and evaluation code paths.
