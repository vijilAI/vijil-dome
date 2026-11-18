# VijilDome absorbs Dome — Phase 1

**Date:** 2026-11-18
**Status:** Draft for review
**Tracks:** vijil-dome#353, items 1 and 2 of the reporter's list

## Why

A Console Dome config is one document holding two kinds of protection:

- **guards**: `input-guards`, `output-guards`, and the guard definitions they name
- **controls**: a `controls` list, written by Console's PolicyGapCompiler

Today two classes each enforce half of it, and neither says what it dropped:

- `Dome.create_from_api` applies the guards and ignores `controls`.
- `VijilDome.create_from_api` applies the controls and ignores the guards. A base64 prompt injection the guards would block gets through.
- `Dome.create_from_vijil_agent`, the loader the docs name, can't reach Console at all. Its default host is the retired Evaluate API, its path (`/agents/{id}/dome-configs`) doesn't exist on Console, it never sends the required `team_id`, and it takes the first config in the list, which may be an unreviewed draft.

A customer running Dome inside their own agent (the Brightmind local-agent use case) has to load both classes and wire them together by hand.

This split is an unfinished migration. The April 2026 plans (vijil-sdk `2026-04-22-dome-agent-control-incorporation-plan.md`, vijil-console `2026-04-28-dome-agent-control-overlap-analysis.md`) designed one `VijilDome` that absorbs `Dome`. PR #194 shipped `VijilDome` "alongside the unchanged Dome class", and the plans' open question ("does the condition engine replace Dome's guard pipeline, or sit alongside it?") was never answered. **Decision: `VijilDome` absorbs `Dome`.** This document covers Phase 1.

## Goals

1. One `VijilDome` instance enforces a complete Console config: guards and controls.
2. Each guard behaves exactly as it does under `Dome` today: masking, chaining, early-exit, parallel execution, per-detector settings, and `on-error`.
3. `VijilDome.create_from_api` and `VijilDome.create_from_s3` apply the whole config.
4. `Dome.create_from_vijil_agent` loads the active Console config and returns something that enforces all of it.
5. When guards run inside `VijilDome`, they emit the same telemetry `instrument_dome` produces today: `dome-detection` spans, per-guard spans, and `dome-input-*` / `dome-output-*` metrics. Console's Monitor reads these, and switching classes must not blank it.
6. Nothing in a Console config is ever dropped silently.

## Non-goals (later phases)

- Porting ADK, Strands, MCP and `TrustRuntime` from `Dome` to `VijilDome`.
- Removing `Dome`, `Guard` or `Guardrail`.
- Representing guards as controls, i.e. one model all the way down (approach B below).
- Steer handling for apps (#353 item 3), telemetry export to Console (item 4), and the `OPENROUTER_API_KEY` check (item 14). Each gets its own design.
- Fixing `_translate_dome_toml`. It's a separate bug, below.

## Anti-goals

- **Any behaviour change for current `Dome` users** beyond deprecation warnings. Their scans must return identical results.
- **A guard that `Dome` would enforce passing silently under `VijilDome`.** For example, PII left unmasked, or a block downgraded to allow.
- **`VijilDome` growing a second, divergent copy of the guard pipeline.** It must reuse `Guardrail` as it is.

## Approaches considered

- **A. Run the existing guard pipeline inside `VijilDome`.** *Chosen.* `VijilDome` builds the same input and output `Guardrail` objects `Dome` builds, and runs them alongside the control engine. It reuses guard code unchanged and keeps exact parity. Cost: two mechanisms side by side inside one class.
- **B. Wrap each whole guard as one control.** A "guard" evaluator would run an entire `Guardrail`, plus a new "rewrite text" decision for masking. It ends with a single model, but it extends the control model and needs rules for how rewrites chain between controls. Deferred, and possible as a later phase on top of A.
- **C. Per-detector translation (`_translate_dome_toml`).** *Rejected.* Tested against `Dome` on a PII guard:
  - With masking on, `VijilDome` returns allow with no rewritten text, so the app sends the original unredacted text.
  - With masking off, `Dome` blocks and `VijilDome` allows, because the translator puts detector settings under the evaluator config, while the bridge reads only `config["detector_kwargs"]`.

  Masking and chaining can't be expressed per detector.

## Design

### Construction

`VijilDome.__init__` gains `guards: dict | str | DomeConfig | None = None`. It accepts the same forms `Dome(dome_config=...)` accepts: a dict, a TOML path, or a built `DomeConfig`. It's built through the same `create_dome_config(..., default_on_error)` path, with `default_on_error = "fail_closed" if enforce else DEFAULT_ON_ERROR`, matching `Dome` (DOME-167). The resulting `input_guardrail` and `output_guardrail` are stored on the instance as public attributes, as `Dome` does.

`guards=None` means no guards; `VijilDome` keeps its current default of an empty policy. It does **not** inherit `Dome()`'s default guard set.

New: `VijilDome.from_config(config_body: dict, *, enforce=True, agent_id=None, team_id=None, user_id=None)`. It splits one Console `config_body` into its two halves:

- `controls` becomes `policy`.
- `input-guards`, `output-guards` and every guard definition they name become `guards`.

The S3 and API loaders call this method. **Unknown top-level keys** are logged once at WARNING with their names, and never ignored silently. Keys known to be metadata are excluded from the warning. That set mirrors Console's `_RESERVED_KEYS` plus envelope metadata: `id`, `updated_at`, `team_id`, `agent_id`, `agent_configuration_id`, `organization`, `tool-permissions`, and the guard-pipeline flags `create_dome_config` already reads. The builder confirms the exact set from `create_dome_config` and Console's `mapping.py`.

### Evaluation order and the result

For `guard_input`, `guard_output` and their async forms:

1. **Guards run first** on the incoming text. In enforce mode, a guard block ends the call with `action="deny"` and controls don't run, matching the early-exit `Dome` users get. In shadow mode, controls still run after a guard block, so the logs show everything that would have fired. Shadow mode exists to observe without blocking.
2. **Controls then run on the text after guards.** If masking rewrote it, controls see the masked text. This mirrors guard chaining: a control never re-detects PII that a guard already removed.
3. The two outcomes merge into one `EvaluationResult`: `deny` beats `steer`, which beats `allow`.

`guard_tool_call` is unchanged: guards don't apply to tool steps.

`EvaluationResult` gains:

- `guarded_text: str | None`: the text after guards (masked, or unchanged). `None` when no guards are configured. This is the field an app must use when it forwards text, which fixes the unredacted-text failure from approach C. It is **never** the blocked message. On a guard block, the blocked message is on the deny match's `message`. So a shadow-mode app that forwards `guarded_text` forwards the real text, and shadow mode never enforces by accident. In enforce mode a block raises before the app could forward anything. This is a deliberate difference from `Dome`, whose `guarded_response()` returns the blocked message even in shadow mode. The parity test compares like with like: blocked or not, the masked text when not blocked, and the blocked message against the match `message`.
- `guard_result: GuardrailResult | None`: the full guard outcome, for callers and telemetry that need per-guard detail.

A guard block appears in `matches` as a `ControlMatch`:

- `control_name` is the guard name, and `action.decision` is `"deny"`.
- `message` is the guard's blocked response, the same string `Dome`'s `guarded_response()` returns today.
- `metadata` carries `detection_score`, `triggered_methods` and `errored_methods`, and records that the match came from a guard (for example `{"source": "guard"}`).

This means `handle_result` needs no new code. In enforce mode a guard block raises `ControlViolationError` exactly as a deny control does; in shadow mode it logs. That is `VijilDome`'s existing contract, applied to guards.

Inputs: guards scan text. A `dict` passed to `guard_input` / `guard_output` goes to controls as today and **skips guards**, with a one-time WARNING. `DomePayload.coerce` accepts only `str` and `DomePayload`. Supporting dict payloads in guards is out of scope.

Scan identity: `guard_input`, `guard_output` and their async forms gain optional keyword overrides `agent_id`, `team_id` and `user_id`, matching `Dome.guard_*`. Each falls back to the instance value, and the result is passed to `Guardrail.scan` / `async_scan`. This is a deliberate improvement on `Dome`, which passes only per-call values. Its `dome-detection` spans therefore usually lack `team.id`, the attribute Console filters traces on. `from_config` uses explicit constructor arguments first and the `config_body`'s `agent_id` / `team_id` second. The loaders pass the IDs they fetched with, so the two agree.

Sync path: `guard_input` / `guard_output` call `Guardrail.scan` (sync) and then `engine.evaluate_sync`, matching how each half runs today. The async forms use `async_scan` and `engine.evaluate`.

### Loaders

- `VijilDome.create_from_api` and `VijilDome.create_from_s3`: build through `from_config(config_body)`. Their signatures, caching and `config_has_changed` stay as they are. Their docstrings drop the "controls only" wording.
- `Dome.create_from_api` and `Dome.create_from_s3`: behaviour unchanged. If the loaded config has a non-empty `controls`, they emit a `DeprecationWarning` and a WARNING log once. The message says the controls are **not** enforced and points to `VijilDome.create_from_api` / `create_from_s3`.
- `Dome.create_from_vijil_agent(agent_id, api_key, base_url=None, client=None, enforce=True, *, team_id)`:
  - Fetches the **active** config from Console with `api_key` as the bearer token. It reuses `api_config_loader._fetch_active_config(base_url, access_token=api_key, agent_id, team_id)` against `DEFAULT_CONSOLE_API_BASE_URL`.
  - `team_id` becomes a required keyword. The function can't work without it today, so nothing that currently works breaks.
  - It returns a `VijilDome` built with `from_config`, and emits a `DeprecationWarning` naming `VijilDome.create_from_api`.

  Returning a `VijilDome` from a `Dome` static method is deliberate: the only alternative is a `Dome` that drops controls, which is the bug. Changing the signature and return type breaks no working caller:
  - The method can't reach Console with any `base_url` today. I tested the default, the Console host, and Console `/v1`.
  - A search of vijil-console, vijil-sdk, vijil-dome-proxy, the travel agents, vijil-sample-agents, vijil-proxy-servers, vijil-darwin, vijil-diamond and vijil-mcp found no code callers, only design docs (`docs/trust/…` here and in vijil-sdk). Those docs get updated in the same PR.

  `api_key` is the single-string Console access token that the local-agent guide calls `VIJIL_API_KEY`, not a client-credential pair. **Verify during build:** that Console's `/dome-configs/active` accepts a Console API key / `VIJIL_API_KEY` JWT as a bearer token, and not only the client-credential exchange token. If it doesn't, `create_from_vijil_agent` raises a clear error naming `VijilDome.create_from_api` instead of attempting a fetch.

  `create_from_vijil_evaluation` is unchanged.

### Telemetry

`instrument_vijil_dome(vijil_dome, tracer, meter=None, handler=None)`:

- Keeps the existing `dome-control` spans.
- When the instance has guardrails, applies the same instrumentation `instrument_dome` applies, to `vijil_dome.input_guardrail` / `output_guardrail`, by calling the same helpers: `instrument_with_tracer`, `instrument_with_monitors`, `_add_darwin_detection_spans`, and the `vijil.dome` logger handler and level setup.

A guard block emits `dome-detection` (with `guard_enforced`) and **not** `dome-control`. Console's Defend view (`adapters/defend/tempo_source.py`) treats `dome_detection` and `dome_control` as separate event types, and its trace summary uses `dome_control` to mean "a policy control fired". A guard block reported as both would be double-counted. `dome-control` keeps wrapping `engine.evaluate` only.

The shared part is factored into one internal function that both `instrument_dome` and `instrument_vijil_dome` call. `instrument_dome`'s behaviour is unchanged. The new parameters are optional, so existing `instrument_vijil_dome(v, tracer)` calls keep working. It stays idempotent.

### Deprecation of `Dome`

Constructing `Dome` emits nothing new in Phase 1. Only the loader paths above warn. A constructor-level deprecation comes with Phase 2, once the integrations accept `VijilDome`; warning now would fire inside our own ADK, Strands, MCP and TrustRuntime code paths, which users can't act on.

## Testing

Each test is written to fail first:

- **Parity:** for a set of guard configs, `VijilDome(guards=cfg).guard_*(text)` agrees with `Dome(cfg).guard_*(text)` on flagged vs. deny, the rewritten text vs. `guarded_text`, and the blocked message. The configs cover: a security heuristic block, PII masking, PII block, two guards chained (masking then a detector that would fire on the raw PII), early-exit, `run-parallel`, `on-error` fail-closed with a failing detector, and shadow mode.
- **Full config:** a Console-shaped `config_body` with guards and a deny control. The guard blocks the attack and the control denies the bad output; that is the reproduction from #353 inverted. Also: guards-only, controls-only, and an empty config.
- **Order:** a masking guard plus a control matching the raw PII. The control must not fire, because it sees the masked text.
- **Loaders:** `create_from_api` and `create_from_s3` with the fetch mocked apply both halves. `Dome.create_from_api` with controls warns. `create_from_vijil_agent` calls `/v1/agent-configurations/{id}/dome-configs/active` with `team_id` and returns a `VijilDome`. An unknown key warns.
- **Telemetry:** in-memory exporters show `dome-detection` and `dome-control` spans and `dome-input-*` metrics from one `VijilDome` call. Repeated `instrument_vijil_dome` calls are no-ops.
- **No regression:** the existing suite's failing set is unchanged; locally, gated models fail as they already do.

## Separate bug found during design

`_translate_dome_toml` (`controls/engine.py`), used when `VijilDome` loads a legacy TOML file, silently drops detector settings and masking (see approach C). Phase 1 doesn't route through it, but TOML users of `VijilDome` are affected today. File it as its own issue. Once `guards=` exists, the likely fix is to load TOML guards through `guards=` instead of translating them.

## Left to the builder

- Internal helper names, the module layout of the split and merge helpers, and how the shared instrumentation function is factored.
- The exact set of known metadata keys (see Construction).
- Whether `guard_result` holds the `GuardrailResult` model or a trimmed copy, as long as per-guard detail is reachable.
- Docs-site updates live in the docs repo; list the needed changes in the PR description.
- How several triggered guards map to `ControlMatch`es, and how a fail-closed detector crash is named.
- How the sync path avoids `asyncio.run` inside a running event loop. Mirror `ControlEngine.evaluate_sync`.
