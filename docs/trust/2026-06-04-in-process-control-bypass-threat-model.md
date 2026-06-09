# Threat model: bypassing in-process Dome controls

- **Status:** Threat model (2026-06-04). Evidence-backed; every claim cites code.
- **Scope:** the in-process Dome control stack (`secure_agent()` / `TrustRuntime` / the
  guardrail + MAC pipeline) and whether an uncooperative agent developer can evade it.
- **Method:** four independent adversarial code probes across `vijil-dome`, `vijil-console`,
  `vijil-inference`, `vijil-darwin`.

## The thesis under test

> *"Vijil's in-process Dome controls can't be bypassed silently. An uncooperative developer
> would have to **excommunicate their agent from the registry** to escape the controls — and
> that absence **automatically triggers a governance alert** (a 'missing agent')."*

This is the right **target**. It is **not true of the system today**, on either half: bypass
does **not** require leaving the registry, and leaving the registry does **not** alarm.

## Why it matters

In-process enforcement is *cooperative* — it is a library call inside the agent's own process,
not an external chokepoint. Its trust value therefore rests entirely on **tamper-evidence**:
the property that you cannot remove or starve the control without producing a loud,
governance-visible signal. If evasion is silent, "the agent is Dome-protected" is a claim the
platform cannot stand behind. This document enumerates where that property fails and what
restores it.

## The control architecture (and its external dependency)

`secure_agent(agent, agent_id=, mode=)` (`vijil_dome/trust/adapters/auto.py:31`) injects hooks
into the agent object across three framework adapters (LangGraph / ADK / Strands). Four
canonical in-process hook points: prompt-in (`guard_input`), response-out (`guard_output`),
tool-call (`check_tool_call` = MAC), tool-response-out. Two enforcement classes:

- **Content guards** (input/output detectors) — these call out to the **EKS-hosted detection
  server** (`/v1/detect`) over HTTP. *In-process control with a remote dependency.*
- **Tool MAC** (`ToolPolicy.check`) — local, no remote dependency.

The remote dependency is the sharpest attack surface, because the agent's own process controls
its own network egress.

## The bypass surface (all verified, all keep the agent registered)

| # | Bypass | Silent today | Leaves registry | Detectable today |
|---|---|---|---|---|
| 1 | **Block egress to `/v1/detect`** (DNS/firewall/NetworkPolicy) → content guards fail open | partial | no | partial (trace only, opt-in) |
| 2 | **Slow-loris the detector** past the 5s/30s timeouts → same fail-open | partial | no | partial (latency only) |
| 3 | **Run with no tracer wired** (core `Dome()`/`TrustRuntime` without `instrument_dome`) then trigger fail-open | **yes** | no | **no** |
| 4 | **Default `mode='warn'`** — guards run, nothing blocks; or Dome init fails → `guards_disabled` | **yes** | no | partial |
| 5 | **Never call `secure_agent()`** — registry row intact, no Dome code runs | **yes** | no | **no** |
| 6 | **Monkey-patch the hooks** — `guard_input`/`guard_output` are mutable instance attributes | **yes** | no | **no** |
| 7 | **Shape a LangGraph agent with no top-level `.tools`** → tool-MAC silently skipped | **yes** | no | **no** |

### Hole 1 — the content guard fails OPEN by shipped default (the crux)

`vijil_dome/detectors/remote_dispatcher.py:30-33` states it in the code's own words:

> *"Graceful degradation: if the inference server is unreachable, all remote detectors return
> `is_flagged=False` with an error field set… a server outage degrades coverage, not
> availability."*

On `httpx.ConnectError` the dispatcher logs a warning, `break`s the retry loop, and returns
`DetectorResult(is_flagged=False, score=0.0, error=…)` for **every** requested detector
(`remote_dispatcher.py:222-258`). `RemoteDetectionMethod.detect()` launders that into
`(False, …)` — *not a hit* (`remote_method.py:130-140`) — so the guard pipeline treats it as
*allow*. A **second** independent fail-open exists: the Guard layer wraps each detector in
`asyncio.timeout(5.0)` and synthesizes `hit=False` on timeout (`guardrails/__init__.py:195-215`).

The configurable fail-**closed** option exists — the native `ControlEngine` honors
`action.on_error == 'fail_closed'` (`controls/engine.py:268-289`) — but **the legacy
TOML/dict guardrail path that `secure_agent` actually runs has no `on_error` knob**; it
hardcodes `hit=False` on every error/timeout. So fail-closed is not reachable from the
shipped content-guard pipeline. `route=remote` does **not** help: it raises only at
config-parse time if the URL is unset (`config_parser.py:147-160`); a firewall applied after
startup is a pure runtime `ConnectError` → fail-open.

### Hole 2 — the fail-open is nearly invisible

The Prometheus metric governance would scrape (`<name>-error_total`) increments **only** inside
an `except` around a *raised* scan (`guardrails/instrumentation/metrics.py:202-226`). The
fail-open path returns a normal result **without raising**, so `error_total` never increments;
`flagged_total` stays 0; `requests_total` and latency look healthy. The `secure_agent` audit
stream (`runtime.py:312-319`) records `direction`, `flagged`, `score`, and `exec_time_ms`, but
the discriminating detection fields (`flagged` + `score`) are `False`/`0.0` on fail-open,
**byte-identical to a genuinely clean prompt** — only `exec_time_ms` may differ, and it is not a
governance signal. The only governance-visible signal is a Tempo **trace attribute**
`detection.label='errored'` (`vijil_dome/integrations/vijil/telemetry.py:71-88`) — and
it exists only if `instrument_dome(tracer=…)` was wired, which the core `Dome()`/`TrustRuntime`
constructors **do not do** (only the example/smoke tests do). Passive, query-only, opt-in. No
in-repo rule alerts on it.

### Hole 3 — there is no "missing agent" alert

No liveness / heartbeat / last-seen / dark-agent / posture-downgrade detector exists anywhere
in Console. The Trust dashboard reads a **static** `dome_protected` flag plus a hard-coded
empty "under attack" stub (`dome_adapter.py:38`). `attestation_status` (the CON-522 field) is
**written and watched by nothing**; `attested_at` is never re-evaluated; `INCONCLUSIVE` is
never set. **Registration is orthogonal to execution** — `create_agent` is a catalog write with
no runtime hook (`service.py:71-132`); an agent can run unwrapped and never register. Even
archiving (the only de-registration path) emits no event. So even if an attacker *did*
excommunicate, nothing alarms.

## What makes the thesis true (the convergent build)

Four probes reading four subsystems independently converged on the same bounded set — the
"tamper-evidence" spine that turns *bypass ⇒ excommunication ⇒ alarm* from aspiration into a
property:

1. **Fail-CLOSED on detector-unreachable** in enforce mode — block (or loudly degrade) instead
   of silently passing; bring the `ControlEngine`'s `on_error` semantics to the default
   content-guard path. *(dome)*
2. **Make the failure loud** — a `detector_unavailable_total` counter that increments *without*
   a raised exception, and carry `errored_methods` / `guards_disabled` into the audit event
   (today `direction`/`flagged`/`score`/`exec_time_ms`, none of which flags an errored
   detector). *(dome)*
3. **An enforcement-alive heartbeat** signed with the agent's SVID, carrying effective mode +
   hooks-attached + detector-reachable. *(dome)*
4. **A Console liveness reconciler + the missing-agent alert** — `last_dome_report_at` on the
   agent record, a staleness sweep that flips `ATTESTED→INCONCLUSIVE` and flags `dome_protected`
   agents that go dark / drop to warn / lose attestation; wire `attestation_status` into the
   dashboard; **rogue-agent reconciliation** (traffic with no registry row). *(console — this is
   literally the "missing agent" alarm.)*
5. **Reject local `warn` overriding a Console-mandated `enforce`** — record the downgrade as a
   governance event (`runtime.py:92-94`). *(dome)*

## The durable root of trust (roadmap, founder direction)

Items 1–5 raise the cost of evasion and make it **loud** — the correct bar for the
**cooperative-developer** threat model. They do **not** stop a **malicious in-process** attacker
(arbitrary code can patch a library call). The durable answer is to move the root of trust into
**hardware**: a **TPM-rooted measured boot** with the agent container running inside a **TEE**
(confidential compute; see the Phala × Vijil direction), so the Dome control stack is
*measured and sealed* and its integrity is attested by hardware rather than asserted by a
cooperating process. The in-process heartbeat (item 3) becomes a TEE **remote-attestation
quote** rather than an SVID-signed beacon. This is the strategic end state; items 1–5 are the
software increment that delivers tamper-*evidence* now and composes cleanly with it.

## Threat-model boundary (honest)

- **In scope / addressed by 1–5:** an uncooperative developer who starves, downgrades, or
  declines the control — every such move becomes governance-visible.
- **Out of scope for the software increment:** a malicious process with code execution that
  patches Dome in-memory — only the TEE/TPM root of trust closes this.
- **The non-bypassable backstop remains the network proxy**, where Vijil owns the chokepoint:
  in-process control is for *portability + tamper-evidence*; the proxy is for *un-bypassable
  enforcement* where the network is controlled. The two are complementary, not redundant.
