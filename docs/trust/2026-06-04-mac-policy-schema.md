# MAC policy schema: the single source of truth

- **Status:** Reference (2026-06-04). Companion to
  `2026-06-04-in-process-control-bypass-threat-model.md` and
  `2026-06-04-tamper-evident-identity-mac-plan.md` (unit **B6**).
- **Purpose:** define **one** per-identity MAC policy field model, keyed on the agent's
  attested SPIFFE ID, and pin the **proxy ↔ in-process** mapping so the two enforcement
  surfaces stay in sync. The recurring failure here is **drift**: the Go network proxy and the
  in-process Dome runtime grow the same control with slightly different field names and
  semantics, and a policy that denies at the proxy quietly permits in-process (or vice versa).
  This doc is the contract both sides implement against.
- **Audience:** the in-process units that build the keyed decision — **A4** (model-MAC),
  **A5** (SVID-glob resolution), **A10** (fetch-by-SVID) — plus anyone touching the Go proxy
  `Route` structs.
- **Scope:** the field model and the mapping only. It does **not** redefine the threat model
  (see the threat-model doc) or re-plan the build (see the plan doc). No code ships with this
  unit.

## The one rule

> **The MAC decision is keyed on the agent's attested SPIFFE ID**
> (`spiffe://vijil.ai/org/<team>/agent/<uuid>`), **not** on the Console agent UUID and **not**
> on a developer-supplied `agent_id` string. Both the proxy and the in-process runtime resolve
> a policy by matching the *attested* SVID against an `agent_id_pattern` glob, first match wins.

This is founder fork #1 from the plan. Everything below is downstream of it. A field that
cannot be keyed on the SVID does not belong in the unified model.

## Part 1 — the proxy reference model (verified from source)

The Go proxies in `vijil-proxy-servers` are the **reference implementation** of out-of-process
MAC. The in-process model mirrors them. Field names and semantics below are quoted from the Go
`Route` structs and the matcher/enforcer, not paraphrased.

### 1.1 Route resolution (all three proxies)

`llm-proxy/internal/config/config.go::MatchRoute` resolves a policy by **first-match glob** over
the attested SVID:

```go
// MatchRoute finds the first route matching the given agent SPIFFE ID and provider.
func (c *Config) MatchRoute(agentID, provider string) *Route {
    for i := range c.Routes {
        r := &c.Routes[i]
        if provider != "" && r.Provider != provider { continue }
        if matchGlob(r.AgentIDPattern, agentID) { return r }
    }
    return nil
}
```

`matchGlob` semantics (same file):

| Pattern shape | Match rule |
|---|---|
| `*` | matches any SVID |
| no `*` | exact string equality |
| single `*` | prefix (before `*`) **and** suffix (after `*`) |
| multiple `*` | `filepath.Match(pattern, svid)` |

`MatchRoute` returns `nil` when **no** route matches. The proxy's default on a `nil` route is
the surrounding handler's concern; the **in-process default-deny-on-no-match** is specified in
Part 3 — do not assume "no route" means "permit".

### 1.2 The three proxy `Route` field models

**llm-proxy** (`llm-proxy/internal/config/config.go::Route`) — the model-MAC reference:

| Field (json tag) | Go type | Semantics (from `handler.go::checkMAC`) |
|---|---|---|
| `agent_id_pattern` | `string` | SVID glob; route key (§1.1) |
| `provider` | `string` | e.g. `anthropic`, `openai`, `google`; route key filter |
| `allowed_models` | `[]string` | **empty ⇒ any model**; otherwise requested model MUST be in the list, else deny |
| `max_tokens` | `int` | **0 ⇒ no limit**; positive ⇒ deny when requested `max_tokens` exceeds it |
| `allow_tool_use` | `*bool` | **nil ⇒ no restriction**; `false` ⇒ deny when the request carries tools; `true` ⇒ permit |

`checkMAC` enforcement order (verified, `handler.go`): model-allowlist → `max_tokens` ceiling →
`allow_tool_use`. `allow_tool_use` is a **pointer** precisely so that "unset" (`nil`) is
distinct from "explicitly false" — a plain `bool` would silently default to deny.

**mcp-proxy** (`mcp-proxy/internal/config/config.go::Route`) — the tool-server reference:

| Field (json tag) | Go type | Semantics |
|---|---|---|
| `agent_id_pattern` | `string` | SVID glob; route key |
| `allowed_servers` | `[]string` | list of `server_id` values this agent may reach |

**a2a-proxy** (`a2a-proxy/internal/config/config.go::Route`) — the agent-to-agent reference:

| Field (json tag) | Go type | Semantics |
|---|---|---|
| `caller_id_pattern` | `string` | SVID glob of the **caller**; route key |
| `allowed_targets` | `[]string` | list of target `agent_id` values this caller may reach |

> **Naming note (a real drift hazard):** a2a-proxy keys on **`caller_id_pattern`**, not
> `agent_id_pattern`, because the subject is the *calling* agent, and references targets by
> their proxy-path `agent_id` (which is **not** a SVID — it is the `/a2a/{agent_id}` path
> segment). The in-process model normalizes the **subject** field name to `agent_id_pattern`
> across all surfaces (§3) and is explicit about which identifiers are SVIDs versus path IDs,
> so the a2a asymmetry does not leak in as silent confusion.

## Part 2 — the in-process model today (verified from source)

In-process MAC lives in `vijil_dome/trust/constraints.py` and `vijil_dome/trust/policy.py`.

**`ToolPermission`** (`constraints.py`) — the per-tool grant:

| Field | Type | Semantics |
|---|---|---|
| `name` | `str` | tool name; the permission key |
| `identity` | `str` | SPIFFE ID **of the tool** (e.g. `spiffe://vijil.ai/tools/flights/v1`) |
| `endpoint` | `str` | tool endpoint (e.g. `mcp+tls://flights.internal:8443`) |
| `allowed_actions` | `list[str] \| None` | `None` ⇒ all actions; otherwise the call's `args["action"]` MUST be in the list — a missing action, `args["action"]=None`, OR `args=None` all deny (fail-closed, `policy.py`) |

**`OrganizationConstraints`** (`constraints.py`) — the org-wide overlay:

| Field | Type | Semantics |
|---|---|---|
| `required_input_guards` | `list[str]` | guards the agent must run on input |
| `required_output_guards` | `list[str]` | guards the agent must run on output |
| `denied_tools` | `list[str]` | tools denied org-wide; checked before per-tool grants (`policy.py`) |
| `max_model_tier` | `str \| None` | **dead field** — parsed but enforced nowhere; superseded by the model-MAC model in §3 |

**`AgentConstraints`** (`constraints.py`) wraps `dome_config`, `tool_permissions`,
`organization`, `enforcement_mode` (`Literal["warn", "enforce"]`), `agent_id`, `updated_at`.

**`ToolCallResult`** (`policy.py`) — the MAC decision record, already SVID-aware after A1:

| Field | Type | Semantics |
|---|---|---|
| `permitted` | `bool` | the decision |
| `tool_name` | `str` | the tool checked |
| `args` | `dict \| None` | carried for audit; not enforced beyond `allowed_actions` today |
| `identity_verified` | `bool` | set from real attestation state (A1; was hardcoded `False` before) |
| `agent_spiffe_id` | `str \| None` | the calling agent's SVID, recorded on every result |
| `policy_permitted` | `bool` | the policy's own verdict, before mode |
| `enforced` | `bool` | `True` only when the call is blocked **and** mode is `enforce` |
| `error` | `str \| None` | deny reason |

**A1 (merged) is the keystone this doc builds on.** `ToolPolicy.check(...)` already accepts
`spiffe_id` and `attested` and records them on the result, and `runtime.check_tool_call`
threads `self._identity.spiffe_id` / `is_attested()` through. **The SVID is recorded but the
permit/deny outcome is not yet keyed on it** — that is exactly what A3 (subject binding), A5
(glob resolution), and A4 (model-MAC) add. Audit emits `agent_spiffe_id` (the SVID) via
`emit_tool_mac` (`runtime.py`).

## Part 3 — the unified model (what A4/A5/A10 implement)

The unified in-process policy is a **list of SVID-keyed routes**, mirroring the proxy `Route`
list, resolved by the same first-match glob. Each route carries the **union** of the three
proxy field models plus the existing in-process tool-grant fields, all keyed on one subject.

### 3.1 The route subject — `agent_id_pattern` (A5)

Every in-process route carries an `agent_id_pattern` (SVID glob) with the **exact** `matchGlob`
semantics of §1.1: `*` = any; no-`*` = equality; single-`*` = prefix+suffix; multi-`*` =
Go `filepath.Match` semantics. The Python side (A5) must **port the Go `matchGlob` algorithm** —
`fnmatch`/`PurePath.match` are NOT equivalent to `filepath.Match` (they differ on `*`/separator
and character-class handling). Resolution is
**first-match, top-to-bottom**; **no match ⇒ deny** in `enforce` mode (the threat-model's
fail-closed posture — a developer must not escape policy by presenting an unlisted SVID).
A5 binds this against `self._identity.spiffe_id`; A3 asserts the resolved route's subject
matches the attested SVID and fails closed on mismatch.

### 3.2 The model-MAC fields (A4) — one-to-one with llm-proxy

| Unified field | Proxy source field | In-process semantics (must match the proxy) |
|---|---|---|
| `allowed_models: list[str]` | llm-proxy `allowed_models` | empty ⇒ any; else requested model MUST be in list |
| `max_tokens: int` | llm-proxy `max_tokens` | `0` ⇒ no limit; positive ⇒ deny when requested exceeds |
| `allow_tool_use: bool \| None` | llm-proxy `allow_tool_use` (`*bool`) | **`None`** ⇒ no restriction; `False` ⇒ deny when tools present; `True` ⇒ permit — keep the tri-state, do not collapse to `bool` |

A4 enforces these in a `before_model` authorization hook, mirroring `checkMAC`, in the same
order: model-allowlist → `max_tokens` → `allow_tool_use`. `max_model_tier` (§2, dead) is
**retired** by `allowed_models`; do not revive it.

### 3.3 The tool/server/target fields — existing + mcp/a2a parity

| Unified field | Source | In-process semantics |
|---|---|---|
| `tool_permissions: list[ToolPermission]` | in-process (unchanged) | per-tool grants with `allowed_actions` |
| `denied_tools: list[str]` | in-process (unchanged) | org-wide deny, checked first |
| `allowed_servers: list[str]` | mcp-proxy `allowed_servers` | MCP `server_id`s this SVID may reach (future in-process MCP MAC) |
| `allowed_targets: list[str]` | a2a-proxy `allowed_targets` | A2A target `agent_id`s this SVID may call (future in-process A2A MAC) |

`allowed_servers` / `allowed_targets` are reserved in the unified model now so the field model
does not have to be re-cut when in-process MCP/A2A MAC lands; A4/A5 may leave them empty.

### 3.4 The complete proxy ↔ in-process field map (the anti-drift table)

| Concept | llm-proxy | mcp-proxy | a2a-proxy | In-process (unified) | In-process today |
|---|---|---|---|---|---|
| Route subject | `agent_id_pattern` | `agent_id_pattern` | `caller_id_pattern` | `agent_id_pattern` | *(none — keyed on `agent_id` string)* |
| Resolution | first-match glob | first-match glob | first-match glob | first-match glob (A5) | exact `agent_id` lookup |
| Model allowlist | `allowed_models` | — | — | `allowed_models` (A4) | — (`max_model_tier`, dead) |
| Token ceiling | `max_tokens` | — | — | `max_tokens` (A4) | — |
| Tool-use gate | `allow_tool_use` | — | — | `allow_tool_use` (A4) | — |
| Tool grants | — | `allowed_servers` | — | `tool_permissions` + `allowed_servers` | `tool_permissions` |
| Peer targets | — | — | `allowed_targets` | `allowed_targets` | — |
| Org deny | — | — | — | `denied_tools` | `denied_tools` |
| Enforcement mode | *(handler)* | *(handler)* | *(handler)* | `enforcement_mode` | `enforcement_mode` |
| Decision subject | attested SVID | attested SVID | attested SVID (caller) | attested SVID | `agent_id` string |

Every row that has a proxy entry and an in-process-today gap is a unit in Half A of the plan.
Every row where the proxy and in-process names differ is a drift hazard this table pins.

## Part 4 — provenance fetch (A10)

A10 fetches the agent's constraints **by SVID** from Console, replacing the developer-supplied
`agent_id` string lookup. The fetched object MUST carry the route subject (`agent_id_pattern`)
so A3 can assert it matches `self._identity.spiffe_id`. The Console resolver (`get_by_spiffe_id`)
and the workload-auth `/constraints` endpoint are A9 (Console), out of scope here; this doc only
fixes the **field model** the endpoint returns.

## Invariants (what reviewers check against this doc)

1. **One subject.** Every MAC field is keyed on the attested SVID via `agent_id_pattern`. No
   field keyed on the Console UUID or a developer string survives into the unified model.
2. **Tri-state preserved.** `allow_tool_use` stays `bool | None`; `None` ≠ `False`.
3. **Sentinels match the proxy.** `allowed_models == []` ⇒ any; `max_tokens == 0` ⇒ no limit.
   Do not invert these in-process.
4. **No-match is deny in enforce mode.** A `nil`/absent route never silently permits.
5. **Names match §3.4.** Adding a MAC field in either the proxy or in-process updates §3.4 in
   the **same** PR — this table is the drift gate.
6. **`max_model_tier` stays dead.** It is superseded by `allowed_models`; do not re-enforce it.

## References

- Threat model: `docs/trust/2026-06-04-in-process-control-bypass-threat-model.md`
- Plan: `docs/trust/2026-06-04-tamper-evident-identity-mac-plan.md`
- In-process model: `vijil_dome/trust/constraints.py`, `vijil_dome/trust/policy.py`,
  `vijil_dome/trust/runtime.py`
- Proxy reference (repo `vijil-proxy-servers`):
  `llm-proxy/internal/config/config.go` (`Route`, `MatchRoute`, `matchGlob`),
  `llm-proxy/internal/proxy/handler.go` (`checkMAC`),
  `mcp-proxy/internal/config/config.go` (`Route`),
  `a2a-proxy/internal/config/config.go` (`Route`)
