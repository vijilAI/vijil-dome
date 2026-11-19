# setup_vijil_telemetry: send in-process Dome telemetry to Console

**Date:** 2026-11-18
**Status:** Draft for review
**Tracks:** vijil-dome#353 item 4 (G19)
**Builds on:** #359 (`VijilDome` runs guards; `instrument_vijil_dome(vijil_dome, tracer, meter=None, handler=None)`)

## Why

Here is the reporter's use case. A local agent, with no URL and no chat-completions or A2A interface, is registered with the SDK. The user configures a Dome policy in Console, runs Dome in-process, and then sees the agent's Dome events in Console for the last 7 days. The last step fails today unless the user hand-builds OpenTelemetry exporters.

Console already has the receiving end, from vijil-console #1495:

- The endpoint is `POST {console}/v1/otlp/v1/{traces,logs,metrics}`. It uses OTLP over HTTP; there is no gRPC path.
- It authenticates with a Console API key pair: `Authorization: Bearer <client_id>:<client_secret>`, where `client_id` is `vk_…`. A single-string `VIJIL_API_KEY` JWT is rejected.
- The collector deletes any team or service attributes the sender supplies. It stamps `team.id` from the key and sets `service.name` to `<team>/vijil.dome`.
- Defend filters on `resource.agent.id` or `span.agent.id`, which must equal the Console agent UUID. Spans without it show as unattributed.
- Limits are 10 requests per second per key (burst 20) and 10 MB per request.
- Probed on 2026-11-18: both `console-api.vijil.ai` and `console-api.staging.vijil.ai` answer an unauthenticated POST with `401 missing_credentials`. The route is live and publicly reachable.

The library has no code that builds exporters, and nothing in it reads an endpoint setting:

- The `instrument_vijil_dome` docstring says to "point `DOME_TRACES_COLLECTOR_ENDPOINT` at Console's collector", but the library never reads that variable.
- `setup_vijil_logging` uses gRPC only and can't send headers.
- The working recipe exists only in the hosted container: `vijil-dome-marketplace/app/instrumentation_setup.py`. It builds OTLP HTTP exporters with the auth header, adds a Resource with `agent.id`, and calls `instrument_dome`.

## Goals

1. **One explicit call sends an instance's Dome telemetry to Console.** It works for `Dome` and `VijilDome`, and it covers guard detections, control decisions, metrics and Dome logs.
2. **Events land under the right agent in Console's Defend and Monitor views without extra attribute plumbing.**
3. **The same call can also, or instead, send to any other backend the user picks.**
4. **The app's own OpenTelemetry setup is untouched.**
5. **Mistakes fail loudly at setup**, never silently at export time. Examples: a missing key, a missing agent ID, or the OTel extra not installed.

## Non-goals

- Changing Console or the collector.
- Migrating `vijil-dome-marketplace` or `vijil-dome-proxy` onto the helper. They can adopt it later.
- Fixing `setup_vijil_logging`. It stays as it is; the helper doesn't use it.
- Resolving the agent UUID from a name, or registering the agent. The SDK registers the agent, and the user passes the UUID.
- Steer handling (item 3), `OPENROUTER_API_KEY` (item 14), and CLI or config-only protection (item 15).

## Anti-goals

- **Sending the app's own traces to Console.** The Console exporter must never be attached to the app's global providers.
- **Replacing or mutating the app's global TracerProvider, MeterProvider or LoggerProvider.**
- **Setup that "succeeds" while nothing is exported.** For example, an empty credential, no agent ID with Console on, or a second call that is silently ignored.

## Approaches considered

- **A. Loaders turn telemetry on automatically.** `VijilDome.create_from_api` already holds the key, agent and URL, so it would start exporting itself. That means zero extra code, but data leaves the process without any visible call, and it only works for the API loader. *Not chosen.*
- **B. An explicit `setup_vijil_telemetry(...)` call.** *Chosen by Varun* because it is explicit in a good way. It is one line in code the agent already contains, it states that data leaves the process, and it works for both classes.
- **Destinations.** We considered Console only, and Console plus the app's global providers. *Chosen: Console by default, plus optional extra exporters on Dome's own providers*, configurable per Varun. Dome's spans still join the app's traces in other backends, because span context is carried per request and not per provider.

## Design

### API

New module `vijil_dome/integrations/vijil/telemetry_setup.py`, exported as `vijil_dome.setup_vijil_telemetry`:

```python
def setup_vijil_telemetry(
    dome: Dome | VijilDome,
    *,
    client_id: str | None = None,        # env VIJIL_CLIENT_ID
    client_secret: str | None = None,    # env VIJIL_CLIENT_SECRET
    agent_id: str | None = None,         # env VIJIL_AGENT_ID
    base_url: str | None = None,         # env VIJIL_CONSOLE_URL, then DEFAULT_CONSOLE_API_BASE_URL
    user_id: str | None = None,
    send_to_console: bool = True,
    span_exporters: Sequence[SpanExporter] = (),
    metric_exporters: Sequence[MetricExporter] = (),
    log_exporters: Sequence[LogExporter] = (),
) -> VijilTelemetry
```

Each explicit argument wins over its environment variable. The env names match those `examples/ci_gate` already uses.

### Behaviour

1. **Check for the OTel extra.** Import the OTel SDK and the OTLP HTTP exporters lazily. If they are missing, raise `ImportError("setup_vijil_telemetry requires the opentelemetry extra: pip install 'vijil-dome[opentelemetry]'")`.
2. **Validate the configuration.**
   - If `send_to_console`: `client_id`, `client_secret` and `agent_id` must all be non-empty after the env fallback, else raise `ValueError` naming the missing ones.
   - If not `send_to_console`: at least one extra exporter is required, else raise `ValueError` (there would be nowhere to send).
   - If the instance is already instrumented (`_instrumented`): raise `RuntimeError`. `instrument_*` would ignore the second call, so continuing would break anti-goal 3.
3. **Check the key (only when `send_to_console`).** Send one authenticated POST of an empty body (`Content-Type: application/x-protobuf`; zero bytes is a valid empty `ExportTraceServiceRequest`) to `{base}/v1/otlp/v1/traces`, with a 5-second timeout. Chosen by Varun.
   - On 401 or 403, raise `ValueError` including Console's JSON `message`, for example a revoked key or an unknown key.
   - On any other status (including 429 and 5xx), a network error or a timeout, log one WARNING and carry on, so an agent that starts offline or while Console is degraded still runs.
   - Nothing is instrumented until the check has passed or been waived.
4. **Build the Resource:** `service.name="vijil.dome"`, `agent.id` (when known), and `user.id` (when given). `team.id` isn't sent; the collector stamps it from the key.
5. **Build private providers.** Never use `set_tracer_provider` or the other global setters.
   - `TracerProvider(resource)`, with a `BatchSpanProcessor` per exporter.
   - `MeterProvider(resource, metric_readers=[PeriodicExportingMetricReader(e) for e in exporters])`.
   - `LoggerProvider(resource)`, with a `BatchLogRecordProcessor` per exporter, wrapped in an OTel `LoggingHandler`.
   - When `send_to_console` is on, the Console exporters come first: `OTLP{Span,Metric,Log}Exporter(endpoint=f"{base}/v1/otlp/v1/{signal}", headers={"Authorization": f"Bearer {client_id}:{client_secret}"})`. The user's exporters follow.
6. **Instrument the instance** with `tracer_provider.get_tracer("vijil.dome")`, `meter_provider.get_meter("vijil.dome")` and the handler:
   - `Dome` → `instrument_dome(dome, handler, tracer, meter)`
   - `VijilDome` → `instrument_vijil_dome(vijil_dome, tracer, meter=meter, handler=handler)`
7. **Return a handle.** `VijilTelemetry` holds the three providers and has `force_flush()` and `shutdown()`. Setup also registers `shutdown` with `atexit`, so a short-lived script doesn't lose the last batch.

### Settled during review (builder's calls, made by Claude)

- **`agent_id` fallback.** The order is argument, then `VIJIL_AGENT_ID`, then the instance's own `agent_id`. Most loader users therefore pass nothing. If an explicit or env value differs from a non-empty instance `agent_id`, raise `ValueError`, because the Resource and span-level IDs would disagree and the span-level one wins in Defend.
- **Several instances in one process.** Spans and metrics are per instance, but Dome's loggers are process-wide. The first `setup_vijil_telemetry` call attaches the log handler. Later calls for other instances export spans and metrics only, and log one WARNING that Dome logs keep going through the first setup. This avoids duplicated logs carrying another agent's `agent.id`.
- **Changes to the app's logging config.** The helper makes the same changes `instrument_dome` already makes:
  - Dome's loggers go to INFO with the forward-to-ancestors shim, so the app's console is unchanged.
  - OTel's `LoggingInstrumentor` is enabled.

  Goal 4 covers the app's OTel *providers*. These logging changes are an accepted, documented limit, not new behaviour.
- **Exporter ownership.** Exporters passed to the helper belong to it. `shutdown()` and the `atexit` hook shut them down along with the providers. The docs say to pass exporters dedicated to Dome, not ones shared with the app's pipeline.
- **`base_url` normalization.** Strip trailing slashes. The value is the Console API base without `/v1`, the same convention as `DEFAULT_CONSOLE_API_BASE_URL` and the config loaders. `http://` is accepted (tests and local collectors).

### Identity on spans

`agent.id` on the Resource covers every span the providers export. That includes the per-guard and per-detector spans, which don't carry identity themselves. Span-level `agent.id` from the instance (`Dome.guard_*(agent_id=…)`, or the `VijilDome` instance values) still applies, and Defend accepts either. If they differ, the span-level value wins in Defend's query. That's acceptable: it is what the caller asked for per call.

### Logs for controls-only `VijilDome`

`instrument_vijil_dome` attaches the log handler only when the instance has guards (#359). A controls-only instance would export spans but no logs. The helper therefore calls the shared Dome-log export (`_export_dome_logs(handler)`) itself whenever it has a handler, so logs flow either way.

### Docs

- **`instrument_vijil_dome` docstring:** replace the `DOME_TRACES_COLLECTOR_ENDPOINT` sentence with a pointer to `setup_vijil_telemetry`.
- **`docs/AGENTS.md`:** add the one-line usage, the env vars, and the note that the key pair (not `VIJIL_API_KEY`) is required.
- **PR description:** list the docs-site change: a "Send telemetry from an in-process agent" section on `developer-guide/protect/observability.mdx`, with the extra-exporters example.

## Testing

- **Wire format.** Run a local HTTP server standing in for Console and point `base_url` at it. One guarded scan must produce:
  - POSTs to `/v1/otlp/v1/traces`, `/v1/otlp/v1/metrics` and `/v1/otlp/v1/logs`
  - `Authorization: Bearer vk_…:secret` on each
  - an OTLP traces payload, decoded with `opentelemetry-proto`, that contains a `dome-detection` span and a Resource with `agent.id`
  - and, for a `VijilDome` with a deny control, a `dome-control` span
- **Extra exporters.** With `span_exporters=[InMemorySpanExporter()]` and Console on, both the local server and the in-memory exporter receive `dome-detection`. With `send_to_console=False`, only the in-memory exporter does, and no key is needed.
- **Isolation.** The global tracer provider is the same object before and after setup.
- **Validation.**
  - Each missing credential or `agent_id` raises `ValueError` naming it.
  - `send_to_console=False` with no exporters raises.
  - A second call raises `RuntimeError`.
  - A missing extra raises the `ImportError` (simulated by blocking `opentelemetry.exporter.otlp.proto.http`).
  - Env fallback works, and an explicit argument beats the env value.
- **Key check.**
  - The local server returns 401 with a JSON message: setup raises `ValueError` carrying that message, and the instance is not instrumented.
  - It returns 503, or is unreachable: setup warns and completes.
  - With `send_to_console=False`, no check request is sent.
- **Flush.** `force_flush()` delivers pending spans before return.
- **Live smoke test (manual, before release).** One run against staging with a real team key and a registered agent; confirm the event appears in Defend. Record the result in the PR.

## Left to the builder

- Internal helper names and module layout inside `telemetry_setup.py`.
- Exporter timeouts and batch settings: use OTel defaults unless a test shows a problem.
- Whether `VijilTelemetry` is a dataclass or a small class.
- The exact docs wording.
