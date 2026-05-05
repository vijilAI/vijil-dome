# AgentCore integration (S3 config polling + telemetry shutdown)

Optional helpers for long-lived agents (for example on AWS AgentCore) that load Dome
configuration from S3 and export OpenTelemetry. **S3 polling** starts when
`start_agentcore_background_services` sees either `Dome.create_from_s3()` metadata on the
instance, or `DOME_CONFIG_S3_BUCKET` plus `TEAM_ID` and `AGENT_ID` on settings. **OTLP
exporters** install when `DOME_OTEL_EXPORTER_OTLP_ENDPOINT` is set (or per-signal
`OTEL_EXPORTER_OTLP_*_ENDPOINT` only), unless you pass `enabled=False` to the setup helpers.

## Install

```bash
pip install 'vijil-dome[s3,opentelemetry]'
```

With Poetry, enable the ``s3`` and ``opentelemetry`` optional dependency groups.

`boto3` is required for S3 reload; the OpenTelemetry SDK is required for OTLP
exporters and graceful shutdown.

## One-call telemetry + Dome hooks (recommended)

``setup_agentcore_otel_for_dome(dome, settings=…)`` configures OTLP/HTTP exporters (same
as ``setup_agentcore_otel_exporters_from_env``) and attaches Dome guardrail
instrumentation—traces, metrics, and Darwin-style ``dome-detection`` spans—using the
global tracer and meter. Pass the returned handle to ``start_agentcore_background_services``;
when the S3 config poller runs, it invokes ``reinstrument_dome`` on that handle
**before** your ``on_s3_reload`` callback so reloaded guardrails stay wired.

```python
from vijil_dome.integrations.agentcore import (
    load_agentcore_runtime_settings_from_env,
    setup_agentcore_otel_for_dome,
    start_agentcore_background_services,
)

settings = load_agentcore_runtime_settings_from_env()
otel = setup_agentcore_otel_for_dome(dome, settings=settings)
bg = start_agentcore_background_services(
    dome, settings=settings, otel_exporter_handle=otel
)
```

Use ``setup_agentcore_otel_exporters_from_env`` only when you need OTLP export without
calling ``instrument_dome`` (custom integration).

### Coexisting with host ``setup_telemetry``

If your runtime already calls ``setup_telemetry(service_name="...")`` (or otherwise sets
global SDK tracer/meter providers), initialize Dome **after** that. vijil-dome detects
existing providers, skips ``set_tracer_provider`` / ``set_meter_provider`` (so you avoid
OpenTelemetry's "Overriding ... is not allowed" warnings), still runs
``instrument_dome``, and returns a handle whose ``manages_global_providers`` is false so
``handle.shutdown()`` does not shut down the host's providers (your framework should still
flush telemetry on process exit).

Dome OTLP uses ``DOME_OTEL_EXPORTER_OTLP_ENDPOINT`` (not ``OTEL_EXPORTER_OTLP_ENDPOINT``) so
it does not fight the host's default OpenTelemetry env configuration.

## Environment variables (AgentCore)

These populate :class:`~.settings.AgentCoreRuntimeSettings` via
``load_agentcore_runtime_settings_from_env()``. Pass that object to
``setup_agentcore_otel_for_dome(dome, settings=…)`` (or the exporter-only helper) and
``start_agentcore_background_services(dome, settings=…)`` so S3 polling and OTLP export
see the same values.

| Variable | Purpose |
|----------|---------|
| `TEAM_ID` | `team.id` / `service.namespace` on the OTel resource; with `AGENT_ID`, builds S3 key `teams/{team}/agents/{agent}/dome/config.json` for settings-based polling. Override with `team_id=` on setup. |
| `AGENT_ID` | `agent.id` on the OTel resource and S3 key segment (required for settings-based polling with `DOME_CONFIG_S3_BUCKET`). Override with `agent_id=` on setup. |
| `DOME_CONFIG_S3_BUCKET` | Bucket for that config key when **not** using `Dome.create_from_s3()`. With `TEAM_ID` and `AGENT_ID`, enables the S3 config poller in `start_agentcore_background_services`. |
| `DOME_OTEL_EXPORTER_OTLP_ENDPOINT` | Dome OTLP/HTTP **base** URL; paths `v1/traces`, `v1/metrics`, and `v1/logs` are appended per signal unless you set matching `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` / `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`. |
| `DEPLOYMENT_ENVIRONMENT` | `deployment.environment` on the OTel resource (default `production`). |

## OpenTelemetry (optional extras)

Headers, timeouts, compression, and **per-signal** OTLP URLs follow the standard
OpenTelemetry names (for example `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TIMEOUT`,
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`). The **base** URL for Dome-managed exporters is only
`DOME_OTEL_EXPORTER_OTLP_ENDPOINT` on :class:`~.settings.AgentCoreRuntimeSettings`; the
process-wide `OTEL_EXPORTER_OTLP_ENDPOINT` is not read by AgentCore setup.

**Resource** attributes come from `create_agentcore_otel_resource` (fixed `service.name=vijil.dome`,
`team.id`, optional `agent.id`, no `service.instance.id` / `service.ip`) so DELTA counters
from many short-lived sessions stay on one series for the same team/agent.

**Counter**, **UpDownCounter**, **ObservableCounter**, and **ObservableUpDownCounter** use
**delta** temporality on the OTLP metric exporter; **Histogram** and **ObservableGauge**
keep default temporality (see `setup_agentcore_otel_exporters_from_env`).

## Typical lifecycle

### A. `Dome.create_from_s3` (existing)

1. Create `Dome` with `Dome.create_from_s3(...)` (bucket/key/cache as today).
2. Optionally export `TEAM_ID` / `AGENT_ID`, `DOME_OTEL_EXPORTER_OTLP_ENDPOINT`, and/or `DOME_CONFIG_S3_BUCKET` as needed (see table).
3. **Telemetry (optional)** — when `DOME_OTEL_EXPORTER_OTLP_ENDPOINT` is set (or per-signal OTLP env only), install exporters once at startup:

   ```python
   from vijil_dome.integrations.agentcore import (
       load_agentcore_runtime_settings_from_env,
       setup_agentcore_otel_for_dome,
       start_agentcore_background_services,
   )

   settings = load_agentcore_runtime_settings_from_env()
   otel = setup_agentcore_otel_for_dome(dome, settings=settings)
   ```

4. **Background poller** — starts automatically for `create_from_s3` domes (S3 metadata on the instance). For settings-driven polling, set `DOME_CONFIG_S3_BUCKET`, `TEAM_ID`, and `AGENT_ID`.

   ```python
   dome = Dome.create_from_s3(bucket, team_id=team_id, agent_id=agent_id)
   bg = start_agentcore_background_services(dome, settings=settings, otel_exporter_handle=otel)
   ```

   Poll interval is fixed at **300 seconds** unless you pass `poll_interval_seconds=` to `start_agentcore_background_services` (for tests or special cases).

### Shutdown (both paths)

On shutdown (SIGTERM, `atexit`, framework `on_stop`, etc.):

```python
bg.shutdown(join_timeout=15.0)
```

`shutdown` stops the S3 poller first, then calls `otel.shutdown()` if you passed
`otel_exporter_handle`, which removes the optional stdlib logging bridge and shuts
down global tracer, meter, and logger providers. If you initialized OTel elsewhere
without a handle, call `shutdown_opentelemetry_providers()` yourself after `bg.shutdown()`.

### B. Local `Dome` + settings-driven S3 polling (no `create_from_s3`)

1. Build `Dome` from a dict or file as usual.
2. Set `DOME_CONFIG_S3_BUCKET`, `TEAM_ID`, and `AGENT_ID` (S3 key is always the standard `teams/{team}/agents/{agent}/dome/config.json` path).
3. Use the same `settings` object for OTel and `start_agentcore_background_services` as in (A).

On each poll tick, config is loaded with `load_dome_config_from_s3` and applied via
`Dome.apply_config_dict` (no `create_from_s3` required).

## Programmatic control (no env)

```python
from vijil_dome.integrations.agentcore import (
    AgentCoreRuntimeSettings,
    DomeS3ConfigPoller,
    setup_agentcore_otel_for_dome,
    shutdown_opentelemetry_providers,
)

settings = AgentCoreRuntimeSettings(
    s3_config_bucket="my-bucket",
    team_id="t",
    agent_id="a",
    otel_exporter_otlp_endpoint="http://localhost:4318",
)
poller = DomeS3ConfigPoller(dome, settings, interval_seconds=60.0)
poller.start()
otel = setup_agentcore_otel_for_dome(dome, enabled=True, settings=settings)
# ... later ...
poller.stop()
if otel:
    otel.shutdown()
else:
    shutdown_opentelemetry_providers()
```

## Manual reload without a poller

S3-backed `Dome` from `create_from_s3`:

```python
if dome.reload_from_s3_if_changed():
    ...
```

Or load JSON yourself and call `dome.apply_config_dict(remote_dict)`.

Change detection for the poller matches `Dome.config_has_changed()` semantics (including
`id`-based fast path when present).
