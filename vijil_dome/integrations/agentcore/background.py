# Copyright 2025 Vijil, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Background S3 config polling and coordinated shutdown for long-lived agents."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable, Dict, Optional

from vijil_dome.Dome import Dome
from vijil_dome.utils.config_loader import build_s3_config_key, load_dome_config_from_s3

from .settings import AgentCoreRuntimeSettings, load_agentcore_runtime_settings_from_env

if TYPE_CHECKING:
    from .otel_exporters import AgentCoreOtelExporterHandle

logger = logging.getLogger("vijil.dome")

# Fixed interval when :func:`start_agentcore_background_services` starts the S3 poller.
AGENTCORE_S3_POLL_INTERVAL_SECONDS = 300.0


def _should_start_s3_config_poller(dome: Dome, settings: AgentCoreRuntimeSettings) -> bool:
    """True if *dome* was built with S3 origin metadata, or *settings* has bucket + team + agent."""
    if dome._s3_bucket is not None and dome._s3_key is not None:
        return True
    return bool(
        settings.s3_config_bucket
        and (settings.team_id or "").strip()
        and (settings.agent_id or "").strip()
    )


def combine_agentcore_s3_reload_callbacks(
    *,
    otel_exporter_handle: Optional["AgentCoreOtelExporterHandle"] = None,
    on_s3_reload: Optional[Callable[[Dome], None]] = None,
) -> Optional[Callable[[Dome], None]]:
    """Return a callback that re-instruments OTel on *dome* (when supported), then *on_s3_reload*.

    If *otel_exporter_handle* provides ``reinstrument_dome`` (as with
    :func:`vijil_dome.integrations.agentcore.otel_exporters.setup_agentcore_otel_for_dome`),
    it is invoked first so traces, metrics, and Darwin spans attach to guardrails loaded from S3.
    """
    if otel_exporter_handle is None:
        return on_s3_reload
    rd = getattr(otel_exporter_handle, "reinstrument_dome", None)
    if rd is None or not callable(rd):
        return on_s3_reload
    user_cb = on_s3_reload

    def merged(d: Dome) -> None:
        rd(d)
        if user_cb is not None:
            user_cb(d)

    return merged


class DomeS3ConfigPoller:
    """Background thread that reloads Dome config from S3 on an interval.

    Two modes:

    - **Dome origin** — *dome* was created with :meth:`~vijil_dome.Dome.Dome.create_from_s3`;
      each tick calls :meth:`~vijil_dome.Dome.Dome.reload_from_s3_if_changed`.
    - **Settings** — *settings* supplies ``s3_config_bucket`` plus ``team_id`` and ``agent_id``
      (key from :func:`~vijil_dome.utils.config_loader.build_s3_config_key`); each tick loads
      JSON and applies via :meth:`~vijil_dome.Dome.Dome.apply_config_dict`.

    The poller does not start automatically; call :meth:`start` after your agent is ready,
    and :meth:`stop` during graceful shutdown. The first poll runs after the first
    *interval_seconds* wait (not immediately).
    """

    def __init__(
        self,
        dome: Dome,
        settings: AgentCoreRuntimeSettings,
        interval_seconds: float = 300.0,
        *,
        on_reload: Optional[Callable[[Dome], None]] = None,
        thread_name: str = "vijil-dome-s3-config-poller",
        daemon: bool = False,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._dome = dome
        self._settings = settings
        self._interval = interval_seconds
        self._on_reload = on_reload
        self._thread_name = thread_name
        self._daemon = daemon
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._poll_mode: str = ""
        self._last_remote: Optional[Dict] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("DomeS3ConfigPoller.start() called while thread is already running")
            return
        if self._dome._s3_bucket is not None and self._dome._s3_key is not None:
            self._poll_mode = "dome_origin"
            self._last_remote = getattr(self._dome, "_s3_config_dict", None)
        elif (
            self._settings.s3_config_bucket
            and self._settings.team_id
            and self._settings.agent_id
        ):
            self._poll_mode = "settings"
            self._last_remote = None
        else:
            raise ValueError(
                "DomeS3ConfigPoller requires either Dome.create_from_s3() metadata on dome, or "
                "AgentCoreRuntimeSettings with s3_config_bucket, team_id, and agent_id."
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=self._daemon,
        )
        self._thread.start()
        logger.info(
            "Started Dome S3 config poller (mode=%s, interval=%ss, daemon=%s)",
            self._poll_mode,
            self._interval,
            self._daemon,
        )

    def stop(self, *, join_timeout: float = 15.0) -> None:
        """Signal the poller to exit and wait for the thread to finish."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logger.warning(
                    "DomeS3ConfigPoller thread did not exit within join_timeout=%s",
                    join_timeout,
                )
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _poll_via_settings(self) -> bool:
        bucket = self._settings.s3_config_bucket
        assert bucket is not None
        aid = self._settings.agent_id or ""
        key = build_s3_config_key(self._settings.team_id, aid)
        remote = load_dome_config_from_s3(bucket=bucket, key=key, cache_ttl_seconds=0)
        if Dome._equivalent_s3_configs(self._last_remote, remote):
            return False
        self._dome.apply_config_dict(remote)
        self._last_remote = remote
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self._interval):
                break
            try:
                if self._poll_mode == "dome_origin":
                    changed = self._dome.reload_from_s3_if_changed()
                else:
                    changed = self._poll_via_settings()
                if changed:
                    logger.info("Dome guardrails reloaded from S3")
                    if self._on_reload is not None:
                        try:
                            self._on_reload(self._dome)
                        except Exception:
                            logger.exception("on_reload callback failed")
            except Exception:
                logger.exception("Dome S3 config poll iteration failed")


class AgentCoreBackgroundServices:
    """Coordinates optional S3 polling and optional OTel exporter teardown.

    Construct via :func:`start_agentcore_background_services` so defaults and env
    wiring stay consistent.
    """

    def __init__(
        self,
        *,
        poller: Optional[DomeS3ConfigPoller],
        settings: AgentCoreRuntimeSettings,
        otel_exporter_handle: Optional["AgentCoreOtelExporterHandle"] = None,
    ) -> None:
        self._poller = poller
        self._settings = settings
        self._otel_exporter_handle = otel_exporter_handle

    def shutdown(self, *, join_timeout: float = 15.0) -> None:
        """Stop background threads; if an OTel handle was passed in, shut it down."""
        if self._poller is not None:
            self._poller.stop(join_timeout=join_timeout)
        if self._otel_exporter_handle is not None:
            self._otel_exporter_handle.shutdown()


def start_agentcore_background_services(
    dome: Dome,
    *,
    settings: Optional[AgentCoreRuntimeSettings] = None,
    on_s3_reload: Optional[Callable[[Dome], None]] = None,
    otel_exporter_handle: Optional["AgentCoreOtelExporterHandle"] = None,
    poll_interval_seconds: float = AGENTCORE_S3_POLL_INTERVAL_SECONDS,
) -> AgentCoreBackgroundServices:
    """Build (and optionally start) background services from explicit or env settings.

    If *settings* is omitted, values are read from :func:`load_agentcore_runtime_settings_from_env`.

    A :class:`DomeS3ConfigPoller` is created and **started** when *dome* was created with
    :meth:`~vijil_dome.Dome.Dome.create_from_s3` (S3 metadata on the instance), or when
    *settings* supplies ``DOME_CONFIG_S3_BUCKET``-backed ``s3_config_bucket`` together with
    non-empty ``team_id`` and ``agent_id``. Otherwise no poller runs. Poll interval defaults
    to :data:`AGENTCORE_S3_POLL_INTERVAL_SECONDS` (300s); override *poll_interval_seconds*
    for tests or special deployments (not read from the environment).

    Always call :meth:`AgentCoreBackgroundServices.shutdown` from your agent's shutdown path
    (signal handler, ``atexit``, framework hook, etc.).

    Pass *otel_exporter_handle* (for example from :func:`setup_agentcore_otel_for_dome` or
    :func:`setup_agentcore_otel_exporters_from_env`) so :meth:`~AgentCoreBackgroundServices.shutdown`
    calls :meth:`AgentCoreOtelExporterHandle.shutdown` and flushes global OTel providers.

    When the poller runs and *otel_exporter_handle* implements ``reinstrument_dome``
    (the handle from :func:`~vijil_dome.integrations.agentcore.otel_exporters.setup_agentcore_otel_for_dome`),
    the S3 reload callback automatically invokes it **before** *on_s3_reload* so new
    guardrails stay instrumented.

    Install extras for full functionality: ``pip install 'vijil-dome[s3,opentelemetry]'``.
    """
    resolved = settings or load_agentcore_runtime_settings_from_env()
    poller: Optional[DomeS3ConfigPoller] = None
    merged_reload = combine_agentcore_s3_reload_callbacks(
        otel_exporter_handle=otel_exporter_handle,
        on_s3_reload=on_s3_reload,
    )
    if _should_start_s3_config_poller(dome, resolved):
        poller = DomeS3ConfigPoller(
            dome,
            resolved,
            interval_seconds=poll_interval_seconds,
            on_reload=merged_reload,
        )
        poller.start()
    return AgentCoreBackgroundServices(
        poller=poller,
        settings=resolved,
        otel_exporter_handle=otel_exporter_handle,
    )
