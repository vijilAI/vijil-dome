# Copyright 2025 Vijil, Inc.
# Licensed under the Apache License, Version 2.0

"""Tests for per-detector routing resolution.

Three priority levels (highest first): YAML ``route`` key → env
``DOME_LOCAL_DETECTORS`` → default (auto). The resolver lives in
``config_parser._resolve_route``; the capability check lives in
``config_parser._should_use_remote``. These are exercised separately
so adversarial cases (invalid route value, non-remote-capable
detector forced remote) can be tested without spinning up the
dispatcher.

Test hierarchy:
- Adversarial: invalid YAML route, route=remote without URL,
  route=remote on local-only detector.
- Perturbation: env override doesn't override YAML; YAML local
  beats DOME_INFERENCE_URL set; AUTO falls through to local
  when URL is absent.
- Boundary: empty CSV in env, whitespace in env values.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from vijil_dome.guardrails.config_parser import (
    REMOTE_DETECTORS,
    _Route,
    _resolve_route,
    _should_use_remote,
)


# ---------------------------------------------------------------------------
# Fixture: clean env between tests so DOME_INFERENCE_URL and
# DOME_LOCAL_DETECTORS bleed-through can't poison results.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_routing_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("DOME_INFERENCE_URL", raising=False)
    monkeypatch.delenv("DOME_LOCAL_DETECTORS", raising=False)
    yield


# ---------------------------------------------------------------------------
# _resolve_route — pure resolver, no I/O
# ---------------------------------------------------------------------------


class TestResolveRoute:
    def test_default_is_auto_with_source_default(self) -> None:
        # Source field exists so the caller's audit log can distinguish
        # "explicit YAML" from "fell through to default" from "env override".
        route, source = _resolve_route("prompt-injection-mbert", {})
        assert route is _Route.AUTO
        assert source == "default"

    def test_yaml_local_wins_with_source_yaml(self) -> None:
        route, source = _resolve_route("privacy-presidio", {"route": "local"})
        assert route is _Route.LOCAL
        assert source == "yaml"

    def test_yaml_remote_wins_with_source_yaml(self) -> None:
        route, source = _resolve_route("privacy-presidio", {"route": "remote"})
        assert route is _Route.REMOTE
        assert source == "yaml"

    def test_env_forces_local_with_source_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", "privacy-presidio,fact-check-llm")
        route, source = _resolve_route("privacy-presidio", {})
        assert route is _Route.LOCAL
        assert source == "env"
        route, source = _resolve_route("fact-check-llm", {})
        assert route is _Route.LOCAL
        assert source == "env"
        # detector NOT in env list → default auto
        route, source = _resolve_route("prompt-injection-mbert", {})
        assert route is _Route.AUTO
        assert source == "default"

    def test_yaml_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # env says local, YAML says remote — YAML wins. This is the
        # core priority contract; if it ever flips, the env var
        # becomes a sticky override that can't be unstuck from config.
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", "privacy-presidio")
        route, source = _resolve_route("privacy-presidio", {"route": "remote"})
        assert route is _Route.REMOTE
        assert source == "yaml"

    def test_yaml_auto_explicit_does_not_inherit_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Setting route=auto in YAML is a deliberate "use the default"
        # statement and must NOT be overridden by the env var. Otherwise
        # users who explicitly opt into auto get surprised by env-var
        # leakage from their dev environment.
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", "privacy-presidio")
        route, source = _resolve_route("privacy-presidio", {"route": "auto"})
        assert route is _Route.AUTO
        # Source is "yaml" because the user explicitly specified auto,
        # even though the resolved value matches the default.
        assert source == "yaml"

    def test_invalid_yaml_route_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid route"):
            _resolve_route("privacy-presidio", {"route": "remoet"})

    def test_invalid_yaml_route_lists_valid_values(self) -> None:
        with pytest.raises(ValueError) as exc:
            _resolve_route("privacy-presidio", {"route": "off"})
        msg = str(exc.value)
        assert "auto" in msg and "local" in msg and "remote" in msg

    def test_env_csv_handles_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Real shell exports often have stray whitespace; tolerate it.
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", "  privacy-presidio  , fact-check-llm  ")
        assert _resolve_route("privacy-presidio", {})[0] is _Route.LOCAL
        assert _resolve_route("fact-check-llm", {})[0] is _Route.LOCAL

    def test_empty_env_does_not_force(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", "")
        assert _resolve_route("privacy-presidio", {})[0] is _Route.AUTO

    def test_env_only_commas_does_not_force(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ",,," should resolve to an empty set (no whitespace tokens),
        # not to "force every empty-named detector".
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", ",,,")
        assert _resolve_route("privacy-presidio", {})[0] is _Route.AUTO


# ---------------------------------------------------------------------------
# _should_use_remote — combines resolution with capability check
# ---------------------------------------------------------------------------


class TestShouldUseRemote:
    def test_auto_with_url_unset_returns_local(self) -> None:
        # Default behavior preserved: no URL, no remote.
        assert _should_use_remote("prompt-injection-mbert", {}) is False

    def test_auto_with_url_set_returns_remote_for_remote_capable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        assert _should_use_remote("prompt-injection-mbert", {}) is True

    def test_auto_with_url_set_returns_local_for_non_remote_capable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        # Pick a detector NOT in REMOTE_DETECTORS — local-only stays local.
        local_only = "flashtext-injection-detector"
        assert local_only not in REMOTE_DETECTORS
        assert _should_use_remote(local_only, {}) is False

    def test_yaml_local_overrides_url_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The headline use case: customer wants PII to stay local for
        # compliance even when URL is configured for everything else.
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        assert _should_use_remote(
            "privacy-presidio", {"route": "local"}
        ) is False

    def test_yaml_remote_with_url_set_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        assert _should_use_remote(
            "privacy-presidio", {"route": "remote"}
        ) is True

    def test_yaml_remote_without_url_raises(self) -> None:
        # The fail-loud contract: route=remote means "I require remote";
        # silent fallback to local would be a quiet downgrade.
        with pytest.raises(ValueError, match="DOME_INFERENCE_URL is not set"):
            _should_use_remote("privacy-presidio", {"route": "remote"})

    def test_yaml_remote_on_non_remote_capable_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        local_only = "flashtext-injection-detector"
        assert local_only not in REMOTE_DETECTORS
        with pytest.raises(ValueError, match="not a remote-capable detector"):
            _should_use_remote(local_only, {"route": "remote"})

    def test_env_local_overrides_url_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Dev-shortcut path: flip one detector to local without
        # editing YAML. URL is set, but the env var pulls one back.
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", "privacy-presidio")
        assert _should_use_remote("privacy-presidio", {}) is False
        # Other remote-capable detectors still go remote.
        assert _should_use_remote("prompt-injection-mbert", {}) is True


# ---------------------------------------------------------------------------
# create_detector_for_guard — route key consumed before reaching detector
# ---------------------------------------------------------------------------


class TestCreateDetectorForGuard:
    def test_route_key_not_passed_to_remote_detector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Perturbation test: ``route`` is config metadata for this
        layer; downstream detector constructors don't accept it.
        Without the explicit pop, a detector that uses ``**kwargs``
        validation would raise on the unknown ``route`` arg. This
        test catches a regression where the pop is moved or deleted.
        """
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        from vijil_dome.guardrails.config_parser import create_detector_for_guard

        detector = create_detector_for_guard(
            "prompt-injection-mbert",
            "security",
            {"route": "remote", "threshold": 0.5},
        )
        # RemoteDetectionMethod stores its config; verify route isn't there.
        assert "route" not in detector._config  # type: ignore[attr-defined]

    def test_env_forced_local_emits_audit_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Regression test for the audit-trail gap the round-3 code review
        caught: when DOME_LOCAL_DETECTORS forces a detector to local,
        the resolution must show up in logs so an operator can confirm
        the env override took effect. Without this test, removing the
        ``source`` log field would silently strip the only evidence
        env-forced LOCAL routing produces.
        """
        import logging
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        monkeypatch.setenv("DOME_LOCAL_DETECTORS", "prompt-injection-mbert")
        from vijil_dome.guardrails.config_parser import create_detector_for_guard

        with caplog.at_level(logging.INFO, logger="vijil.dome"):
            try:
                create_detector_for_guard(
                    "prompt-injection-mbert",
                    "security",
                    {},
                )
            except ValueError:
                # Local construction may fail without local model deps;
                # we only care that the routing decision was logged
                # before construction.
                pass

        env_log = [r for r in caplog.records if "source=env" in r.getMessage()]
        assert env_log, (
            "Env-forced LOCAL routing must emit a log line with "
            f"source=env. Captured: {[r.getMessage() for r in caplog.records]}"
        )

    def test_route_local_creates_local_detector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When YAML pins route=local even with URL set, the factory
        must produce a real local detector — not a RemoteDetectionMethod.
        Skipped when local deps aren't installed (HuggingFace transformers).
        """
        monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
        from vijil_dome.detectors.methods.remote_method import RemoteDetectionMethod
        from vijil_dome.guardrails.config_parser import create_detector_for_guard

        # privacy-presidio has the lightest local deps to exercise here,
        # but installing presidio is still a heavy lift in CI. We pick
        # a heuristic-only detector that has no model dep — flashtext
        # for prompt injection. It's NOT in REMOTE_DETECTORS, so AUTO
        # would already produce local; the assertion here is that
        # explicit route=local still works (and isn't accidentally
        # routed to a non-existent remote handler).
        try:
            detector = create_detector_for_guard(
                "flashtext-injection-detector",
                "security",
                {"route": "local"},
            )
        except ValueError:
            pytest.skip("flashtext-injection-detector not registered in this env")
        assert not isinstance(detector, RemoteDetectionMethod)
