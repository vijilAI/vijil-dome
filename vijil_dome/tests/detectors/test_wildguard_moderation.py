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
#
# vijil and vijil-dome are trademarks owned by Vijil Inc.

"""WildGuard moderation tests. The endpoint is mocked; no network or model is used."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vijil_dome import Dome
from vijil_dome.detectors import MODERATION_WILDGUARD, DetectionCategory, DetectionFactory
from vijil_dome.detectors.methods import wildguard_moderation as wg
from vijil_dome.detectors.methods.wildguard_moderation import (
    WILDGUARD_INSTRUCTION_FORMAT,
    WildGuardModeration,
    parse_wildguard_label,
)
from vijil_dome.guardrails import Guard
from vijil_dome.types import DomePayload

BASE_URL = "http://wildguard.test:8000/v1"

# sha256 of the instruction_format string in the allenai/wildguard model card
# (README.md at revision cbba4823f3e8020e5a74a5e29bf85072def6f2ff), from the
# opening <s> to the newline after <|assistant|>.
MODEL_CARD_TEMPLATE_SHA256 = "fa7f1c313ef1463a2fe249ff50c537adfbd84ff2fe82b3ffa72ee21124afb0d6"
MODEL_CARD_TEMPLATE_LEN = 415

CARD_EXAMPLE_OUTPUT = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no\n"


def _completion(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(text=text)])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("WILDGUARD_BASE_URL", "WILDGUARD_MODEL", "WILDGUARD_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _mock_endpoint(monkeypatch, **kwargs) -> AsyncMock:
    mock = AsyncMock(**kwargs)
    monkeypatch.setattr(wg, "atext_completion", mock)
    return mock


# --- fail closed ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["request", "response"])
async def test_endpoint_down_is_error_not_safe(monkeypatch, mode):
    _mock_endpoint(monkeypatch, side_effect=ConnectionError("connection refused"))
    hit, data = await WildGuardModeration(mode=mode, base_url=BASE_URL).detect(
        DomePayload(text="hello")
    )
    assert hit is False
    assert data["label"] == "error"
    assert "connection refused" in data["error"]


@pytest.mark.asyncio
async def test_endpoint_down_blocks_under_fail_closed_guard(monkeypatch):
    _mock_endpoint(monkeypatch, side_effect=ConnectionError("connection refused"))
    guard = Guard(
        "wg", [WildGuardModeration(mode="request", base_url=BASE_URL)], on_error="fail_closed"
    )
    result = await guard.sequential_guard("hello")
    assert result.triggered is True


@pytest.mark.parametrize("enforce, blocked", [(True, True), (False, False)])
def test_endpoint_down_dome_enforce_blocks(monkeypatch, enforce, blocked):
    """Enforce mode defaults content guards to fail_closed, so a dead endpoint blocks."""
    _mock_endpoint(monkeypatch, side_effect=ConnectionError("connection refused"))
    monkeypatch.setenv("WILDGUARD_BASE_URL", BASE_URL)
    dome = Dome(
        dome_config={
            "input-guards": ["wg-in"],
            "output-guards": [],
            "wg-in": {
                "type": "moderation",
                "methods": [MODERATION_WILDGUARD],
                MODERATION_WILDGUARD: {"mode": "request"},
            },
        },
        enforce=enforce,
    )
    assert dome.guard_input("hello").is_safe() is (not blocked)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, output",
    [
        ("request", ""),
        ("request", "I cannot classify this."),
        ("request", "Harmful request: maybe\nHarmful response: no"),
        ("request", "Response refusal: no\nHarmful response: no"),  # request label missing
        ("response", "Harmful request: no\nResponse refusal: yes\nHarmful response: N/A"),
        ("response", "Harmful request: yes\nResponse refusal: no"),  # response label missing
    ],
)
async def test_unparseable_output_is_error(monkeypatch, mode, output):
    _mock_endpoint(monkeypatch, return_value=_completion(output))
    hit, data = await WildGuardModeration(mode=mode, base_url=BASE_URL).detect("x")
    assert hit is False
    assert data["label"] == "error"
    assert data["model_response"] == output


@pytest.mark.asyncio
async def test_none_completion_text_is_error(monkeypatch):
    _mock_endpoint(monkeypatch, return_value=_completion(None))  # type: ignore[arg-type]
    hit, data = await WildGuardModeration(mode="request", base_url=BASE_URL).detect("x")
    assert (hit, data["label"]) == (False, "error")


# --- parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    "output, mode, expected",
    [
        (CARD_EXAMPLE_OUTPUT, "request", True),
        (CARD_EXAMPLE_OUTPUT, "response", False),
        ("Harmful request: no\nResponse refusal: no\nHarmful response: yes", "request", False),
        ("Harmful request: no\nResponse refusal: no\nHarmful response: yes", "response", True),
        (" harmful request:  YES \n", "request", True),
    ],
)
def test_parse_label(output, mode, expected):
    assert parse_wildguard_label(output, mode) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, output, expected_hit",
    [
        ("request", CARD_EXAMPLE_OUTPUT, True),
        ("response", CARD_EXAMPLE_OUTPUT, False),
        ("request", "Harmful request: no\nResponse refusal: no\nHarmful response: yes", False),
        ("response", "Harmful request: no\nResponse refusal: no\nHarmful response: yes", True),
    ],
)
async def test_detect_reads_the_mode_label(monkeypatch, mode, output, expected_hit):
    _mock_endpoint(monkeypatch, return_value=_completion(output))
    detector = WildGuardModeration(mode=mode, base_url=BASE_URL)
    hit, data = await detector.detect("some text")
    assert hit is expected_hit
    assert data["label"] == ("harmful" if expected_hit else "safe")
    assert data["response_string"] == (
        detector.blocked_response_string if expected_hit else "some text"
    )


# --- prompt construction ----------------------------------------------------


def test_template_matches_model_card_byte_for_byte():
    raw = WILDGUARD_INSTRUCTION_FORMAT.encode("utf-8")
    assert len(raw) == MODEL_CARD_TEMPLATE_LEN
    assert hashlib.sha256(raw).hexdigest() == MODEL_CARD_TEMPLATE_SHA256


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, payload, prompt, response",
    [
        ("request", DomePayload(text="the user turn"), "the user turn", ""),
        ("response", DomePayload(text="the agent turn"), "", "the agent turn"),
        ("response", DomePayload(prompt="p", response="r"), "p", "r"),
        ("request", DomePayload(prompt="p", response="r"), "p", "r"),
    ],
)
async def test_request_sent_to_endpoint(monkeypatch, mode, payload, prompt, response):
    mock = _mock_endpoint(monkeypatch, return_value=_completion(CARD_EXAMPLE_OUTPUT))
    await WildGuardModeration(mode=mode, base_url=BASE_URL, model_name="wg").detect(payload)
    kwargs = mock.await_args.kwargs
    assert kwargs["prompt"] == WILDGUARD_INSTRUCTION_FORMAT.format(
        prompt=prompt, response=response
    )
    assert kwargs["model"] == "text-completion-openai/wg"
    assert kwargs["api_base"] == BASE_URL
    assert kwargs["temperature"] == 0
    assert kwargs["extra_body"] == {"add_special_tokens": False}


# --- construction -----------------------------------------------------------


def test_requires_endpoint():
    with pytest.raises(ValueError, match="WILDGUARD_BASE_URL"):
        WildGuardModeration(mode="request")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        WildGuardModeration(mode="both", base_url=BASE_URL)  # type: ignore[arg-type]


def test_env_config_and_registry(monkeypatch):
    monkeypatch.setenv("WILDGUARD_BASE_URL", BASE_URL)
    monkeypatch.setenv("WILDGUARD_MODEL", "served-wildguard")
    detector = DetectionFactory.get_detector(
        DetectionCategory.Moderation, MODERATION_WILDGUARD, mode="response"
    )
    assert isinstance(detector, WildGuardModeration)
    assert detector.config == {"model": "served-wildguard", "base_url": BASE_URL, "mode": "response"}
