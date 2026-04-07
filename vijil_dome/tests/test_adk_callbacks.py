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

"""Tests for Dome ADK callback integration."""

import pytest
from unittest.mock import MagicMock

pytest.importorskip("google.adk", reason="google-adk not installed")

from vijil_dome.integrations.adk.callbacks import (
    generate_adk_input_callback,
    generate_adk_output_callback,
)
from vijil_dome.types import DomePayload


def _make_llm_request(user_text: str):
    """Build a mock LlmRequest with a single user message."""
    part = MagicMock()
    part.text = user_text
    content = MagicMock()
    content.role = "user"
    content.parts = [part]
    req = MagicMock()
    req.contents = [content]
    return req


def _make_llm_response(model_text: str):
    """Build a mock LlmResponse with model content."""
    part = MagicMock()
    part.text = model_text
    content = MagicMock()
    content.role = "model"
    content.parts = [part]
    resp = MagicMock()
    resp.content = content
    return resp


class TestAdkInputCallback:
    def test_safe_input_passes_through(self):
        dome = MagicMock()
        scan = MagicMock()
        scan.flagged = False
        scan.enforced = False
        dome.guard_input.return_value = scan

        callback = generate_adk_input_callback(dome)
        ctx = MagicMock()
        req = _make_llm_request("hello")

        result = callback(ctx, req)
        assert result is None
        dome.guard_input.assert_called_once()

    def test_flagged_enforced_input_returns_blocked(self):
        dome = MagicMock()
        scan = MagicMock()
        scan.flagged = True
        scan.enforced = True
        dome.guard_input.return_value = scan

        callback = generate_adk_input_callback(dome)
        ctx = MagicMock()
        req = _make_llm_request("bad input")

        result = callback(ctx, req)
        assert result is not None

    def test_input_callback_passes_string_to_dome(self):
        dome = MagicMock()
        scan = MagicMock()
        scan.flagged = False
        scan.enforced = False
        dome.guard_input.return_value = scan

        callback = generate_adk_input_callback(dome)
        ctx = MagicMock()
        req = _make_llm_request("test input")

        callback(ctx, req)
        call_args = dome.guard_input.call_args
        assert call_args[0][0] == "test input"


class TestAdkOutputCallback:
    def test_safe_output_passes_through(self):
        dome = MagicMock()
        scan = MagicMock()
        scan.flagged = False
        scan.enforced = False
        dome.guard_output.return_value = scan

        callback = generate_adk_output_callback(dome)
        ctx = MagicMock()
        resp = _make_llm_response("safe response")

        result = callback(ctx, resp)
        assert result is None

    def test_output_callback_sends_dome_payload(self):
        dome = MagicMock()
        scan = MagicMock()
        scan.flagged = False
        scan.enforced = False
        dome.guard_output.return_value = scan

        callback = generate_adk_output_callback(dome)
        ctx = MagicMock()
        resp = _make_llm_response("model response text")

        callback(ctx, resp)
        call_args = dome.guard_output.call_args
        payload = call_args[0][0]
        assert isinstance(payload, DomePayload)
        assert payload.response == "model response text"

    def test_flagged_enforced_output_returns_blocked(self):
        dome = MagicMock()
        scan = MagicMock()
        scan.flagged = True
        scan.enforced = True
        dome.guard_output.return_value = scan

        callback = generate_adk_output_callback(dome)
        ctx = MagicMock()
        resp = _make_llm_response("bad output")

        result = callback(ctx, resp)
        assert result is not None
