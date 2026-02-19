"""Tests for DomeA2AMiddleware.

Covers:
- Input scanning: blocked and clean requests
- Output scanning: blocked and clean JSON responses
- SSE streaming responses bypass output scanning
- Configurable scan_timeout
- Non-A2A traffic passes through unmodified
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vijil_dome.integrations.a2a.middleware import (
    DomeA2AMiddleware,
    extract_a2a_message,
    a2a_blocked_response,
    DEFAULT_BLOCKED_MESSAGE,
)


# --- Unit tests for pure helpers ---


class TestExtractA2AMessage:
    def test_extracts_text_from_message_send(self):
        body = {
            "method": "message/send",
            "params": {
                "message": {
                    "parts": [{"type": "text", "text": "hello"}]
                }
            },
        }
        assert extract_a2a_message(body) == "hello"

    def test_concatenates_multiple_parts(self):
        body = {
            "method": "tasks/send",
            "params": {
                "message": {
                    "parts": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ]
                }
            },
        }
        assert extract_a2a_message(body) == "hello world"

    def test_returns_none_for_non_message_method(self):
        body = {"method": "tasks/get", "params": {}}
        assert extract_a2a_message(body) is None

    def test_returns_none_when_no_text_parts(self):
        body = {
            "method": "message/send",
            "params": {"message": {"parts": [{"type": "file"}]}},
        }
        assert extract_a2a_message(body) is None

    def test_returns_none_for_missing_params(self):
        body = {"method": "message/send"}
        assert extract_a2a_message(body) is None


class TestA2ABlockedResponse:
    def test_structure(self):
        resp = a2a_blocked_response("req-1", "blocked")
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-1"
        assert resp["result"]["status"]["state"] == "completed"
        assert resp["result"]["status"]["message"]["role"] == "agent"

    def test_uses_default_message(self):
        resp = a2a_blocked_response(None)
        text = resp["result"]["status"]["message"]["parts"][0]["text"]
        assert text == DEFAULT_BLOCKED_MESSAGE

    def test_echoes_request_id(self):
        resp = a2a_blocked_response("my-id")
        assert resp["id"] == "my-id"


# --- Middleware integration tests ---


def _make_dome(input_flagged: bool = False, output_flagged: bool = False):
    """Build a mock Dome with controllable guardrail outcomes."""
    dome = MagicMock()

    input_scan = MagicMock(flagged=input_flagged)
    dome.input_guardrail = MagicMock()
    dome.input_guardrail.async_scan = AsyncMock(return_value=input_scan)

    output_scan = MagicMock(flagged=output_flagged)
    dome.output_guardrail = MagicMock()
    dome.output_guardrail.async_scan = AsyncMock(return_value=output_scan)

    return dome


def _make_request(method: str = "POST", content_type: str = "application/json", body: dict | None = None):
    request = MagicMock()
    request.method = method
    request.headers = {"content-type": content_type}
    raw = json.dumps(body or {}).encode()
    request.body = AsyncMock(return_value=raw)
    return request


def _make_response(body: dict, status_code: int = 200, content_type: str = "application/json"):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    response.media_type = content_type
    encoded = json.dumps(body).encode()

    async def body_iterator():
        yield encoded

    response.body_iterator = body_iterator()
    return response


@pytest.mark.asyncio
async def test_non_post_passes_through():
    dome = _make_dome()
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome)
    request = _make_request(method="GET")
    call_next = AsyncMock(return_value=MagicMock())

    await middleware.dispatch(request, call_next)

    call_next.assert_awaited_once()
    dome.input_guardrail.async_scan.assert_not_called()


@pytest.mark.asyncio
async def test_non_json_content_type_passes_through():
    dome = _make_dome()
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome)
    request = _make_request(content_type="text/plain")
    call_next = AsyncMock(return_value=MagicMock())

    await middleware.dispatch(request, call_next)

    call_next.assert_awaited_once()
    dome.input_guardrail.async_scan.assert_not_called()


@pytest.mark.asyncio
async def test_non_a2a_method_skips_input_scan():
    dome = _make_dome()
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome)
    body = {"method": "tasks/get", "id": "1"}
    request = _make_request(body=body)

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    await middleware.dispatch(request, call_next)

    dome.input_guardrail.async_scan.assert_not_called()


@pytest.mark.asyncio
async def test_clean_input_passes_through():
    dome = _make_dome(input_flagged=False, output_flagged=False)
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome)
    body = {
        "method": "message/send",
        "id": "1",
        "params": {"message": {"parts": [{"type": "text", "text": "hello"}]}},
    }
    request = _make_request(body=body)
    downstream_response = _make_response({"result": "ok"})
    call_next = AsyncMock(return_value=downstream_response)

    result = await middleware.dispatch(request, call_next)

    dome.input_guardrail.async_scan.assert_awaited_once()
    call_next.assert_awaited_once()


@pytest.mark.asyncio
async def test_flagged_input_returns_blocked_response():
    dome = _make_dome(input_flagged=True)
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome)
    body = {
        "method": "message/send",
        "id": "req-42",
        "params": {"message": {"parts": [{"type": "text", "text": "jailbreak attempt"}]}},
    }
    request = _make_request(body=body)
    call_next = AsyncMock()

    result = await middleware.dispatch(request, call_next)

    call_next.assert_not_awaited()
    assert isinstance(result, type(result))  # JSONResponse
    response_body = json.loads(result.body)
    assert response_body["id"] == "req-42"
    assert response_body["result"]["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_flagged_output_returns_blocked_response():
    dome = _make_dome(input_flagged=False, output_flagged=True)
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome)
    body = {
        "method": "message/send",
        "id": "req-99",
        "params": {"message": {"parts": [{"type": "text", "text": "safe input"}]}},
    }
    request = _make_request(body=body)
    downstream_response = _make_response({"result": "unsafe agent output"})
    call_next = AsyncMock(return_value=downstream_response)

    result = await middleware.dispatch(request, call_next)

    dome.output_guardrail.async_scan.assert_awaited_once()
    response_body = json.loads(result.body)
    assert response_body["result"]["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_sse_response_skips_output_scan():
    dome = _make_dome(input_flagged=False)
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome)
    body = {
        "method": "tasks/sendSubscribe",
        "id": "1",
        "params": {"message": {"parts": [{"type": "text", "text": "stream this"}]}},
    }
    request = _make_request(body=body)

    sse_response = MagicMock()
    sse_response.status_code = 200
    sse_response.headers = {"content-type": "text/event-stream"}
    call_next = AsyncMock(return_value=sse_response)

    await middleware.dispatch(request, call_next)

    dome.output_guardrail.async_scan.assert_not_called()


@pytest.mark.asyncio
async def test_configurable_scan_timeout():
    dome = _make_dome()
    middleware = DomeA2AMiddleware(MagicMock(), dome=dome, scan_timeout=5.0)
    assert middleware.scan_timeout == 5.0


@pytest.mark.asyncio
async def test_input_timeout_passes_through():
    dome = MagicMock()
    dome.input_guardrail = MagicMock()
    dome.input_guardrail.async_scan = AsyncMock(side_effect=Exception("timeout"))
    dome.output_guardrail = None

    middleware = DomeA2AMiddleware(MagicMock(), dome=dome, scan_timeout=0.001)
    body = {
        "method": "message/send",
        "id": "1",
        "params": {"message": {"parts": [{"type": "text", "text": "hello"}]}},
    }
    request = _make_request(body=body)

    response = MagicMock()
    response.headers = {}
    call_next = AsyncMock(return_value=response)

    import asyncio

    async def slow_scan(*args, **kwargs):
        await asyncio.sleep(10)
        return MagicMock(flagged=False)

    dome.input_guardrail.async_scan = slow_scan

    result = await middleware.dispatch(request, call_next)
    call_next.assert_awaited_once()
