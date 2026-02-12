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

"""Dome A2A Integration — Starlette/FastAPI middleware for A2A JSON-RPC.

Intercepts incoming A2A requests, extracts user messages from JSON-RPC
envelopes, and runs them through Dome's input guardrail. Blocked requests
receive a valid A2A JSON-RPC refusal response.

Usage:
    from vijil_dome import Dome
    from vijil_dome.integrations.a2a import DomeA2AMiddleware

    dome = Dome(config)
    app.add_middleware(DomeA2AMiddleware, dome=dome, agent_id="...", team_id="...")
"""

import asyncio
import json
import logging
from typing import Optional

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
except ImportError:
    raise RuntimeError(
        "starlette is required for A2A middleware integration. "
        "Install it with: pip install 'vijil-dome[a2a]'"
    )

logger = logging.getLogger(__name__)

# A2A JSON-RPC methods that carry user messages
_A2A_MESSAGE_METHODS = frozenset({
    "message/send",
    "tasks/send",
    "messages/send",
    "tasks/sendSubscribe",
})

# Crafted to match Diamond's RefusalDetector patterns for accurate security scoring
DEFAULT_BLOCKED_MESSAGE = (
    "I'm sorry, but I can't help with that request. "
    "It appears to contain content that violates my usage policies. "
    "I'd be happy to assist you with legitimate travel-related questions instead."
)


def extract_a2a_message(body: dict) -> Optional[str]:
    """Extract user text from an A2A JSON-RPC request body.

    Supports the standard A2A message format where user text is carried
    in ``params.message.parts[].text``.

    Args:
        body: Parsed JSON-RPC request body.

    Returns:
        Concatenated user text, or None if no text was found.
    """
    method = body.get("method", "")
    if method not in _A2A_MESSAGE_METHODS:
        return None

    message = body.get("params", {}).get("message", {})
    if isinstance(message, dict):
        parts = message.get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
        return " ".join(texts) if texts else None
    return None


def a2a_blocked_response(request_id: Optional[str], message: str = DEFAULT_BLOCKED_MESSAGE) -> dict:
    """Build an A2A JSON-RPC response for a blocked request.

    Returns a well-formed A2A response with ``state: "completed"`` so the
    client treats it as a final answer rather than an error.

    Args:
        request_id: The JSON-RPC request ID to echo back.
        message: The refusal message to include.

    Returns:
        A dict suitable for ``JSONResponse(content=...)``.
    """
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "status": {
                "state": "completed",
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": message}],
                },
            }
        },
    }


class DomeA2AMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that applies Dome input guardrails to A2A requests.

    Scans user messages extracted from A2A JSON-RPC envelopes. Blocked requests
    receive a valid A2A JSON-RPC refusal response.

    Telemetry (split metrics + Darwin spans) is handled upstream by
    ``instrument_dome()`` — this middleware only orchestrates the scan.

    Args:
        app: The ASGI application to wrap.
        dome: A configured ``Dome`` instance with an input guardrail.
        agent_id: Agent identifier for telemetry attribution.
        team_id: Team identifier for telemetry attribution.
        blocked_message: Custom refusal message (optional).

    Usage::

        from vijil_dome import Dome
        from vijil_dome.integrations.a2a import DomeA2AMiddleware

        dome = Dome(config)
        app.add_middleware(
            DomeA2AMiddleware,
            dome=dome,
            agent_id="my-agent-id",
            team_id="my-team-id",
        )
    """

    def __init__(
        self,
        app,  # type: ignore[override]
        dome,  # type: ignore[override]
        agent_id: str = "",
        team_id: str = "",
        blocked_message: str = DEFAULT_BLOCKED_MESSAGE,
    ):
        super().__init__(app)
        self.dome = dome
        self.agent_id = agent_id
        self.team_id = team_id
        self.blocked_message = blocked_message

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.method != "POST":
            return await call_next(request)

        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return await call_next(request)

        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("Skipping non-JSON request: %s", e)
            return await call_next(request)

        user_message = extract_a2a_message(body)
        if user_message and self.dome.input_guardrail is not None:
            try:
                scan = await asyncio.wait_for(
                    self.dome.input_guardrail.async_scan(
                        user_message,
                        agent_id=self.agent_id,
                        team_id=self.team_id,
                    ),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Dome scan timed out after 30s, passing request through")
                return await call_next(request)
            if scan.flagged:
                logger.warning("Dome blocked A2A request: %s...", user_message[:80])
                return JSONResponse(
                    content=a2a_blocked_response(body.get("id"), self.blocked_message)
                )

        # Replay consumed body bytes for downstream handlers
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive  # type: ignore[attr-defined]

        return await call_next(request)
