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

"""Dome Strands Integration -- HookProvider for Strands agents.

Registers hooks that scan model input and output through Dome's
guardrails. Flagged content is replaced with a blocked message.

Usage:
    from vijil_dome import Dome
    from vijil_dome.integrations.strands import DomeHookProvider

    dome = Dome(config)
    agent = Agent(hooks=[DomeHookProvider(dome, agent_id="...", team_id="...")])
"""

import logging
from typing import Any, Optional

from vijil_dome import Dome

try:
    from strands.hooks import (
        AfterModelCallEvent,
        BeforeModelCallEvent,
        HookProvider,
        HookRegistry,
    )
except ImportError:
    raise RuntimeError(
        "Strands SDK is not installed. "
        "Install it with: pip install strands-agents"
    )

logger = logging.getLogger(__name__)

DEFAULT_INPUT_BLOCKED_MESSAGE = (
    "I'm sorry, but I can't help with that request. "
    "It appears to contain content that violates my usage policies. "
    "I'd be happy to assist you with a different question."
)

DEFAULT_OUTPUT_BLOCKED_MESSAGE = (
    "The generated response was blocked by the guardrail policy. "
    "I'd be happy to assist you with a different request."
)


def _extract_last_user_text(messages: list[dict]) -> tuple[Optional[str], Optional[int]]:
    """Find the last user message and extract its text.

    Returns:
        (text, index) if found, (None, None) if no user message exists.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        parts = msg.get("content", [])
        texts = [
            block["text"]
            for block in parts
            if isinstance(block, dict) and "text" in block
        ]
        if texts:
            return " ".join(texts), i
    return None, None


def _extract_response_text(message: dict) -> Optional[str]:
    """Extract text from a model response message."""
    parts = message.get("content", [])
    texts = [
        block["text"]
        for block in parts
        if isinstance(block, dict) and "text" in block
    ]
    return " ".join(texts) if texts else None


class DomeHookProvider(HookProvider):
    """Dome guardrail integration for Strands agents.

    Registers hooks for BeforeModelCallEvent (input) and
    AfterModelCallEvent (output) that scan content through Dome's
    guardrails and replace flagged content with a blocked message.
    """

    def __init__(
        self,
        dome: Dome,
        agent_id: str = "",
        team_id: str = "",
        input_blocked_message: str = DEFAULT_INPUT_BLOCKED_MESSAGE,
        output_blocked_message: str = DEFAULT_OUTPUT_BLOCKED_MESSAGE,
    ):
        self.dome = dome
        self.agent_id = agent_id
        self.team_id = team_id
        self.input_blocked_message = input_blocked_message
        self.output_blocked_message = output_blocked_message

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._guard_input)
        registry.add_callback(AfterModelCallEvent, self._guard_output)

    async def _guard_input(self, event: BeforeModelCallEvent) -> None:
        """Scan the last user message; replace if flagged."""
        text, idx = _extract_last_user_text(event.agent.messages)
        if text is None or idx is None:
            return

        scan = await self.dome.async_guard_input(text, agent_id=self.agent_id)
        if scan.flagged:
            logger.warning("Dome blocked input: %s...", text[:80])
            event.agent.messages[idx]["content"] = [
                {"text": self.input_blocked_message}
            ]

    async def _guard_output(self, event: AfterModelCallEvent) -> None:
        """Scan the model response; replace if flagged."""
        if event.stop_response is None:
            return

        text = _extract_response_text(event.stop_response.message)
        if not text:
            return

        scan = await self.dome.async_guard_output(text, agent_id=self.agent_id)
        if scan.flagged:
            logger.warning("Dome blocked output: %s...", text[:80])
            event.stop_response.message["content"] = [
                {"text": self.output_blocked_message}
            ]
