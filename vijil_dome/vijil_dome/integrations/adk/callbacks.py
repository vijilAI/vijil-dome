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

"""
These functions enable easy compatibility with Google ADK
"""

from typing import Callable, Optional
from vijil_dome import Dome

try:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
except ImportError:
    raise RuntimeError(
        "Google ADK is not installed. " "Install ADK with `pip install google-adk`."
    )

INPUT_BLOCKED_MESSAGE = "This request was blocked by Vijil Dome's guardrail policy. I cannot respond to this request."
OUTPUT_BLOCKED_MESSAGE = "The generated response was blocked by Vijil Dome's guardrail policy. I cannot respond to this request."

def generate_adk_input_callback(
    dome: Dome,
    blocked_message: Optional[str] = None,
    additional_callback: Optional[Callable] = None,
) -> dict:
    """
    Given an instance of Dome, a blocked message, and any optional additional callbacks, generate a callback to guard the inputs to the agent.
    """
    input_blocked_message = blocked_message or INPUT_BLOCKED_MESSAGE

    def guard_input(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        # Extract the user's latest message
        last_user_message_text = ""
        if llm_request.contents:
            # Find the most recent message with role 'user'
            for content in reversed(llm_request.contents):
                if content.role == "user" and content.parts:
                    for part in content.parts:
                        last_user_message_text += part.text or ""
                    break

        # Pass the message through dome
        if last_user_message_text:
            # Unfortunately, ADK callbacks don't support async yet
            scan = dome.guard_input(last_user_message_text)
            if not scan.is_safe():
                return LlmResponse(
                    content=types.Content(
                        role="model",  # Mimic an agent response
                        parts=[types.Part(text=input_blocked_message)],
                    )
                )

        if additional_callback:
            # Call the additional callback if provided
            return additional_callback(callback_context, llm_request)
        return None

    return guard_input


def generate_adk_output_callback(
    dome: Dome,
    blocked_message: Optional[str] = None,
    additional_callback: Optional[Callable] = None,
) -> dict:
    """
    Given an instance of Dome, a blocked message, and any optional additional callbacks, generate a callback to guard the outputs from the agent.
    """
    output_blocked_message = blocked_message or OUTPUT_BLOCKED_MESSAGE

    def guard_output(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        # Extract the agent's message
        last_model_message_text = ""
        if llm_response.content:
            if (
                llm_response.content
                and llm_response.content.role == "model"
                and llm_response.content.parts
            ):
                for part in llm_response.content.parts:
                    if part.text:
                        last_model_message_text += part.text or ""

        # Pass the message through dome
        if last_model_message_text:
            scan = dome.guard_output(last_model_message_text)
            if not scan.is_safe():
                return LlmResponse(
                    content=types.Content(
                        role="model",  # Mimic an agent response
                        parts=[types.Part(text=output_blocked_message)],
                    )
                )

        if additional_callback:
            # Call the additional callback if provided
            return additional_callback(callback_context, llm_response)
        return None

    return guard_output
