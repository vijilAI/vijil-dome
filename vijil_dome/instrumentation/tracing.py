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

from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Optional, Union

from pydantic import BaseModel
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import Tracer
from opentelemetry.trace.span import Span

from vijil_dome.guardrails import GuardResult, GuardrailResult
from vijil_dome.types import DomePayload

# OpenInference semantic-convention names (github.com/Arize-ai/openinference),
# kept as plain strings so Dome needs no extra dependency to emit them.
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
GUARDRAIL_SPAN_KIND = "GUARDRAIL"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
METADATA = "metadata"
SESSION_ID = "session.id"
USER_ID = "user.id"

# OTEL AttributeValue — the union of types span.set_attribute accepts.
_AttributeValue = Union[str, bool, int, float, list]

_MAX_ATTR_LENGTH = 2048


def _safe_set_attribute(span: Span, key: str, value: _AttributeValue | None) -> None:
    """Set a span attribute only when the value is not None.

    The OTLP protobuf encoder rejects None values, crashing the entire
    span batch export. This guard prevents one bad attribute from
    dropping all traces in a BatchSpanProcessor flush cycle.
    """
    if value is None:
        return
    span.set_attribute(key, value)


def _truncate(value: str) -> str:
    if len(value) <= _MAX_ATTR_LENGTH:
        return value
    _MARKER = "...<truncated>"
    return value[:_MAX_ATTR_LENGTH - len(_MARKER)] + _MARKER


def _set_func_span_attributes(span: Span, *args, **kwargs):
    _safe_set_attribute(span, "function.args", _truncate(str(args)))
    _safe_set_attribute(span, "function.kwargs", _truncate(str(kwargs)))

    agent_id = kwargs.get("agent_id")
    if agent_id:
        _safe_set_attribute(span, "agent.id", str(agent_id))
    team_id = kwargs.get("team_id")
    if team_id:
        _safe_set_attribute(span, "team.id", str(team_id))
    _set_context_id_attributes(span, kwargs.get("user_id"))


def _set_context_id_attributes(span: Span, user_id: Optional[str]) -> None:
    """Set session.id and user.id, falling back to the OTEL context.

    OpenInference's ``using_session`` and ``using_user`` store these ids in
    the current context under the attribute names, so an app that wraps a
    Dome call in them gets the ids on Dome's spans. An explicit ``user_id``
    argument wins over the context value.
    """
    session_id = otel_context.get_value(SESSION_ID)
    if session_id:
        _safe_set_attribute(span, SESSION_ID, str(session_id))
    resolved_user_id = user_id or otel_context.get_value(USER_ID)
    if resolved_user_id:
        _safe_set_attribute(span, USER_ID, str(resolved_user_id))


def _payload_text(value: Any) -> Optional[str]:
    if isinstance(value, DomePayload):
        text = value.query_string
    elif isinstance(value, str):
        text = value
    else:
        return None
    # A lone surrogate (e.g. from JSON "\ud800") makes the OTLP encoder drop
    # the attribute, so escape anything that is not valid UTF-8.
    return _truncate(text.encode("utf-8", "backslashreplace").decode("utf-8"))


def _scan_input_text(args: tuple, kwargs: dict) -> Optional[str]:
    """The text a scan inspects: its first argument, by position or name."""
    if args:
        return _payload_text(args[0])
    return _payload_text(kwargs.get("query_string", kwargs.get("dome_input")))


def _scan_output_text(result: Any) -> Optional[str]:
    """The text a scan passes on: the guarded response, or the detector's
    possibly masked response string."""
    if isinstance(result, GuardrailResult):
        return _payload_text(result.guardrail_response_message)
    if isinstance(result, GuardResult):
        return _payload_text(result.response)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return _payload_text(result[1].get("response_string"))
    return None


def _set_guardrail_span_input(span: Span, args: tuple, kwargs: dict) -> None:
    _safe_set_attribute(span, OPENINFERENCE_SPAN_KIND, GUARDRAIL_SPAN_KIND)
    _safe_set_attribute(span, INPUT_VALUE, _scan_input_text(args, kwargs))


def _set_guardrail_span_output(span: Span, result: Any) -> None:
    _safe_set_attribute(span, OUTPUT_VALUE, _scan_output_text(result))


def _set_func_span_result_attributes(span: Span, result):
    if result is None:
        return
    if isinstance(result, BaseModel):
        _safe_set_attribute(span, "function.result", _truncate(str(result.model_dump())))
    else:
        _safe_set_attribute(span, "function.result", _truncate(str(result)))


# Wrap any function with a Tracer to record Spans. Works with both sync and async functions
# Automatically captures function arguments and outputs. guardrail=True also
# records the OpenInference GUARDRAIL kind and the scan's input and output text.
def auto_trace(tracer: Tracer, name: str, guardrail: bool = False):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Add input arguments to the span
                _set_func_span_attributes(span, *args, **kwargs)
                if guardrail:
                    _set_guardrail_span_input(span, args, kwargs)
                # Execute the function
                result = func(*args, **kwargs)
                # set function output result in span
                _set_func_span_result_attributes(span, result)
                if guardrail:
                    _set_guardrail_span_output(span, result)
                return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Add input arguments to the span
                _set_func_span_attributes(span, *args, **kwargs)
                if guardrail:
                    _set_guardrail_span_input(span, args, kwargs)
                # Execute the function
                result = await func(*args, **kwargs)
                # set function output result in span
                _set_func_span_result_attributes(span, result)
                if guardrail:
                    _set_guardrail_span_output(span, result)
                return result

        # Pick the wrapper to use if the function is async or not
        if iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
