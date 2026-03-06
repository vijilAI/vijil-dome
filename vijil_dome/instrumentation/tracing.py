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
from pydantic import BaseModel
from opentelemetry.sdk.trace import Tracer, Span


def _safe_set_attribute(span: Span, key: str, value) -> None:
    """Set a span attribute only if the value is a valid OTEL type (not None).

    The OTLP protobuf encoder rejects None values, crashing the entire
    span batch export. This guard prevents one bad attribute from
    dropping all traces in a BatchSpanProcessor flush cycle.
    """
    if value is None:
        return
    span.set_attribute(key, value)


def _set_func_span_attributes(span: Span, *args, **kwargs):
    _safe_set_attribute(span, "function.args", str(args))
    _safe_set_attribute(span, "function.kwargs", str(kwargs))

    agent_id = kwargs.get("agent_id")
    if agent_id:
        _safe_set_attribute(span, "agent.id", str(agent_id))


def _set_func_span_result_attributes(span: Span, result):
    if result is None:
        return
    if isinstance(result, BaseModel):
        _safe_set_attribute(span, "function.result", str(result.model_dump()))
    else:
        _safe_set_attribute(span, "function.result", str(result))


# Wrap any function with a Tracer to record Spans. Works with both sync and async functions
# Automatically captures function arguments and outputs
def auto_trace(tracer: Tracer, name: str):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Add input arguments to the span
                _set_func_span_attributes(span, *args, **kwargs)
                # Execute the function
                result = func(*args, **kwargs)
                # set function output result in span
                _set_func_span_result_attributes(span, result)
                return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Add input arguments to the span
                _set_func_span_attributes(span, *args, **kwargs)
                # Execute the function
                result = await func(*args, **kwargs)
                # set function output result in span
                _set_func_span_result_attributes(span, result)
                return result

        # Pick the wrapper to use if the function is async or not
        if iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
