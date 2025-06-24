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


def _set_func_span_attributes(span: Span, *args, **kwargs):
    span.set_attribute("function.args", str(args))
    span.set_attribute("function.kwargs", str(kwargs))


def _set_func_span_result_attributes(span: Span, result):
    if isinstance(result, BaseModel):
        span.set_attribute("function.result", str(result.model_dump()))
    else:
        span.set_attribute("function.result", str(result))


# Wrap any function with a Tracer to record Spans. Works with both sync and async functions
# Automatically captures function arguments and outputs
def auto_trace(tracer: Tracer, name: str):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Add input arguments to the span
                _set_func_span_attributes(span, args, kwargs)
                # Execute the function
                result = func(*args, **kwargs)
                # set function output result in span
                _set_func_span_result_attributes(span, result)
                return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Add input arguments to the span
                _set_func_span_attributes(span, args, kwargs)
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
