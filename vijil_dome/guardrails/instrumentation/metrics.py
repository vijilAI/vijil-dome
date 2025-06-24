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

import time
from inspect import iscoroutinefunction
from opentelemetry.metrics import Meter, Counter, Histogram
from functools import wraps
from typing import Union, Callable, Coroutine
from vijil_dome.guardrails import GuardResult, GuardrailResult
from vijil_dome.detectors import DetectionTimingResult


def _return_wrapper(
    func: Union[Callable, Coroutine], sync_wrapper: Callable, async_wrapper: Coroutine
) -> Union[Callable, Coroutine]:
    # Pick the wrapper to use if the function is async or not
    if iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def _create_request_counter(name: str, meter: Meter):
    request_counter = meter.create_counter(
        f"{name}-requests",
        description=f"Number of requests to {name}",
    )
    return request_counter


# Add counter to track number of requests
def _add_request_counter(request_counter: Counter):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request_counter.add(1)
            result = func(*args, **kwargs)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            request_counter.add(1)
            result = await func(*args, **kwargs)
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator


def _create_latency_histogram(name: str, meter: Meter):
    request_latency = meter.create_histogram(
        f"{name}-latency",
        description=f"{name} latency",
        unit="seconds",
    )
    return request_latency


# Add histogram to track latency of function
def _add_request_latency_histogram(request_latency: Histogram):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            request_latency.record(time.time() - start_time)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            request_latency.record(time.time() - start_time)
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator


def _create_request_flagged_counter(name: str, meter: Meter):
    request_flagged_counter = meter.create_counter(
        f"{name}-flagged",
        description=f"Number of requests to {name} that are flagged",
    )
    return request_flagged_counter


def _set_result_counter(
    result: Union[GuardrailResult, GuardResult, DetectionTimingResult],
    request_flagged_counter: Counter,
) -> None:
    if isinstance(result, GuardrailResult):
        if result.flagged:
            request_flagged_counter.add(1)
    elif isinstance(result, GuardResult):
        if result.triggered:
            request_flagged_counter.add(1)
    elif isinstance(result, DetectionTimingResult):
        if result.hit:
            request_flagged_counter.add(1)


# Add counter to track number of requests flagged
def _add_request_flagged_counter(request_flagged_counter: Counter):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            _set_result_counter(result, request_flagged_counter)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            _set_result_counter(result, request_flagged_counter)
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator


def _create_request_error_counter(name: str, meter: Meter):
    request_error_counter = meter.create_counter(
        f"{name}-error",
        description=f"Number of requests to {name} that errored",
    )
    return request_error_counter


def _add_request_error_counter(request_error_counter: Counter):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except:
                request_error_counter.add(1)
                raise
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
            except:
                request_error_counter.add(1)
                raise
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator
