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
from typing import Union, Callable, Coroutine, Optional
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
        f"{name}-requests_total",
        description=f"Number of requests to {name}",
    )
    return request_counter


# Add counter to track number of requests
def _add_request_counter(request_counter: Counter):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            request_counter.add(1, attributes=attributes or {})
            result = func(*args, **kwargs)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            request_counter.add(1, attributes=attributes or {})
            result = await func(*args, **kwargs)
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator


def _create_latency_histogram(name: str, meter: Meter):
    bucket_advisory = [0.05 * i for i in range(21)]
    bucket_advisory = bucket_advisory + [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 10]
    request_latency = meter.create_histogram(
        f"{name}-latency_seconds",
        description=f"{name} latency",
        unit="seconds",
        explicit_bucket_boundaries_advisory=bucket_advisory,
    )
    return request_latency


# Add histogram to track latency of function
def _add_request_latency_histogram(request_latency: Histogram):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            start_time = time.time()
            result = func(*args, **kwargs)
            hist_time = time.time() - start_time
            request_latency.record(hist_time, attributes=attributes or {})
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            start_time = time.time()
            result = await func(*args, **kwargs)
            hist_time = time.time() - start_time
            request_latency.record(hist_time, attributes=attributes or {})
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator


def _create_request_flagged_counter(name: str, meter: Meter):
    request_flagged_counter = meter.create_counter(
        f"{name}-flagged_total",
        description=f"Number of requests to {name} that are flagged",
    )
    return request_flagged_counter


def _set_result_counter(
    result: Union[GuardrailResult, GuardResult, DetectionTimingResult],
    request_flagged_counter: Counter,
    attributes: Optional[dict] = None,
) -> None:
    if isinstance(result, GuardrailResult):
        if result.flagged:
            request_flagged_counter.add(1, attributes=attributes or {})
    elif isinstance(result, GuardResult):
        if result.triggered:
            request_flagged_counter.add(1, attributes=attributes or {})
    elif isinstance(result, DetectionTimingResult):
        if result.hit:
            request_flagged_counter.add(1, attributes=attributes or {})


# Add counter to track number of requests flagged
def _add_request_flagged_counter(request_flagged_counter: Counter):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            result = func(*args, **kwargs)
            _set_result_counter(result, request_flagged_counter, attributes)
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            result = await func(*args, **kwargs)
            _set_result_counter(result, request_flagged_counter, attributes)
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator


def _create_request_error_counter(name: str, meter: Meter):
    request_error_counter = meter.create_counter(
        f"{name}-error_total",
        description=f"Number of requests to {name} that errored",
    )
    return request_error_counter


def _add_request_error_counter(request_error_counter: Counter):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            try:
                result = func(*args, **kwargs)
            except:
                request_error_counter.add(1, attributes=attributes or {})
                raise
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            agent_id = kwargs.get("agent_id")
            attributes = {"agent.id": agent_id} if agent_id else None
            try:
                result = await func(*args, **kwargs)
            except:
                request_error_counter.add(1, attributes=attributes or {})
                raise
            return result

        return _return_wrapper(func, sync_wrapper, async_wrapper)

    return decorator
