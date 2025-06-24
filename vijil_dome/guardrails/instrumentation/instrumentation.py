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

from vijil_dome.guardrails import DetectionMethod, Guard, Guardrail
from opentelemetry.sdk.trace import Tracer
from opentelemetry.metrics import Meter, Instrument
from typing import Optional, Union, Callable, Coroutine, List
from vijil_dome.instrumentation.tracing import auto_trace
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from vijil_dome.guardrails.instrumentation.metrics import (
    _create_request_counter,
    _create_request_error_counter,
    _create_request_flagged_counter,
    _create_latency_histogram,
    _add_request_counter,
    _add_request_error_counter,
    _add_request_flagged_counter,
    _add_request_latency_histogram,
)


def _instrument_func_with_monitors(
    func: Union[Callable, Coroutine],
    instruments: List[Instrument],
    wrappers: List[Callable],
):
    decorated_function = func
    for instrument, wrapper in zip(instruments, wrappers):
        decorated_function = wrapper(instrument)(decorated_function)
    return decorated_function


def _add_instrumentation_to_scan(
    obj: Union[Guardrail, Guard, DetectionMethod],
    name: str,
    meter: Meter,
    instrument_generators: List[Callable],
    wrappers: List[Callable],
) -> None:
    instruments = []
    for generator in instrument_generators:
        instruments.append(generator(name, meter))

    if isinstance(obj, Guardrail) or isinstance(obj, Guard):
        obj.scan = _instrument_func_with_monitors(obj.scan, instruments, wrappers)  # type: ignore [method-assign]
        obj.async_scan = _instrument_func_with_monitors(  # type: ignore [method-assign]
            obj.async_scan, instruments, wrappers
        )  # type: ignore [method-assign]

    elif isinstance(obj, DetectionMethod):
        obj.detect = _instrument_func_with_monitors(obj.detect, instruments, wrappers)  # type: ignore [method-assign]
        if hasattr(obj, "sync_detect") and callable(obj.sync_detect):
            obj.sync_detect = _instrument_func_with_monitors(
                obj.sync_detect, instruments, wrappers
            )  # type: ignore [method-assign]


def instrument_with_monitors(
    guardrail: Guardrail, meter: Meter, guardrail_name: Optional[str] = None
) -> None:
    # These control the monitoring instruments added to the scanning function
    # The generators specify functions for how to create the instrument
    # The wrapper is a decorator that uses the instrument in the code
    # New monitors can be set up by creating a new instrument and a new corresponding wrapper for it
    instrument_generators = [
        _create_request_counter,
        _create_request_error_counter,
        _create_request_flagged_counter,
        _create_latency_histogram,
    ]  # type: List[Callable]
    wrappers = [
        _add_request_counter,
        _add_request_error_counter,
        _add_request_flagged_counter,
        _add_request_latency_histogram,
    ]  # type: List[Callable]

    if not guardrail_name:
        guardrail_name = f"{guardrail.level}-guardrail"

    _add_instrumentation_to_scan(
        guardrail, guardrail_name, meter, instrument_generators, wrappers
    )
    for guard in guardrail.guard_list:
        guard_name = f"{guardrail_name}.{guard.guard_name}"
        _add_instrumentation_to_scan(
            guard, guard_name, meter, instrument_generators, wrappers
        )

        for detector in guard.detector_list:
            detector_name = f"{guard_name}.{detector.__class__.__name__}"
            _add_instrumentation_to_scan(
                detector, detector_name, meter, instrument_generators, wrappers
            )


def _add_tracing_to_scan(
    obj: Union[Guardrail, Guard, DetectionMethod], name: str, tracer: Tracer
) -> None:
    if isinstance(obj, Guardrail) or isinstance(obj, Guard):
        obj.scan = auto_trace(tracer, f"{name}.scan")(  # type: ignore[method-assign]
            obj.scan
        )
        obj.async_scan = auto_trace(tracer, f"{name}.async_scan")(  # type: ignore[method-assign]
            obj.async_scan
        )

    elif isinstance(obj, DetectionMethod):
        obj.detect = auto_trace(tracer, f"{name}.scan")(  # type: ignore[method-assign]
            obj.detect
        )
        if hasattr(obj, "sync_detect") and callable(obj.sync_detect):
            obj.sync_detect = auto_trace(tracer, f"{name}.scan")(  # type: ignore[method-assign]
                obj.sync_detect
            )


# Wrap scanning functions with a tracer
def instrument_with_tracer(
    guardrail: Guardrail, tracer: Tracer, guardrail_name: Optional[str] = None
) -> None:
    if not AsyncioInstrumentor().is_instrumented_by_opentelemetry:
        AsyncioInstrumentor().instrument()
    if not ThreadingInstrumentor().is_instrumented_by_opentelemetry:
        ThreadingInstrumentor().instrument()

    if not guardrail_name:
        guardrail_name = f"{guardrail.level}-guardrail"

    _add_tracing_to_scan(guardrail, guardrail_name, tracer)
    for guard in guardrail.guard_list:
        guard_name = f"{guardrail_name}.{guard.guard_name}"
        _add_tracing_to_scan(guard, guard_name, tracer)

        for detector in guard.detector_list:
            detector_name = f"{guard_name}.{detector.__class__.__name__}"
            _add_tracing_to_scan(detector, detector_name, tracer)
