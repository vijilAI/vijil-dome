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

import asyncio
import random
from pydantic import BaseModel
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from vijil_dome.instrumentation.tracing import auto_trace


# Pydantic Model Example
class ExampleOutput(BaseModel):
    result: float
    description: str


def get_otel_tracer():
    default_tracer_provider = TracerProvider()
    span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
    default_tracer_provider.add_span_processor(span_processor)
    default_tracer = default_tracer_provider.get_tracer(__name__)
    return default_tracer


def example_division(x: int, y: int):
    if y == 0:
        raise ValueError("y cannot be 0")
    z = x / y
    return ExampleOutput(result=z, description=f"{x} divided by {y} is {z}")


async def async_example_division(x: int, y: int):
    await asyncio.sleep(random.random())
    return example_division(x, y)


if __name__ == "__main__":
    tracer = get_otel_tracer()

    # Replace the functions with their traced versions
    example_division = auto_trace(tracer, "eg-division")(example_division)  # type: ignore[method-assign]
    async_example_division = auto_trace(tracer, "eg-division-async")(
        async_example_division
    )  # type: ignore[method-assign]

    print("Sync Function Traces")
    example_division(100, 10)
    try:
        example_division(5, 0)
    except ValueError as e:
        print("Error", e)

    print("\n---------\nAsync Function Traces")
    asyncio.run(async_example_division(100, 10))
    try:
        asyncio.run(async_example_division(5, 0))
    except ValueError as e:
        print("Error", e)
