# Copyright 2025 Vijil, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import re

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from vijil_dome.guardrails.instrumentation.metrics import (
    _OTEL_INSTRUMENT_NAME_MAX_LEN,
    _instrument_name_for_otel,
    _create_request_counter,
)


def test_instrument_name_spaces_and_dots_normalized():
    name = "dome-input.Privacy Input Guard"
    out = _instrument_name_for_otel(name, "-requests_total")
    assert " " not in out
    assert len(out) <= _OTEL_INSTRUMENT_NAME_MAX_LEN
    assert out.endswith("-requests_total")


def test_instrument_name_truncates_very_long_prefix_with_hash():
    long_prefix = "dome-input." + "x" * 80 + ".Privacy Input Guard"
    out = _instrument_name_for_otel(long_prefix, "-latency_seconds")
    assert len(out) <= _OTEL_INSTRUMENT_NAME_MAX_LEN
    assert out.endswith("-latency_seconds")
    assert re.search(r"_[0-9a-f]{8}-latency_seconds$", out)


def test_sdk_accepts_counter_for_problematic_guard_name():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")

    logical = "dome-input.Privacy Input Guard"
    counter = _create_request_counter(logical, meter)
    counter.add(1)
    reader.collect()
