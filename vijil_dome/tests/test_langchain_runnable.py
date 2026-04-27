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

"""Tests for Dome LangChain GuardrailRunnable integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock

pytest.importorskip("langchain_core", reason="langchain-core not installed")

from vijil_dome.integrations.langchain.runnable import GuardrailRunnable
from vijil_dome.types import DomePayload


def _make_guardrail_mock(flagged=False):
    guardrail = MagicMock()
    result = MagicMock()
    result.flagged = flagged
    result.guardrail_response_message = "blocked" if flagged else "safe"
    result.guard_exec_details = {}
    result.exec_time = 0.01
    result.detection_score = 1.0 if flagged else 0.0
    result.triggered_methods = ["MockDetector"] if flagged else []
    guardrail.scan.return_value = result
    guardrail.async_scan = AsyncMock(return_value=result)
    return guardrail


class TestGuardrailRunnableString:
    def test_invoke_with_string(self):
        guardrail = _make_guardrail_mock()
        runnable = GuardrailRunnable(guardrail=guardrail)
        output = runnable.invoke("hello")
        assert output["enforced"] is False
        guardrail.scan.assert_called_once()

    def test_invoke_with_dict_query(self):
        guardrail = _make_guardrail_mock()
        runnable = GuardrailRunnable(guardrail=guardrail)
        output = runnable.invoke({"query": "what is this?"})
        assert output["enforced"] is False


class TestGuardrailRunnableStructured:
    def test_invoke_with_dict_query_and_response(self):
        guardrail = _make_guardrail_mock()
        runnable = GuardrailRunnable(guardrail=guardrail)
        runnable.invoke({"query": "what?", "response": "this"})
        call_args = guardrail.scan.call_args
        payload = call_args[0][0]
        assert isinstance(payload, DomePayload)
        assert payload.prompt == "what?"
        assert payload.response == "this"

    def test_invoke_with_dict_prompt_and_response(self):
        guardrail = _make_guardrail_mock()
        runnable = GuardrailRunnable(guardrail=guardrail)
        runnable.invoke({"prompt": "question", "response": "answer"})
        call_args = guardrail.scan.call_args
        payload = call_args[0][0]
        assert isinstance(payload, DomePayload)
        assert payload.prompt == "question"
        assert payload.response == "answer"

    @pytest.mark.asyncio
    async def test_ainvoke_with_structured_dict(self):
        guardrail = _make_guardrail_mock()
        runnable = GuardrailRunnable(guardrail=guardrail)
        await runnable.ainvoke({"query": "q", "response": "r"})
        call_args = guardrail.async_scan.call_args
        payload = call_args[0][0]
        assert isinstance(payload, DomePayload)
