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

from langchain_core.runnables import Runnable
from vijil_dome.guardrails import Guardrail, GuardrailResult
from typing import Optional, Any, Dict, Union
from langchain_core.runnables.config import RunnableConfig


class GuardrailRunnable(Runnable):
    def __init__(self, guardrail: Guardrail):
        self.guardrail = guardrail
        super().__init__()

    def _handle_result(
        self, guardrail_result: GuardrailResult, query: str
    ) -> Dict[str, Any]:
        result = vars(guardrail_result)
        result["original_query"] = query
        return result

    def invoke(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ):
        query = input if isinstance(input, str) else input.get("query", "")
        guardrail_result = self.guardrail.scan(query)
        return self._handle_result(guardrail_result, query)

    async def ainvoke(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ):
        query = input if isinstance(input, str) else input.get("query", "")
        guardrail_result = await self.guardrail.async_scan(query)
        return self._handle_result(guardrail_result, query)
