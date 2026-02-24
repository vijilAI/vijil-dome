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

from vijil_dome.guardrails import Guardrail, GuardrailConfig, BatchGuardrailResult
from vijil_dome.guardrails.config_parser import (
    convert_dict_to_guardrails,
    convert_toml_to_guardrails,
)
from vijil_dome.integrations.vijil.evaluate import (
    get_config_from_vijil_agent,
    get_config_from_vijil_evaluation,
)
from openai import OpenAI
from typing import Union, Dict, List, Optional, Callable, Iterator
from .defaults import get_default_config
from pydantic import BaseModel


class DomeConfig(GuardrailConfig):
    def __init__(
        self,
        input_guardrail: Guardrail,
        output_guardrail: Guardrail,
        agent_id: Optional[str] = None,
    ):
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.agent_id = agent_id

    def get_input_guardrail(self) -> Guardrail:
        return self.input_guardrail

    def get_output_guardrail(self) -> Guardrail:
        return self.output_guardrail

    def get_agent_id(self) -> Optional[str]:
        return self.agent_id


def create_dome_config(config: Union[Dict, str]) -> DomeConfig:
    if isinstance(config, str):
        input_guardrail, output_guardrail, agent_id = convert_toml_to_guardrails(config)
        config_object = DomeConfig(input_guardrail, output_guardrail, agent_id)
        return config_object
    elif isinstance(config, dict):
        input_guardrail, output_guardrail, agent_id = convert_dict_to_guardrails(config)
        config_object = DomeConfig(input_guardrail, output_guardrail, agent_id)
        return config_object
    else:
        raise ValueError(
            "Only dicts or .toml filepaths can be converted to Dome Config objects"
        )


class ScanResult(BaseModel):
    flagged: bool
    response_string: str
    trace: dict
    exec_time: float

    def is_safe(self):
        return not self.flagged

    def guarded_response(self):
        return self.response_string

    def traceback(self):
        return self.trace

    def __str__(self):
        result_dict = {
            "flagged": self.flagged,
            "trace": self.trace,
            "exec_time": self.exec_time,
            "response": self.response_string,
        }
        return str(result_dict)

    def __repr__(self):
        return self.__str__()


class BatchScanResult(BaseModel):
    items: List[ScanResult]
    exec_time: float

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ScanResult:
        return self.items[index]

    def __iter__(self) -> Iterator[ScanResult]:  # type: ignore[override]
        return iter(self.items)

    def all_safe(self) -> bool:
        return all(item.is_safe() for item in self.items)

    def any_flagged(self) -> bool:
        return any(item.flagged for item in self.items)


class Dome:
    def __init__(
        self,
        dome_config: Optional[Union[DomeConfig, Dict, str]] = None,
        client: Optional[OpenAI] = None,
    ):
        self.client = client
        self.input_guardrail = None  # type: Optional[Guardrail]
        self.output_guardrail = None  # type: Optional[Guardrail]
        self.agent_id = None  # type: Optional[str]
        if dome_config is not None:
            if isinstance(dome_config, DomeConfig):
                self._init_from_dome_config(dome_config)
            elif isinstance(dome_config, str) or isinstance(dome_config, dict):
                self._init_from_dome_config(create_dome_config(dome_config))
            else:
                raise ValueError(
                    f"Dome recieved an invalid type ({type(dome_config)}) for configuration. Please use a dictionary, a path to a .toml file, or a DomeConfig object to initialize Dome."
                )
        else:
            self._init_from_dome_config(create_dome_config(get_default_config()))

    @staticmethod
    def create_from_config(
        config: Union[Dict, str], client: Optional[OpenAI] = None
    ) -> "Dome":
        return Dome(dome_config=config, client=client)

    @staticmethod
    def create_from_vijil_agent(
        agent_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> "Dome":
        config_dict = get_config_from_vijil_agent(api_key, agent_id, base_url)
        if config_dict is None:
            raise ValueError(f"No Dome configuration found for agent ID {agent_id}")
        return Dome(dome_config=config_dict, client=client)

    @staticmethod
    def create_from_vijil_evaluation(
        evaluation_id: str,
        api_key: str,
        latency_threshold: Optional[float] = None,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> "Dome":
        config_dict = get_config_from_vijil_evaluation(
            api_key, evaluation_id, base_url, latency_threshold
        )
        if config_dict is None:
            raise ValueError(
                f"No Dome configuration recommendation could be generated for evaluation ID {evaluation_id}"
            )
        return Dome(dome_config=config_dict, client=client)

    def _init_from_dome_config(self, dome_config: DomeConfig):
        self.input_guardrail = dome_config.input_guardrail
        self.output_guardrail = dome_config.output_guardrail
        self.agent_id = dome_config.agent_id

    def get_guardrails(self):
        return [self.input_guardrail, self.output_guardrail]

    @staticmethod
    def _empty_guardrail_result(query_string: str):
        return ScanResult(
            flagged=False, response_string=query_string, trace={}, exec_time=0.0
        )

    def guard_input(self, query_string: str, *, agent_id: Optional[str] = None):
        if self.input_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = self.input_guardrail.scan(query_string, agent_id=agent_id)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    async def async_guard_input(
        self, query_string: str, *, agent_id: Optional[str] = None
    ):
        if self.input_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = await self.input_guardrail.async_scan(query_string, agent_id=agent_id)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    def guard_output(self, query_string: str, *, agent_id: Optional[str] = None):
        if self.output_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = self.output_guardrail.scan(query_string, agent_id=agent_id)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    async def async_guard_output(
        self, query_string: str, *, agent_id: Optional[str] = None
    ):
        if self.output_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = await self.output_guardrail.async_scan(query_string, agent_id=agent_id)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    @staticmethod
    def _empty_batch_result(inputs: List[str]) -> "BatchScanResult":
        return BatchScanResult(
            items=[
                ScanResult(
                    flagged=False, response_string=s, trace={}, exec_time=0.0
                )
                for s in inputs
            ],
            exec_time=0.0,
        )

    def _batch_guardrail_to_scan(
        self, batch_result: BatchGuardrailResult
    ) -> "BatchScanResult":
        items = []
        for gr in batch_result.items:
            items.append(ScanResult(
                flagged=gr.flagged,
                response_string=gr.guardrail_response_message,
                trace=gr.guard_exec_details,
                exec_time=gr.exec_time,
            ))
        return BatchScanResult(items=items, exec_time=batch_result.exec_time)

    def guard_input_batch(
        self, inputs: List[str], *, agent_id: Optional[str] = None
    ) -> "BatchScanResult":
        if self.input_guardrail is None:
            return self._empty_batch_result(inputs)
        result = self.input_guardrail.scan_batch(inputs, agent_id=agent_id)
        return self._batch_guardrail_to_scan(result)

    async def async_guard_input_batch(
        self, inputs: List[str], *, agent_id: Optional[str] = None
    ) -> "BatchScanResult":
        if self.input_guardrail is None:
            return self._empty_batch_result(inputs)
        result = await self.input_guardrail.async_scan_batch(
            inputs, agent_id=agent_id
        )
        return self._batch_guardrail_to_scan(result)

    def guard_output_batch(
        self, inputs: List[str], *, agent_id: Optional[str] = None
    ) -> "BatchScanResult":
        if self.output_guardrail is None:
            return self._empty_batch_result(inputs)
        result = self.output_guardrail.scan_batch(inputs, agent_id=agent_id)
        return self._batch_guardrail_to_scan(result)

    async def async_guard_output_batch(
        self, inputs: List[str], *, agent_id: Optional[str] = None
    ) -> "BatchScanResult":
        if self.output_guardrail is None:
            return self._empty_batch_result(inputs)
        result = await self.output_guardrail.async_scan_batch(
            inputs, agent_id=agent_id
        )
        return self._batch_guardrail_to_scan(result)

    def apply_decorator(self, decoration_method: Callable) -> None:
        # Replace the default guard input and output functions with the decorated versions
        # Note - your decorator method should support both sync and async functions
        self.guard_input = decoration_method(self.guard_input)  # type: ignore[method-assign]
        self.guard_output = decoration_method(self.guard_output)  # type: ignore[method-assign]
        self.async_guard_input = decoration_method(self.async_guard_input)  # type: ignore[method-assign]
        self.async_guard_output = decoration_method(self.async_guard_output)  # type: ignore[method-assign]
