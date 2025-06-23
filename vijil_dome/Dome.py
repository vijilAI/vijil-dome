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

from vijil_dome.guardrails import Guardrail, GuardrailConfig
from vijil_dome.guardrails.config_parser import (
    convert_dict_to_guardrails,
    convert_toml_to_guardrails,
)
from openai import OpenAI
from typing import Union, Dict, Optional, Callable
from .defaults import get_default_config
from pydantic import BaseModel


class DomeConfig(GuardrailConfig):
    def __init__(self, input_guardrail: Guardrail, output_guardrail: Guardrail):
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail

    def get_input_guardrail(self) -> Guardrail:
        return self.input_guardrail

    def get_output_guardrail(self) -> Guardrail:
        return self.output_guardrail


def create_dome_config(config: Union[Dict, str]) -> DomeConfig:
    if isinstance(config, str):
        input_guardrail, output_guardrail = convert_toml_to_guardrails(config)
        config_object = DomeConfig(input_guardrail, output_guardrail)
        return config_object
    elif isinstance(config, dict):
        input_guardrail, output_guardrail = convert_dict_to_guardrails(config)
        config_object = DomeConfig(input_guardrail, output_guardrail)
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
        return result_dict

    def __repr__(self):
        return self.__str__()


class Dome:
    def __init__(
        self,
        dome_config: Optional[Union[DomeConfig, Dict, str]] = None,
        client: Optional[OpenAI] = None,
    ):
        self.client = client
        self.input_guardrail = None  # type: Optional[Guardrail]
        self.output_guardrail = None  # type: Optional[Guardrail]
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

    def _init_from_dome_config(self, dome_config: DomeConfig):
        self.input_guardrail = dome_config.input_guardrail
        self.output_guardrail = dome_config.output_guardrail

    def get_guardrails(self):
        return [self.input_guardrail, self.output_guardrail]

    @staticmethod
    def _empty_guardrail_result(query_string: str):
        return ScanResult(
            flagged=False, response_string=query_string, trace={}, exec_time=0.0
        )

    def guard_input(self, query_string: str):
        if self.input_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = self.input_guardrail.scan(query_string)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    async def async_guard_input(self, query_string: str):
        if self.input_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = await self.input_guardrail.async_scan(query_string)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    def guard_output(self, query_string: str):
        if self.output_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = self.output_guardrail.scan(query_string)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    async def async_guard_output(self, query_string: str):
        if self.output_guardrail is None:
            return self._empty_guardrail_result(query_string)
        result = await self.output_guardrail.async_scan(query_string)
        return ScanResult(
            flagged=result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
        )

    def apply_decorator(self, decoration_method: Callable) -> None:
        # Replace the default guard input and output functions with the decorated versions
        # Note - your decorator method should support both sync and async functions
        self.guard_input = decoration_method(self.guard_input)  # type: ignore[method-assign]
        self.guard_output = decoration_method(self.guard_output)  # type: ignore[method-assign]
        self.async_guard_input = decoration_method(self.async_guard_input)  # type: ignore[method-assign]
        self.async_guard_output = decoration_method(self.async_guard_output)  # type: ignore[method-assign]
