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
from vijil_dome.integrations.vijil.evaluate import (
    get_config_from_vijil_agent,
    get_config_from_vijil_evaluation,
)
from vijil_dome.utils.policy_loader import load_policy_sections_from_s3
from vijil_dome.utils.policy_config_builder import build_dome_config_from_sections
from openai import OpenAI
from typing import Union, Dict, Optional, Callable
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

    @staticmethod
    def create_from_s3_policy(
        bucket: str,
        key: str,
        cache_dir: Optional[str] = None,
        client: Optional[OpenAI] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        model_name: str = "openai/gpt-oss-120b",
        reasoning_effort: str = "medium",
        hub_name: str = "nebius",
        timeout: Optional[int] = 60,
        max_retries: Optional[int] = 3,
        api_key: Optional[str] = None,
        early_exit: bool = True,
        run_parallel: bool = True,
    ) -> "Dome":
        """
        Create a Dome instance from policy sections stored in S3.

        Loads policy sections JSON from S3 (with local caching), builds a Dome
        config, and returns an initialized Dome instance.

        Args:
            bucket: S3 bucket name
            key: S3 object key (full path to sections.json)
            cache_dir: Local cache directory (defaults to ~/.cache/vijil-dome/policies/)
            client: Optional OpenAI client
            aws_access_key_id: AWS access key (optional, uses boto3 defaults if not provided)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
            region_name: AWS region (optional, uses boto3 defaults if not provided)
            model_name: LLM model to use for all detectors (default: "openai/gpt-oss-120b")
            reasoning_effort: Reasoning depth - "low", "medium", "high" (default: "medium")
            hub_name: LLM hub to use (default: "nebius")
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts (default: 3)
            api_key: API key for the hub (optional, uses env var if not provided)
            early_exit: Enable early exit when first violation detected (default: True)
            run_parallel: Run detectors in parallel (default: True)

        Returns:
            Initialized Dome instance

        Example:
            dome = Dome.create_from_s3_policy(
                bucket="my-policy-bucket",
                key="teams/team-123/policies/policy-456/sections.json",
                model_name="openai/gpt-oss-120b",
                reasoning_effort="medium"
            )
        """
        policy_data = load_policy_sections_from_s3(
            bucket=bucket,
            key=key,
            cache_dir=cache_dir,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )

        config = build_dome_config_from_sections(
            policy_data,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            hub_name=hub_name,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            early_exit=early_exit,
            run_parallel=run_parallel,
        )

        return Dome(dome_config=config, client=client)

    @staticmethod
    def create_from_s3_policy_by_ids(
        bucket: str,
        team_id: str,
        policy_id: str,
        cache_dir: Optional[str] = None,
        client: Optional[OpenAI] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        model_name: str = "openai/gpt-oss-120b",
        reasoning_effort: str = "medium",
        hub_name: str = "nebius",
        timeout: Optional[int] = 60,
        max_retries: Optional[int] = 3,
        api_key: Optional[str] = None,
        early_exit: bool = True,
        run_parallel: bool = True,
    ) -> "Dome":
        """
        Create a Dome instance from policy sections using team_id and policy_id.

        Constructs the S3 key path as: teams/{team_id}/policies/{policy_id}/sections.json
        and calls create_from_s3_policy internally.

        Args:
            bucket: S3 bucket name
            team_id: Team identifier
            policy_id: Policy identifier
            cache_dir: Local cache directory (defaults to ~/.cache/vijil-dome/policies/)
            client: Optional OpenAI client
            aws_access_key_id: AWS access key (optional, uses boto3 defaults if not provided)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
            region_name: AWS region (optional, uses boto3 defaults if not provided)
            model_name: LLM model to use for all detectors (default: "openai/gpt-oss-120b")
            reasoning_effort: Reasoning depth - "low", "medium", "high" (default: "medium")
            hub_name: LLM hub to use (default: "nebius")
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts (default: 3)
            api_key: API key for the hub (optional, uses env var if not provided)
            early_exit: Enable early exit when first violation detected (default: True)
            run_parallel: Run detectors in parallel (default: True)

        Returns:
            Initialized Dome instance

        Example:
            dome = Dome.create_from_s3_policy_by_ids(
                bucket="my-policy-bucket",
                team_id="550e8400-e29b-41d4-a716-446655440000",
                policy_id="123e4567-e89b-12d3-a456-426614174000",
                model_name="openai/gpt-oss-120b",
                reasoning_effort="medium"
            )
        """
        key = f"teams/{team_id}/policies/{policy_id}/sections.json"
        return Dome.create_from_s3_policy(
            bucket=bucket,
            key=key,
            cache_dir=cache_dir,
            client=client,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            hub_name=hub_name,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            early_exit=early_exit,
            run_parallel=run_parallel,
        )

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

    def apply_decorator(self, decoration_method: Callable) -> None:
        # Replace the default guard input and output functions with the decorated versions
        # Note - your decorator method should support both sync and async functions
        self.guard_input = decoration_method(self.guard_input)  # type: ignore[method-assign]
        self.guard_output = decoration_method(self.guard_output)  # type: ignore[method-assign]
        self.async_guard_input = decoration_method(self.async_guard_input)  # type: ignore[method-assign]
        self.async_guard_output = decoration_method(self.async_guard_output)  # type: ignore[method-assign]
