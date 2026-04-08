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
from vijil_dome.types import DomePayload
from vijil_dome.utils.config_loader import (
    load_dome_config_from_s3,
    config_has_changed as _config_has_changed,
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
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.agent_id = agent_id
        self.team_id = team_id
        self.user_id = user_id

    def get_input_guardrail(self) -> Guardrail:
        return self.input_guardrail

    def get_output_guardrail(self) -> Guardrail:
        return self.output_guardrail

    def get_agent_id(self) -> Optional[str]:
        return self.agent_id

    def get_team_id(self) -> Optional[str]:
        return self.team_id

    def get_user_id(self) -> Optional[str]:
        return self.user_id


def create_dome_config(config: Union[Dict, str]) -> DomeConfig:
    if isinstance(config, str):
        input_guardrail, output_guardrail, agent_id, team_id, user_id = (
            convert_toml_to_guardrails(config)
        )
        config_object = DomeConfig(
            input_guardrail,
            output_guardrail,
            agent_id=agent_id,
            team_id=team_id,
            user_id=user_id,
        )
        return config_object
    elif isinstance(config, dict):
        input_guardrail, output_guardrail, agent_id, team_id, user_id = (
            convert_dict_to_guardrails(config)
        )
        config_object = DomeConfig(
            input_guardrail,
            output_guardrail,
            agent_id=agent_id,
            team_id=team_id,
            user_id=user_id,
        )
        return config_object
    else:
        raise ValueError(
            "Only dicts or .toml filepaths can be converted to Dome Config objects"
        )


class ScanResult(BaseModel):
    flagged: bool
    enforced: bool = False  # True only when flagged AND enforcement is active
    response_string: str
    trace: dict
    exec_time: float
    detection_score: float = 0.0
    triggered_methods: List[str] = []

    def is_safe(self):
        return not self.flagged

    def guarded_response(self):
        return self.response_string

    def traceback(self):
        return self.trace

    def __str__(self):
        result_dict = {
            "flagged": self.flagged,
            "enforced": self.enforced,
            "trace": self.trace,
            "exec_time": self.exec_time,
            "response": self.response_string,
            "detection_score": self.detection_score,
            "triggered_methods": self.triggered_methods,
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
        enforce: bool = True,  # False = shadow mode (log but don't block)
    ):
        self.enforce = enforce
        self.client = client
        self.input_guardrail = None  # type: Optional[Guardrail]
        self.output_guardrail = None  # type: Optional[Guardrail]
        self.agent_id = None  # type: Optional[str]
        self.team_id = None  # type: Optional[str]
        self.user_id = None  # type: Optional[str]
        # S3 origin tracking (set by create_from_s3)
        self._s3_bucket = None  # type: Optional[str]
        self._s3_key = None  # type: Optional[str]
        self._s3_config_dict = None  # type: Optional[Dict]
        self._s3_aws_kwargs = None  # type: Optional[Dict]
        self._s3_cache_dir = None  # type: Optional[str]
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
        config: Union[Dict, str],
        client: Optional[OpenAI] = None,
        enforce: bool = True,
    ) -> "Dome":
        return Dome(dome_config=config, client=client, enforce=enforce)

    @staticmethod
    def create_from_vijil_agent(
        agent_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
        enforce: bool = True,
    ) -> "Dome":
        config_dict = get_config_from_vijil_agent(api_key, agent_id, base_url)
        if config_dict is None:
            raise ValueError(f"No Dome configuration found for agent ID {agent_id}")
        return Dome(dome_config=config_dict, client=client, enforce=enforce)

    @staticmethod
    def create_from_vijil_evaluation(
        evaluation_id: str,
        api_key: str,
        latency_threshold: Optional[float] = None,
        base_url: Optional[str] = None,
        client: Optional[OpenAI] = None,
        enforce: bool = True,
    ) -> "Dome":
        config_dict = get_config_from_vijil_evaluation(
            api_key, evaluation_id, base_url, latency_threshold
        )
        if config_dict is None:
            raise ValueError(
                f"No Dome configuration recommendation could be generated for evaluation ID {evaluation_id}"
            )
        return Dome(dome_config=config_dict, client=client, enforce=enforce)

    @staticmethod
    def create_from_s3(
        bucket: str,
        key: Optional[str] = None,
        team_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        client: Optional[OpenAI] = None,
        enforce: bool = True,
    ) -> "Dome":
        """Create a Dome instance from a config stored in S3.

        The S3 key can be provided directly, or constructed from
        *team_id* and *agent_id* using the standard path
        ``teams/{team_id}/agents/{agent_id}/dome/config.json``.

        The loaded config is cached locally and the S3 coordinates are
        stored on the instance so that :meth:`config_has_changed` can
        later check for remote updates.
        """
        aws_kwargs = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_session_token": aws_session_token,
            "region_name": region_name,
        }
        config_dict = load_dome_config_from_s3(
            bucket=bucket,
            key=key,
            team_id=team_id,
            agent_id=agent_id,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
            **aws_kwargs,
        )
        dome = Dome(dome_config=config_dict, client=client, enforce=enforce)
        # Store S3 origin for config_has_changed()
        from vijil_dome.utils.config_loader import _resolve_key

        dome._s3_bucket = bucket
        dome._s3_key = _resolve_key(key, team_id, agent_id)
        dome._s3_config_dict = config_dict
        dome._s3_aws_kwargs = aws_kwargs
        dome._s3_cache_dir = cache_dir
        return dome

    def config_has_changed(self) -> bool:
        """Check whether the S3 config has changed since this instance was created.

        Only works for instances created via :meth:`create_from_s3`.

        Returns:
            ``True`` if the remote config differs from the one used to
            create this instance, ``False`` otherwise.

        Raises:
            ValueError: If the instance was not created from S3.
        """
        if self._s3_bucket is None or self._s3_key is None:
            raise ValueError(
                "config_has_changed() is only available for Dome instances "
                "created via Dome.create_from_s3()."
            )
        return _config_has_changed(
            local_config=self._s3_config_dict,  # type: ignore[arg-type]
            bucket=self._s3_bucket,
            key=self._s3_key,
            cache_dir=self._s3_cache_dir,
            **(self._s3_aws_kwargs or {}),
        )

    def _init_from_dome_config(self, dome_config: DomeConfig):
        self.input_guardrail = dome_config.input_guardrail
        self.output_guardrail = dome_config.output_guardrail
        self.agent_id = dome_config.agent_id
        self.team_id = dome_config.team_id
        self.user_id = dome_config.user_id

    def get_agent_id(self) -> Optional[str]:
        return self.agent_id

    def get_team_id(self) -> Optional[str]:
        return self.team_id

    def get_user_id(self) -> Optional[str]:
        return self.user_id

    def get_guardrails(self):
        return [self.input_guardrail, self.output_guardrail]

    @staticmethod
    def _empty_guardrail_result(query_string: Union[str, DomePayload]):
        qs = DomePayload.coerce(query_string).query_string
        return ScanResult(
            flagged=False, response_string=qs, trace={}, exec_time=0.0
        )

    def guard_input(
        self,
        query_string: Union[str, DomePayload],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        dome_input = DomePayload.coerce(query_string)
        if self.input_guardrail is None:
            return self._empty_guardrail_result(dome_input)
        result = self.input_guardrail.scan(
            dome_input, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return ScanResult(
            flagged=result.flagged,
            enforced=self.enforce and result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
            detection_score=result.detection_score,
            triggered_methods=result.triggered_methods,
        )

    async def async_guard_input(
        self,
        query_string: Union[str, DomePayload],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        dome_input = DomePayload.coerce(query_string)
        if self.input_guardrail is None:
            return self._empty_guardrail_result(dome_input)
        result = await self.input_guardrail.async_scan(
            dome_input, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return ScanResult(
            flagged=result.flagged,
            enforced=self.enforce and result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
            detection_score=result.detection_score,
            triggered_methods=result.triggered_methods,
        )

    def guard_output(
        self,
        query_string: Union[str, DomePayload],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        dome_input = DomePayload.coerce(query_string)
        if self.output_guardrail is None:
            return self._empty_guardrail_result(dome_input)
        result = self.output_guardrail.scan(
            dome_input, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return ScanResult(
            flagged=result.flagged,
            enforced=self.enforce and result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
            detection_score=result.detection_score,
            triggered_methods=result.triggered_methods,
        )

    async def async_guard_output(
        self,
        query_string: Union[str, DomePayload],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        dome_input = DomePayload.coerce(query_string)
        if self.output_guardrail is None:
            return self._empty_guardrail_result(dome_input)
        result = await self.output_guardrail.async_scan(
            dome_input, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return ScanResult(
            flagged=result.flagged,
            enforced=self.enforce and result.flagged,
            response_string=result.guardrail_response_message,
            trace=result.guard_exec_details,
            exec_time=result.exec_time,
            detection_score=result.detection_score,
            triggered_methods=result.triggered_methods,
        )

    @staticmethod
    def _empty_batch_result(inputs: List[Union[str, DomePayload]]) -> "BatchScanResult":
        return BatchScanResult(
            items=[
                ScanResult(
                    flagged=False,
                    response_string=DomePayload.coerce(s).query_string,
                    trace={},
                    exec_time=0.0,
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
                enforced=self.enforce and gr.flagged,
                response_string=gr.guardrail_response_message,
                trace=gr.guard_exec_details,
                exec_time=gr.exec_time,
                detection_score=gr.detection_score,
                triggered_methods=gr.triggered_methods,
            ))
        return BatchScanResult(items=items, exec_time=batch_result.exec_time)

    def guard_input_batch(
        self,
        inputs: List[Union[str, DomePayload]],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> "BatchScanResult":
        if self.input_guardrail is None:
            return self._empty_batch_result(inputs)
        result = self.input_guardrail.scan_batch(
            inputs, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return self._batch_guardrail_to_scan(result)

    async def async_guard_input_batch(
        self,
        inputs: List[Union[str, DomePayload]],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> "BatchScanResult":
        if self.input_guardrail is None:
            return self._empty_batch_result(inputs)
        result = await self.input_guardrail.async_scan_batch(
            inputs, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return self._batch_guardrail_to_scan(result)

    def guard_output_batch(
        self,
        inputs: List[Union[str, DomePayload]],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> "BatchScanResult":
        if self.output_guardrail is None:
            return self._empty_batch_result(inputs)
        result = self.output_guardrail.scan_batch(
            inputs, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return self._batch_guardrail_to_scan(result)

    async def async_guard_output_batch(
        self,
        inputs: List[Union[str, DomePayload]],
        *,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> "BatchScanResult":
        if self.output_guardrail is None:
            return self._empty_batch_result(inputs)
        result = await self.output_guardrail.async_scan_batch(
            inputs, agent_id=agent_id, team_id=team_id, user_id=user_id
        )
        return self._batch_guardrail_to_scan(result)

    def apply_decorator(self, decoration_method: Callable) -> None:
        # Replace the default guard input and output functions with the decorated versions
        # Note - your decorator method should support both sync and async functions
        self.guard_input = decoration_method(self.guard_input)  # type: ignore[method-assign]
        self.guard_output = decoration_method(self.guard_output)  # type: ignore[method-assign]
        self.async_guard_input = decoration_method(self.async_guard_input)  # type: ignore[method-assign]
        self.async_guard_output = decoration_method(self.async_guard_output)  # type: ignore[method-assign]
