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
import time
import json
import logging
import inspect
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Type, Dict, List, Tuple, Callable, Coroutine, Any, Optional, Union
from collections import defaultdict
from pydantic import BaseModel

from vijil_dome.types import DomePayload


MODERATION_FLASHTXT_BANLIST = "moderation-flashtext"
MODERATION_LLAMA_GUARD = "moderation-llamaguard"
MODERATION_OPENAI = "moderations-oai-api"
MODERATION_LLM = "moderation-prompt-engineering"
MODERATION_PERSPECTIVE = "moderation-perspective-api"
MODERATION_DEBERTA = "moderation-deberta"
MODERATION_MBERT = "moderation-mbert"

PRIVACY_PRESIDIO = "privacy-presidio"
DETECT_SECRETS = "detect-secrets"

JB_LENGTH_PER_PERPLEXITY = "jb-length-per-perplexity"
JB_PREFIX_SUFFIX_PERPLEXITY = "jb-prefix-suffix-perplexity"
PI_DEBERTA_V3_BASE = "prompt-injection-deberta-v3-base"
PI_DEBERTA_FINETUNED_11122024 = "prompt-injection-deberta-finetuned-11122024"
PI_MBERT = "prompt-injection-mbert"
SECURITY_LLM = "security-llm"
SECURITY_EMBEDDINGS = "security-embeddings"
SECURITY_PROMPTGUARD = "security-promptguard"
ENCODING_HEURISTICS = "encoding-heuristics"

HHEM = "hhem-hallucination"
HALLUCINATION_LLM = "hallucination-llm"
FACTCHECK_ROBERTA = "fact-check-roberta"
FACTCHECK_LLM = "fact-check-llm"

GENERIC_LLM = "generic-llm"
POLICY_GPT_OSS_SAFEGUARD = "policy-gpt-oss-safeguard"
POLICY_SECTIONS = "policy-sections"

STEREOTYPE_EEOC_FAST = "stereotype-eeoc-fast"
STEREOTYPE_EEOC_SAFEGUARD = "stereotype-eeoc-safeguard"
STEREOTYPE_EEOC_HYBRID = "stereotype-eeoc-hybrid"

# Define types for detection results and data
DetectorType = Dict["type", str]
Hit = bool
HitData = Dict[str, Any]
DetectionResult = Tuple[Hit, HitData]
BatchDetectionResult = List[DetectionResult]


class DetectionCategory(Enum):
    Security = auto()
    Privacy = auto()
    Moderation = auto()
    Integrity = auto()
    Generic = auto()
    Policy = auto()


class DetectionTimingResult(BaseModel):
    hit: bool
    result: HitData
    exec_time: float

    def __str__(self):
        result_serialized = {}
        for key, value in self.result.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                result_serialized[key] = value
            else:
                # Convert non-serializable types to string
                result_serialized[key] = str(value)

        result_dict = {
            "hit": self.hit,
            "result": result_serialized,
            "exec_time": self.exec_time,
        }

        # Serialize to a JSON formatted string
        return json.dumps(result_dict, indent=4, default=str)

    def __repr__(self):
        return self.__str__()


class BatchDetectionTimingResult(BaseModel):
    results: List[DetectionTimingResult]
    exec_time: float  # total wall-clock for entire batch


class DetectionMethod(ABC):
    """
    Abstract base class for all detection methods.
    """

    DEFAULT_MAX_BATCH_CONCURRENCY = 5
    max_batch_concurrency: int = DEFAULT_MAX_BATCH_CONCURRENCY

    async def _gather_with_concurrency(
        self, coros: List[Coroutine],
    ) -> List[Any]:
        """Run coroutines with a concurrency cap of self.max_batch_concurrency."""
        semaphore = asyncio.Semaphore(self.max_batch_concurrency)

        async def _limited(coro):
            async with semaphore:
                return await coro

        return list(await asyncio.gather(*[_limited(c) for c in coros]))

    @staticmethod
    def _call_with_supported_kwargs(func, *args, **kwargs):
        signature = inspect.signature(func)
        supports_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if supports_var_kwargs:
            return func(*args, **kwargs)
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        return func(*args, **filtered_kwargs)

    @abstractmethod
    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        """
        Perform the detection logic.
        """
        pass

    async def detect_with_time(
        self,
        query_string: Union[str, DomePayload],
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> DetectionTimingResult:
        """
        Perform the detection and return the result along with the execution time.
        """
        dome_input = DomePayload.coerce(query_string)
        start_time = time.time()
        result = await self._call_with_supported_kwargs(
            self.detect,
            dome_input,
            agent_id=agent_id,
            team_id=team_id,
            user_id=user_id,
        )
        execution_time = round((time.time() - start_time) * 1000, 3)

        result_payload = dict(result[1])
        if agent_id:
            result_payload.setdefault("agent_id", agent_id)
        if team_id:
            result_payload.setdefault("team_id", team_id)
        if user_id:
            result_payload.setdefault("user_id", user_id)

        detection_result = DetectionTimingResult(
            hit=result[0], result=result_payload, exec_time=execution_time
        )
        sanitized_result = self._sanitize_result(result)
        logging.info(f"{sanitized_result}")

        return detection_result

    async def detect_batch(
        self, inputs: List[Union[str, DomePayload]]
    ) -> BatchDetectionResult:
        """
        Process a batch of inputs. Default implementation loops over detect().
        Subclasses can override for optimized batch processing.
        """
        results = []
        for item in inputs:
            dome_input = DomePayload.coerce(item)
            result = await self.detect(dome_input)
            results.append(result)
        return results

    async def detect_batch_with_time(
        self,
        inputs: List[Union[str, DomePayload]],
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> BatchDetectionTimingResult:
        """
        Timing wrapper for batch detection. Calls detect_batch and wraps
        each result in DetectionTimingResult.
        """
        start_time = time.time()
        batch_results = await self._call_with_supported_kwargs(
            self.detect_batch,
            inputs,
            agent_id=agent_id,
            team_id=team_id,
            user_id=user_id,
        )
        total_time = round((time.time() - start_time) * 1000, 3)

        timing_results = []
        for result in batch_results:
            result_payload = dict(result[1])
            if agent_id:
                result_payload.setdefault("agent_id", agent_id)
            if team_id:
                result_payload.setdefault("team_id", team_id)
            if user_id:
                result_payload.setdefault("user_id", user_id)
            timing_results.append(
                DetectionTimingResult(
                    hit=result[0], result=result_payload, exec_time=0.0
                )
            )

        return BatchDetectionTimingResult(
            results=timing_results, exec_time=total_time
        )

    def setDetectorHyperParams(self, **params):
        """
        Set hyperparameters for the detection method.
        This method updates the detector's hyperparameter settings.
        Args:
            params (Dict[str, Any]): A dictionary of hyperparameters.
        """
        self.apply_hyperparams()

    def apply_hyperparams(self):
        """
        Apply the hyperparameters to the detector. This method should be overridden by subclasses
        that need specific actions taken when hyperparameters are changed.
        """
        pass  # Default implementation does nothing, to be overridden by subclasses.

    def _sanitize_result(self, result: Tuple[bool, dict]) -> dict:
        """
        Sanitize the result dictionary to mask sensitive data.
        """
        sanitized_result = result[1].copy()
        if "secrets" in sanitized_result:
            sanitized_result["secrets"] = ["***" for _ in sanitized_result["secrets"]]
        if "query_string" in sanitized_result:
            sanitized_result["query_string"] = "***"
        return {"hit": result[0], "result": sanitized_result}


# Registry to maintain a mapping between detection categories and their available methods (models)
method_registry: Dict[DetectionCategory, Dict[str, Type[DetectionMethod]]] = (
    defaultdict(dict)
)


def register_method(category: DetectionCategory, model_name: str):
    def decorator(cls):
        cls.type = f"{category.name}.{model_name}"
        method_registry[category][model_name] = cls
        return cls

    return decorator


class DetectionFactory:
    @staticmethod
    def get_detector(
        category: DetectionCategory, method_name: str, **kwargs
    ) -> DetectionMethod:
        method_class = method_registry[category].get(method_name)
        if not method_class:
            logging.error(
                f"No method available for {category.name} with model {method_name}"
            )
            raise ValueError(
                f"No method available for {category.name} with model {method_name}"
            )
        return method_class(**kwargs)

    @staticmethod
    async def get_detect(
        category: DetectionCategory, method_name: str, **kwargs
    ) -> Callable[..., Coroutine[Any, Any, DetectionResult]]:
        detector = DetectionFactory.get_detector(category, method_name, **kwargs)
        return detector.detect

    @staticmethod
    async def get_detect_with_time(
        category: DetectionCategory, method_name: str, **kwargs
    ) -> Callable[[Union[str, DomePayload]], Coroutine[Any, Any, DetectionTimingResult]]:
        detector = DetectionFactory.get_detector(category, method_name, **kwargs)
        # warm the model
        await detector.detect(DomePayload(text=""))

        return detector.detect_with_time

    @staticmethod
    async def list_detector_names(category: DetectionCategory) -> list[str]:
        if category not in method_registry:
            logging.error(f"No such category {category.name}")
            raise ValueError(f"No such category {category.name}")
        return list(method_registry[category].keys())
