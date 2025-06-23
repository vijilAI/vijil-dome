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

import time
import json
import logging
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Type, Dict, Tuple, Callable, Coroutine, Any
from collections import defaultdict
from pydantic import BaseModel


MODERATION_FLASHTXT_BANLIST = "moderation-flashtext"
MODERATION_LLAMA_GUARD = "moderation-llamaguard"
MODERATION_OPENAI = "moderations-oai-api"
MODERATION_LLM = "moderation-prompt-engineering"
MODERATION_PERSPECTIVE = "moderation-perspective-api"
MODERATION_DEBERTA = "moderation-deberta"

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

HHEM = "hhem-hallucination"
HALLUCINATION_LLM = "hallucination-llm"
FACTCHECK_ROBERTA = "fact-check-roberta"
FACTCHECK_LLM = "fact-check-llm"

GENERIC_LLM = "generic-llm"

# Define types for detection results and data
DetectorType = Dict["type", str]
Hit = bool
HitData = Dict[str, Any]
DetectionResult = Tuple[Hit, HitData]


class DetectionCategory(Enum):
    Security = auto()
    Privacy = auto()
    Moderation = auto()
    Integrity = auto()
    Generic = auto()


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
        return json.dumps(result_dict, indent=4)

    def __repr__(self):
        return self.__str__()


class DetectionMethod(ABC):
    """
    Abstract base class for all detection methods.
    """

    @abstractmethod
    async def detect(self, query_string: str) -> DetectionResult:
        """
        Perform the detection logic.
        """
        pass

    async def detect_with_time(self, query_string: str) -> DetectionTimingResult:
        """
        Perform the detection and return the result along with the execution time.
        """
        start_time = time.time()
        result = await self.detect(query_string)
        execution_time = round((time.time() - start_time) * 1000, 3)

        detection_result = DetectionTimingResult(
            hit=result[0], result=result[1], exec_time=execution_time
        )
        sanitized_result = self._sanitize_result(result)
        logging.info(f"{sanitized_result}")

        return detection_result

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
    ) -> Callable[[str], Coroutine[Any, Any, DetectionResult]]:
        detector = DetectionFactory.get_detector(category, method_name, **kwargs)
        return detector.detect

    @staticmethod
    async def get_detect_with_time(
        category: DetectionCategory, method_name: str, **kwargs
    ) -> Callable[[str], Coroutine[Any, Any, DetectionTimingResult]]:
        detector = DetectionFactory.get_detector(category, method_name, **kwargs)
        # warm the model
        await detector.detect("")

        return detector.detect_with_time

    @staticmethod
    async def list_detector_names(category: DetectionCategory) -> list[str]:
        if category not in method_registry:
            logging.error(f"No such category {category.name}")
            raise ValueError(f"No such category {category.name}")
        return list(method_registry[category].keys())
