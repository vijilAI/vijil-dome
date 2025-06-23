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

import logging
import torch
import os
from vijil_dome.detectors import (
    PI_DEBERTA_V3_BASE,
    PI_DEBERTA_FINETUNED_11122024,
    SECURITY_PROMPTGUARD,
    register_method,
    DetectionCategory,
    DetectionResult,
)
from transformers import pipeline
from torch.nn.functional import softmax
from vijil_dome.detectors.utils.hf_model import HFBaseModel

logger = logging.getLogger("vijil.dome")


class BaseDebertaPromptInjectionModel(HFBaseModel):
    """
    Base class for DeBERTa-based prompt injection detection models.
    """

    def __init__(
        self,
        model_identifier: str,
        response_method: str,
        model_dir: str = "deberta-prompt-injection",
        truncation: bool = True,
        max_length: int = 512
    ):
        try:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "models",
                model_dir,
            )
            if os.path.exists(model_path):
                super().__init__(model_path, local_files_only=True)
            else:
                super().__init__(model_identifier)

            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=truncation,
                max_length=max_length,
                device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.response_string = f"Method:{response_method}"
            self.run_in_executor = True
            logger.info("Initialized security model..")
        except Exception as e:
            logger.error(f"Failed to initialize DeBERTa model: {str(e)}")
            raise

    def sync_detect(self, query_string: str) -> DetectionResult:
        pred = self.classifier(query_string)
        flagged = pred[0]["label"] != "SAFE"
        return flagged, {
            "type": type(self),
            "predictions": pred,
            "response_string": self.response_string if flagged else query_string,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info(f"Detecting using {self.__class__.__name__}...")
        return self.sync_detect(query_string)


@register_method(DetectionCategory.Security, PI_DEBERTA_V3_BASE)
class DebertaPromptInjectionModel(BaseDebertaPromptInjectionModel):
    """
    https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
    """

    def __init__(self, truncation: bool = True, max_length: int = 512):
        super().__init__(
            model_identifier="protectai/deberta-v3-base-prompt-injection-v2",
            response_method=PI_DEBERTA_V3_BASE,
            truncation=truncation,
            max_length=max_length
        )


@register_method(DetectionCategory.Security, PI_DEBERTA_FINETUNED_11122024)
class DebertaTuned60PromptInjectionModel(BaseDebertaPromptInjectionModel):
    """
    https://huggingface.co/vijil/pi_deberta_finetuned_11122024
    """

    def __init__(self, truncation: bool = True, max_length: int = 512):
        super().__init__(
            model_identifier="vijil/pi_deberta_finetuned_11122024",
            response_method=PI_DEBERTA_FINETUNED_11122024,
            truncation=truncation,
            max_length=max_length
        )


@register_method(DetectionCategory.Security, SECURITY_PROMPTGUARD)
class PromptGuardSecurityModel(HFBaseModel):
    def __init__(
        self,
        score_threshold: float = 0.5,
        truncation: bool = True,
        max_length: int = 512,
    ):
        try:
            model_path = os.path.join(
                os.path.dirname(__file__), "models", "promptguard"
            )
            if os.path.exists(model_path):
                super().__init__(model_path, local_files_only=True)
            else:
                super().__init__("meta-llama/Prompt-Guard-86M", local_files_only=False)
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                padding=True,
                truncation=truncation,
                max_length=max_length,
                device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.score_threshold = score_threshold
            self.response_string = f"Method:{SECURITY_PROMPTGUARD}"
            self.run_in_executor = True
            logger.info("Initialized security model..")
        except Exception as e:
            logger.error(f"Failed to initialize DeBERTa model: {str(e)}")
            raise

    def get_class_probabilities(self, text, temperature=1.0, device="cpu"):
        """
        Evaluate the model on the given text with temperature-adjusted softmax.
        Args:
            text (str): The input text to classify.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.

        Returns:
            torch.Tensor: The probability of each class adjusted by the temperature.
        """
        # Encode the text
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = inputs.to(device)
        # Get logits from the model
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # Apply temperature scaling
        scaled_logits = logits / temperature
        # Apply softmax to get probabilities
        probabilities = softmax(scaled_logits, dim=-1)
        return probabilities

    def get_jailbreak_score(self, text, temperature=1.0, device="cpu"):
        """
        Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
        Appropriate for filtering dialogue between a user and an LLM.
        Args:
            text (str): The input text to evaluate.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.
        Returns:
            float: The probability of the text containing malicious content.
        """
        probabilities = self.get_class_probabilities(text, temperature, device)
        return probabilities[0, 2].item()

    # This is recommended for tool outputs and has a very high FP rate on non=formatted user query strings
    def get_indirect_injection_score(self, text, temperature=1.0, device="cpu"):
        """
        Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
        Appropriate for filtering third party inputs (e.g. web searches, tool outputs) into an LLM.

        Args:
            text (str): The input text to evaluate.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.
        Returns:
            float: The combined probability of the text containing malicious or embedded instructions.
        """
        probabilities = self.get_class_probabilities(text, temperature, device)
        return (probabilities[0, 1] + probabilities[0, 2]).item()

    def sync_detect(self, query_string: str) -> DetectionResult:
        logger.debug("Detecting using Prompt Guard...")
        jb_score = self.get_jailbreak_score(query_string)
        flagged = jb_score >= self.score_threshold
        return flagged, {
            "type": type(self),
            "score": jb_score,
            "response_string": self.response_string if flagged else query_string,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        return self.sync_detect(query_string)