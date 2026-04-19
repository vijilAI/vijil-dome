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
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vijil_dome.detectors import DetectionMethod, DetectionResult
from vijil_dome.types import DomePayload

logger = logging.getLogger("vijil.dome")

# Base directory where the K8s init container (or a local setup script)
# syncs model weights from S3.  Detectors check this path first and fall
# back to the HuggingFace Hub when the local copy is absent.
MODEL_CACHE_DIR = os.environ.get("VIJIL_MODEL_DIR", "/models")


def resolve_model_path(model_name: str) -> str:
    """Return a local path if the model exists on disk, else the original
    HF Hub identifier so ``from_pretrained`` downloads it.

    The convention: if ``model_name`` looks like an HF repo ID (contains
    a ``/`` but is not an absolute path), check whether a matching
    directory exists under ``MODEL_CACHE_DIR``.  For example,
    ``vijil/stereotype-eeoc-detector`` resolves to
    ``/models/vijil/stereotype-eeoc-detector`` when that directory
    contains a ``config.json``.
    """
    if os.path.isabs(model_name) or os.path.isdir(model_name):
        return model_name  # already a concrete path

    candidate = Path(MODEL_CACHE_DIR) / model_name
    if candidate.is_dir() and (candidate / "config.json").exists():
        logger.info(
            "Resolved model to local path: %s (from %s)", candidate, model_name
        )
        return str(candidate)

    return model_name  # fall back to HF Hub download


class HFBaseModel(DetectionMethod, ABC):
    """
    Abstract base class for detection models using Hugging Face transformers.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        resolved = resolve_model_path(model_name)
        logger.info(f"Initializing Hugging Face model: {resolved}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model_tokenizer_name = tokenizer_name or model_name
        resolved_tokenizer = resolve_model_path(model_tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    @abstractmethod
    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        """
        Abstract method to be implemented by subclasses to execute the detection logic.

        Args:
            dome_input: The structured input to be analyzed by the detector.

        Returns:
            DetectionResult: A tuple containing a boolean indicating whether the input was flagged,
                             and a dictionary with additional details about the detection.
        """
        pass


class HFBaseModelWithContext(HFBaseModel):
    """
    Abstract base class for context-dependent detection models using Hugging Face transformers
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        context: Optional[str] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.context = context

    # Replace the existing context with new context
    def update_context(self, new_context: str) -> None:
        self.context = new_context

    # Add additional context to the existing context
    # If no context is present, update it instead
    def add_context(self, addition_context: str) -> None:
        if self.context is None:
            self.update_context(addition_context)
        else:
            self.context += addition_context
