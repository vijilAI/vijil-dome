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

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from vijil_dome.detectors import DetectionMethod, DetectionResult
from vijil_dome.types import DomePayload

logger = logging.getLogger("vijil.dome")

# Base directory where the K8s init container (or a local setup script)
# syncs model weights from S3 (``s3://vijil-inference/models/``).  This is
# the ONLY source of detector weights — see resolve_model_path.
MODEL_CACHE_DIR = os.environ.get("VIJIL_MODEL_DIR", "/models")

_S3_MODEL_SOURCE = "s3://vijil-inference/models/"


class ModelNotAvailableError(RuntimeError):
    """A detector's weights are absent from the local model directory.

    Raised instead of silently reaching out to the HuggingFace Hub. The
    message names the path that was checked and the sync that populates it,
    because the failure a caller sees would otherwise be a network or
    model-card error naming neither.
    """


def resolve_model_path(model_name: str) -> str:
    """Resolve a model id to a concrete local path under ``MODEL_CACHE_DIR``.

    Weights come from S3 and nowhere else. There is deliberately **no
    HuggingFace Hub fallback**: a fallback makes an air-gapped or
    egress-restricted deployment indistinguishable from a correctly-synced
    one until the network call fails, and it makes the supply chain for
    detector weights ambient rather than declared.

    The convention: an HF-style repo id (``vijil/stereotype-eeoc-detector``)
    resolves to ``$VIJIL_MODEL_DIR/vijil/stereotype-eeoc-detector``, which is
    the layout ``aws s3 sync s3://vijil-inference/models/ $VIJIL_MODEL_DIR``
    produces.

    Raises:
        ModelNotAvailableError: no directory with a ``config.json`` exists at
            the resolved path.
    """
    if os.path.isabs(model_name) or os.path.isdir(model_name):
        return model_name  # already a concrete path

    candidate = Path(MODEL_CACHE_DIR) / model_name
    if candidate.is_dir() and (candidate / "config.json").exists():
        logger.info("Resolved model to local path: %s (from %s)", candidate, model_name)
        return str(candidate)

    raise ModelNotAvailableError(
        f"Model {model_name!r} is not present at {candidate}. Detector weights "
        f"are served from {_S3_MODEL_SOURCE} and are never fetched from the "
        f"HuggingFace Hub. Populate the directory with:\n"
        f"    aws s3 sync {_S3_MODEL_SOURCE} {MODEL_CACHE_DIR}/\n"
        f"or point VIJIL_MODEL_DIR (currently {MODEL_CACHE_DIR!r}) at an "
        f"existing sync."
    )


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
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                f"{self.__class__.__name__} requires 'torch' and 'transformers'. "
                "Install with: pip install vijil-dome[local]"
            )
        # resolve_model_path returns a concrete local path or raises, so every
        # load is offline by construction. local_files_only is pinned True
        # rather than derived: it is the assertion that no code path here can
        # reach an external host, and the `local_files_only` argument is kept
        # only for API compatibility with existing callers.
        resolved = resolve_model_path(model_name)
        logger.info("Initializing Hugging Face model from %s", resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model_tokenizer_name = tokenizer_name or model_name
        resolved_tokenizer = resolve_model_path(model_tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer,
            local_files_only=True,
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
