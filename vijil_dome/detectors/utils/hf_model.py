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

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )
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

    An HF-style repo id (``vijil/stereotype-eeoc-detector``) resolves under
    ``$VIJIL_MODEL_DIR``. The bucket stores each model **nested one level
    deeper, under its commit SHA**::

        models/vijil/stereotype-eeoc-detector/5a55be4dd419.../config.json

    so the weights are at ``<repo-id>/<revision>/``, not ``<repo-id>/``.
    Several models carry two revisions, and the current one is marked by a
    ``.version`` file inside it. Selection order:

    1. ``<repo-id>/config.json`` — a flat layout, if some caller produces one.
    2. The revision directory containing ``.version``.
    3. The only revision directory, when there is exactly one.

    Anything else — two unmarked revisions — is ambiguous and raises rather
    than picking one, because silently loading the wrong revision of a
    detector changes its verdicts without changing anything visible.

    Raises:
        ModelNotAvailableError: nothing resolvable at the path, or the
            revision is ambiguous.
    """
    if os.path.isabs(model_name) or os.path.isdir(model_name):
        return model_name  # already a concrete path

    candidate = Path(MODEL_CACHE_DIR) / model_name

    if (candidate / "config.json").is_file():
        logger.info("Resolved model to local path: %s (from %s)", candidate, model_name)
        return str(candidate)

    revisions = (
        sorted(d for d in candidate.iterdir() if (d / "config.json").is_file())
        if candidate.is_dir()
        else []
    )

    if revisions:
        marked = [d for d in revisions if (d / ".version").is_file()]
        if len(marked) == 1:
            logger.info(
                "Resolved %s to revision %s (.version-marked)",
                model_name,
                marked[0].name,
            )
            return str(marked[0])
        if not marked and len(revisions) == 1:
            logger.info(
                "Resolved %s to revision %s (only revision present)",
                model_name,
                revisions[0].name,
            )
            return str(revisions[0])
        raise ModelNotAvailableError(
            f"Model {model_name!r} has an ambiguous revision at {candidate}: "
            f"{len(revisions)} revisions present "
            f"({', '.join(d.name for d in revisions)}) and "
            f"{len(marked)} marked with .version. Exactly one revision must "
            f"carry a .version file. Re-sync from {_S3_MODEL_SOURCE} or delete "
            f"the stale revision directories."
        )

    raise ModelNotAvailableError(
        f"Model {model_name!r} is not present at {candidate}. Detector weights "
        f"are served from {_S3_MODEL_SOURCE} and are never fetched from the "
        f"HuggingFace Hub. Populate the directory with:\n"
        f"    aws s3 sync {_S3_MODEL_SOURCE} {MODEL_CACHE_DIR}/\n"
        f"or point VIJIL_MODEL_DIR (currently {MODEL_CACHE_DIR!r}) at an "
        f"existing sync."
    )


class UnknownLabelError(RuntimeError):
    """A classifier emitted a label its own config does not declare.

    Raised rather than guessing. The alternative — treating an unrecognized
    label as the negative class — turns a broken detector into one that
    reports "safe" for everything, which is the worst failure a guard has.
    """


def positive_class_score(item: dict, config: object) -> float:
    """Return P(positive class) from one ``text-classification`` prediction.

    These detectors are binary, and index 1 is the flagged class by
    convention (injection, toxic, biased, harmful). What the pipeline *calls*
    that class depends entirely on the loaded config: ``id2label`` gives
    ``'injection'`` when the model card is well-formed and ``'LABEL_1'`` when
    it declares no mapping at all.

    Deriving the label from the config rather than matching a hardcoded list
    is the point. A hardcoded list has to be kept in sync by hand with every
    model any detector might load, and when it falls out of sync the failure
    is silent: an unmatched positive label falls through to ``1.0 - score``
    and inverts the detector. That is exactly how the prompt-injection guard
    came to report 0.0 for an input its own classifier scored 1.0.

    Raises:
        UnknownLabelError: the emitted label matches neither class.
    """
    label = item["label"]
    id2label = getattr(config, "id2label", None) or {}
    # Keys arrive as int or str depending on whether the config came from
    # JSON or from a live model object.
    normalized = {str(k): v for k, v in id2label.items()}
    positive = normalized.get("1", "LABEL_1")
    negative = normalized.get("0", "LABEL_0")

    if label == positive:
        return float(item["score"])
    if label == negative:
        return 1.0 - float(item["score"])
    raise UnknownLabelError(
        f"Classifier emitted label {label!r}, which is neither the positive "
        f"class ({positive!r}) nor the negative class ({negative!r}) declared "
        f"by the model's config. Refusing to guess: treating it as negative "
        f"would make this detector silently report safe for every input."
    )


def label_config(detector: object) -> object:
    """Return the config of the model that actually produced the prediction.

    Read from ``classifier.model`` rather than ``detector.model``. The Hybrid
    subclasses (PImbertHybrid, StereotypeEEOCHybrid) call super().__init__ to
    load the local model and build the pipeline, then rebind ``self.model`` to
    the *name* of their safeguard LLM — so ``self.model.config`` is an
    AttributeError on a str for exactly those classes.

    Reading from the pipeline is also the more honest source: the labels being
    interpreted belong to whichever model emitted the item, and that is the
    one the pipeline holds.
    """
    classifier = getattr(detector, "classifier", None)
    model = getattr(classifier, "model", None)
    if model is not None:
        return model.config
    # Fall back to the attribute for detectors that classify without a
    # pipeline. positive_class_score fails closed on an unusable config.
    return getattr(getattr(detector, "model", None), "config", None)


class HFBaseModel(DetectionMethod, ABC):
    """
    Abstract base class for detection models using Hugging Face transformers.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_tokenizer,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        except ValueError:
            # Some models ship tokenizer_config.json with a custom
            # tokenizer_class (e.g. "TokenizersBackend") that AutoTokenizer
            # cannot resolve. Fall back to loading tokenizer.json directly.
            #
            # Kept from #299 as defence in depth, though the artifact that
            # motivated it — stereotype-eeoc-detector — has since been
            # corrected in S3, and pipeline/validate.py in vijil-inference
            # now refuses to publish an unconstructible tokenizer_class.
            # The upstream fix is the real one; this catches anything that
            # slips past it.
            #
            # #299's branch downloaded tokenizer.json from the Hub when the
            # local file was absent. That is removed: this module no longer
            # reaches the network by any path, which is the property
            # air-gapped deployments depend on. A missing tokenizer.json is
            # now a hard failure naming the sync that fixes it.
            tokenizer_json = Path(resolved_tokenizer) / "tokenizer.json"
            if not tokenizer_json.exists():
                raise ModelNotAvailableError(
                    f"{resolved_tokenizer} declares a tokenizer_class "
                    f"AutoTokenizer cannot construct, and has no tokenizer.json "
                    f"to fall back to. Re-sync from {_S3_MODEL_SOURCE}."
                ) from None
            logger.info(
                "AutoTokenizer failed; loading tokenizer.json via "
                "PreTrainedTokenizerFast: %s",
                tokenizer_json,
            )
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_json),
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.model.config.pad_token_id

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


class HFBaseModelWithContext(HFBaseModel):
    """
    Abstract base class for context-dependent detection models using Hugging Face transformers
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        context: str | None = None,
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
