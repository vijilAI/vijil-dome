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

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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


# Keys of tokenizer_config.json that are safe to replay into a directly
# constructed PreTrainedTokenizerFast. Everything else in the file is either
# save-time bookkeeping ("backend", "is_local", "local_files_only"), resolved
# separately ("tokenizer_class", "added_tokens_decoder" — the latter already
# lives inside tokenizer.json), or unsupported by the installed transformers.
_REPLAYABLE_TOKENIZER_CONFIG_KEYS = frozenset({
    "bos_token",
    "clean_up_tokenization_spaces",
    "cls_token",
    "eos_token",
    "mask_token",
    "model_input_names",
    "model_max_length",
    "pad_token",
    "padding_side",
    "sep_token",
    "spaces_between_special_tokens",
    "truncation_side",
    "unk_token",
})


def _read_tokenizer_config(
    tokenizer_dir: Path, repo_id: str, local_only: bool
) -> dict[str, Any]:
    """Return the repo's ``tokenizer_config.json`` as a dict, or ``{}``.

    *tokenizer_dir* is the directory the caller's ``tokenizer.json`` came from,
    so config and vocabulary always describe the same artifact; *repo_id* is
    the Hub repo to download from when that directory has no config.

    Never raises: a missing or unreadable config only means the caller falls
    back to transformers' defaults, which is what happened before this existed.
    """
    candidate = tokenizer_dir / "tokenizer_config.json"
    if not candidate.exists():
        if local_only:
            return {}
        try:
            from huggingface_hub import hf_hub_download

            candidate = Path(hf_hub_download(repo_id, "tokenizer_config.json"))
        except Exception as exc:  # network, auth, missing file — all non-fatal
            logger.info("No tokenizer_config.json for %s: %s", repo_id, exc)
            return {}
    try:
        with open(candidate, encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        logger.warning("Could not read %s: %s", candidate, exc)
        return {}
    return config if isinstance(config, dict) else {}


def _replayable_tokenizer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Filter *config* down to kwargs this transformers version accepts.

    A tokenizer_config.json written by a newer transformers can carry keys the
    installed version chokes on — ``extra_special_tokens`` as a list is one
    (transformers 4.53 expects a mapping and raises AttributeError). Replaying
    a known-good subset keeps the repo's real settings (``model_max_length``,
    the special-token map) without inheriting future-format breakage.
    """
    return {
        key: value
        for key, value in config.items()
        if key in _REPLAYABLE_TOKENIZER_CONFIG_KEYS and value is not None
    }


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
        resolved = resolve_model_path(model_name)
        # When loading from a local S3-synced path, force local_files_only
        # so the model never falls back to HuggingFace Hub. This keeps
        # production pods offline — they never reach external hosts.
        is_local = os.path.isdir(resolved)
        effective_local_only = local_files_only or is_local
        logger.info(
            "Initializing Hugging Face model: %s (local=%s)", resolved, is_local
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved,
            local_files_only=effective_local_only,
            trust_remote_code=trust_remote_code,
        )
        model_tokenizer_name = tokenizer_name or model_name
        resolved_tokenizer = resolve_model_path(model_tokenizer_name)
        # A detector may name a tokenizer that lives in a *different* repo from
        # its weights — the ModernBERT finetunes all pass
        # tokenizer_name="answerdotai/ModernBERT-base". Each of them also ships
        # its own copy of that tokenizer beside the weights, so when the model
        # came from disk and the named tokenizer repo did not, read the
        # tokenizer out of the model directory. Without this, syncing a model
        # to disk actively breaks it: is_local flips local_files_only on, and
        # that offline flag is then applied to a base repo that was never
        # synced, so an air-gapped pod fails to initialize the detector on a
        # file it already has.
        if is_local and not os.path.isdir(resolved_tokenizer):
            if (Path(resolved) / "tokenizer.json").exists():
                logger.info(
                    "Tokenizer repo %s is not on disk; using the tokenizer "
                    "shipped with the local model at %s",
                    model_tokenizer_name,
                    resolved,
                )
                resolved_tokenizer = resolved
        # Offline-ness follows the tokenizer, not the model: a local model must
        # never force a Hub-only tokenizer into local_files_only mode.
        tokenizer_local_only = local_files_only or os.path.isdir(resolved_tokenizer)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_tokenizer,
                local_files_only=tokenizer_local_only,
                trust_remote_code=trust_remote_code,
            )
        except ValueError:
            # Some models ship tokenizer_config.json with a custom
            # tokenizer_class (e.g. "TokenizersBackend") that AutoTokenizer
            # cannot resolve. Fall back to loading the tokenizer.json
            # directly via PreTrainedTokenizerFast — and replay the repo's own
            # tokenizer_config on top, so this path keeps the model's real
            # model_max_length and special-token map instead of silently
            # falling back to transformers' defaults (model_max_length would
            # become VERY_LARGE_INTEGER and cls/sep/bos/eos would be None).
            tokenizer_json = Path(resolved_tokenizer) / "tokenizer.json"
            if not tokenizer_json.exists():
                if tokenizer_local_only:
                    raise
                from huggingface_hub import hf_hub_download
                tokenizer_json = Path(
                    hf_hub_download(model_tokenizer_name, "tokenizer.json")
                )
            # Read the config from wherever tokenizer.json came from, so a
            # local vocabulary is never described by a Hub config (or the
            # reverse) when only one of the two is present locally.
            config = _read_tokenizer_config(
                tokenizer_json.parent, model_tokenizer_name, tokenizer_local_only
            )
            replayed = _replayable_tokenizer_kwargs(config)
            logger.info(
                "AutoTokenizer failed (tokenizer_class=%s); loading tokenizer.json "
                "via PreTrainedTokenizerFast with %d replayed config keys: %s",
                config.get("tokenizer_class", "unknown"),
                len(replayed),
                tokenizer_json,
            )
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_json),
                **replayed,
            )

        # A tokenizer with no pad token at all cannot pad, so every batched
        # call raises — fill that in from the model config when the tokenizer
        # itself defines none (the fallback path above loses it whenever the
        # repo ships no tokenizer_config.json to replay).
        #
        # A *mismatched* pad id is a different matter: these classifiers are
        # always called with an attention mask, so padded positions are masked
        # out of the result either way. Measured on the PI detector — scoring
        # the same prompts single and batched with the pad id forced to a
        # wrong token left every score bit-identical. So the mismatch is
        # logged as the tokenizer/model pairing smell it is, not treated as a
        # scoring hazard, and never "corrected" by overwriting the tokenizer's
        # own pad token with a model-config id that may map to another token.
        model_pad_id = getattr(self.model.config, "pad_token_id", None)
        if self.tokenizer.pad_token_id is None:
            if model_pad_id is None:
                logger.warning(
                    "Neither tokenizer nor model config defines a pad token for %s; "
                    "batched inference will fail.",
                    resolved,
                )
            else:
                self.tokenizer.pad_token_id = model_pad_id
        elif model_pad_id is not None and self.tokenizer.pad_token_id != model_pad_id:
            logger.info(
                "Pad token mismatch for %s: tokenizer id=%s (%r) vs model config "
                "id=%s. Harmless while inputs carry an attention mask, but it "
                "usually means the tokenizer and the weights came from "
                "different revisions.",
                resolved,
                self.tokenizer.pad_token_id,
                self.tokenizer.pad_token,
                model_pad_id,
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
