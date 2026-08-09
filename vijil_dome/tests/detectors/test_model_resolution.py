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

"""Detector weights resolve from the local S3-synced directory, or not at all.

The property under test is a negative one: no code path reaches the
HuggingFace Hub. That matters for air-gapped and egress-restricted
deployments, where a silent fallback turns a misconfigured mount into an
outbound network call rather than an error.
"""

import json
from pathlib import Path

import pytest

from vijil_dome.detectors.utils import hf_model
from vijil_dome.detectors.utils.hf_model import (
    ModelNotAvailableError,
    label_config,
    UnknownLabelError,
    positive_class_score,
    resolve_model_path,
)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Point MODEL_CACHE_DIR at an empty temp dir for the duration of a test."""
    monkeypatch.setattr(hf_model, "MODEL_CACHE_DIR", str(tmp_path))
    return tmp_path


def _make_model(root: Path, repo_id: str) -> Path:
    """Create a directory that looks like a synced model (has config.json)."""
    d = root / repo_id
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "bert"}))
    return d


class TestResolvesLocally:
    def test_resolves_a_synced_model_to_its_local_path(self, model_dir):
        expected = _make_model(model_dir, "vijil/vijil_dome_prompt_injection_detection")
        got = resolve_model_path("vijil/vijil_dome_prompt_injection_detection")
        assert got == str(expected)

    def test_resolves_a_third_party_base_model(self, model_dir):
        # Base tokenizers and encoders are not under the vijil/ prefix, but
        # they are loaded through the same resolver — the mBERT detectors pass
        # answerdotai/ModernBERT-base as tokenizer_name. If the sync only
        # covered vijil/, this is the call that would reach the Hub.
        expected = _make_model(model_dir, "answerdotai/ModernBERT-base")
        assert resolve_model_path("answerdotai/ModernBERT-base") == str(expected)

    def test_absolute_paths_pass_through_untouched(self, model_dir):
        d = _make_model(model_dir, "vijil/some-detector")
        assert resolve_model_path(str(d)) == str(d)


class TestRefusesToReachTheHub:
    def test_missing_model_raises_instead_of_returning_a_hub_id(self, model_dir):
        # The old behaviour returned the repo id so from_pretrained would
        # download it. Returning the id at all is the bug: it is what made an
        # unsynced deployment indistinguishable from a synced one.
        with pytest.raises(ModelNotAvailableError):
            resolve_model_path("vijil/not-synced")

    def test_directory_without_config_json_is_not_a_model(self, model_dir):
        # A partial or interrupted sync leaves the directory but not the
        # weights. Treating that as success yields a confusing failure deeper
        # inside transformers.
        (model_dir / "vijil" / "half-synced").mkdir(parents=True)
        with pytest.raises(ModelNotAvailableError):
            resolve_model_path("vijil/half-synced")

    def test_error_names_the_path_and_the_fix(self, model_dir):
        # The failure a caller sees would otherwise be a network or model-card
        # error naming neither the model directory nor the sync that fills it.
        with pytest.raises(ModelNotAvailableError) as exc:
            resolve_model_path("vijil/not-synced")
        message = str(exc.value)
        assert "vijil/not-synced" in message
        assert str(model_dir) in message
        assert "s3://vijil-inference/models/" in message
        assert "aws s3 sync" in message

    def test_error_mentions_that_hub_is_never_used(self, model_dir):
        with pytest.raises(ModelNotAvailableError) as exc:
            resolve_model_path("vijil/not-synced")
        assert "HuggingFace" in str(exc.value)


class TestPositiveClassScore:
    """Label handling derives from the model's own config, and fails closed.

    The bug this pins: a hardcoded positive-label tuple that a model's real
    label ('injection') did not appear in, so every prediction fell through
    to `1.0 - score` and the prompt-injection guard reported 0.0 for an input
    its classifier scored 1.0.
    """

    class _Config:
        def __init__(self, id2label):
            self.id2label = id2label

    SEMANTIC = _Config({"0": "benign", "1": "injection"})
    NUMERIC = _Config({"0": 0, "1": 1})
    ABSENT = _Config(None)

    def test_semantic_positive_label_is_not_inverted(self):
        # The regression. 'injection' is the positive class; a hardcoded
        # (1, "1", "LABEL_1") tuple misses it and returns 1.0 - 1.0 = 0.0.
        score = positive_class_score(
            {"label": "injection", "score": 1.0}, self.SEMANTIC
        )
        assert score == 1.0

    def test_semantic_negative_label_is_inverted(self):
        score = positive_class_score({"label": "benign", "score": 0.99}, self.SEMANTIC)
        assert score == pytest.approx(0.01)

    def test_numeric_label_config_still_works(self):
        # The malformed HuggingFace card shape. It has to keep working, since
        # that is what a Hub-loaded model emits.
        assert positive_class_score({"label": 1, "score": 0.8}, self.NUMERIC) == 0.8

    def test_missing_id2label_falls_back_to_transformers_default(self):
        # No mapping in the config means transformers emits LABEL_0/LABEL_1.
        assert (
            positive_class_score({"label": "LABEL_1", "score": 0.7}, self.ABSENT) == 0.7
        )
        assert positive_class_score(
            {"label": "LABEL_0", "score": 0.7}, self.ABSENT
        ) == pytest.approx(0.3)

    def test_unknown_label_raises_rather_than_reporting_safe(self):
        # Fail closed. Treating an unrecognised label as the negative class is
        # what makes a broken detector look like a clean scan.
        with pytest.raises(UnknownLabelError):
            positive_class_score({"label": "toxic", "score": 1.0}, self.SEMANTIC)

    def test_unknown_label_error_names_both_classes(self):
        with pytest.raises(UnknownLabelError) as exc:
            positive_class_score({"label": "weird", "score": 1.0}, self.SEMANTIC)
        assert "injection" in str(exc.value) and "benign" in str(exc.value)


class TestLabelConfig:
    """The config comes from the pipeline, not the detector attribute.

    The Hybrid subclasses rebind self.model to their safeguard LLM's *name*
    after super().__init__ has loaded the real one, so reading
    self.model.config raises AttributeError on a str for exactly those
    classes — which is how this shipped red.
    """

    class _Config:
        id2label = {"0": "benign", "1": "injection"}

    class _Model:
        config = None

    def test_reads_from_the_classifier(self):
        detector = type("D", (), {})()
        model = self._Model()
        model.config = self._Config()
        detector.classifier = type("P", (), {"model": model})()
        # The Hybrid failure mode: self.model rebound to a plain string.
        detector.model = "openai/gpt-oss-safeguard-20b"
        assert label_config(detector) is model.config

    def test_falls_back_to_the_model_attribute(self):
        # A detector that classifies without a pipeline.
        detector = type("D", (), {})()
        model = self._Model()
        model.config = self._Config()
        detector.model = model
        assert label_config(detector) is model.config

    def test_returns_none_when_there_is_no_usable_config(self):
        # positive_class_score fails closed on this rather than guessing.
        detector = type("D", (), {})()
        detector.model = "just-a-name"
        assert label_config(detector) is None

    def test_none_config_still_fails_closed_on_a_semantic_label(self):
        with pytest.raises(UnknownLabelError):
            positive_class_score({"label": "injection", "score": 1.0}, None)
