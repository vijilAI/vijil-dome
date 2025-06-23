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

import pytest

from vijil_dome.detectors.methods.hhem_hallucination import *  # noqa: F403
from vijil_dome.detectors.methods.factcheck_roberta import *  # noqa: F403
from vijil_dome.detectors.methods.llm_models import *  # noqa: F403

from vijil_dome.detectors import (
    HHEM,
    FACTCHECK_ROBERTA,
    HALLUCINATION_LLM,
    FACTCHECK_LLM,
    DetectionFactory,
    DetectionCategory,
)


@pytest.mark.asyncio
async def test_integrity_detection():
    # Hallucination detection via HHEM
    hhem_detect_with_time = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Integrity,
        HHEM,
        context="I am in California.",
        trust_remote_code=True,
    )
    detection_result = await hhem_detect_with_time("I am in United States.")
    assert detection_result.hit == False
    # You can also get the detector and add the context in later
    hhem_detector = DetectionFactory.get_detector(DetectionCategory.Integrity, HHEM)
    hhem_detector.add_context("I am in United States.")
    detection_result = await hhem_detector.detect_with_time("I am in California.")
    assert detection_result.hit == True


@pytest.mark.asyncio
async def test_integrity_detection_llm():
    # Hallucination detection via LLM Prompt-Engineering
    llm_hallucination_detect_with_time = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Integrity,
        HALLUCINATION_LLM,
        context="I am in California.",
    )
    detection_result = await llm_hallucination_detect_with_time(
        "I am in United States."
    )
    assert detection_result.hit == False
    detector = DetectionFactory.get_detector(
        DetectionCategory.Integrity, HALLUCINATION_LLM
    )
    detector.add_context("I am in United States.")
    detection_result = await detector.detect_with_time("I am in California.")
    assert detection_result.hit == True


@pytest.mark.asyncio
async def test_integrity_detection_roberta():
    # Fact checking via Roberta factcheck Model
    roberta_detect_with_time = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Integrity,
        FACTCHECK_ROBERTA,
        context="Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time.",
    )
    detection_result = await roberta_detect_with_time(
        "Albert Einstein worked in the field of physics."
    )
    assert detection_result.hit == False
    roberta_detector = DetectionFactory.get_detector(
        DetectionCategory.Integrity, FACTCHECK_ROBERTA
    )
    roberta_detector.add_context("I am in United States.")
    detection_result = await roberta_detector.detect_with_time("I am in California.")
    assert detection_result.hit == True


@pytest.mark.asyncio
async def test_integrity_factcheck_llm():
    # Fact checking via LLM Prompt engineering
    factcheck_llm_detect_with_time = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Integrity,
        FACTCHECK_LLM,
        context="I am in California.",
    )
    await factcheck_llm_detect_with_time("I am in United States.")
    factcheck_llm_detector = DetectionFactory.get_detector(
        DetectionCategory.Integrity, FACTCHECK_LLM
    )
    factcheck_llm_detector.add_context("I am in United States.")
    detection_result = await factcheck_llm_detector.detect_with_time(
        "I am in California."
    )
    assert detection_result.hit == True
