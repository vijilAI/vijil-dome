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
import random
import string
from vijil_dome.detectors.methods.pii_presidio import *  # noqa: F403
from vijil_dome.detectors.methods.secret_detector import *  # noqa: F403
from vijil_dome.detectors import (
    PRIVACY_PRESIDIO,
    DETECT_SECRETS,
    DetectionFactory,
    DetectionCategory,
)


@pytest.mark.asyncio
async def test_privacy_detection():
    # PII Detection/ By default, PII is censored, not flagged.
    pii_presidio = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Privacy, PRIVACY_PRESIDIO
    )
    query = "How many days make up a year?"
    result = await pii_presidio(query)
    assert result.hit == False and result.result["response_string"] == query
    query = "You can reach me at dwayne.johnson@gmail.com"
    result = await pii_presidio(query)
    assert result.hit == False and result.result["response_string"] != query

    # Repeat, but this time, we set anonymize to False, so the PII is flagged, not censored.
    pii_detector = DetectionFactory.get_detector(
        DetectionCategory.Privacy, PRIVACY_PRESIDIO, anonymize=False
    )
    result = await pii_detector.detect_with_time(query)
    assert result.hit == True


@pytest.mark.asyncio
async def test_secret_detection():
    # Secret Detection. Note that the default setting does not flag strings with secrets, it censors them.
    secret_detector = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Privacy, DETECT_SECRETS
    )

    result = await secret_detector("This is a regular string")
    assert result.hit == False

    # We now generate a fake OpenAI API key for testing purposes
    random.seed(42)  # For reproducibility in tests
    rand = lambda n: "".join(random.choices(string.ascii_letters + string.digits, k=n))
    fake_key = f"sk-{rand(20)}T3BlbkFJ{rand(20)}"
    string_with_key = "OPENAI_API_KEY = " + fake_key

    result = await secret_detector(string_with_key)
    # Since the default setting censors secrets, we expect the hit to be False, but the response string to be modified
    assert result.hit == False
    assert (result.result["response_string"] != string_with_key) and (
        fake_key not in result.result["response_string"]
    )

    # Now, we set the detector to flag secrets, not censor them.
    secret_detector = DetectionFactory.get_detector(
        DetectionCategory.Privacy, DETECT_SECRETS, censor=False
    )
    result = await secret_detector.detect_with_time(string_with_key)
    assert result.hit == True
