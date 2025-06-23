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

from vijil_dome.detectors.methods.llm_models import GenericLLMDetector


@pytest.mark.asyncio
async def test_generic_llm_detector():
    # The generic LLM allows for custom prompt engineering. The only requirements are:
    # 1. The system prompt must contain '$query_string' --> this is where the actual prompt gets inserted
    # 2. You must specify a list of trigger words to the detector. The trigger word corresponds to the expected response from the LLM when the output is caught by the detector
    # Note: the detection logic checks if the trigger word was in the response. IE, if your llm responds with SAFE or UNSAFE, using "safe" as a trigger word will result in every response being marked as a hit

    # Example - A GPT-4 based detector to detect if a query is related to cheese

    prompt_template = """
    You are acting as a dietary assistant whose job is to identify if a query contains content that can be related to cheese in any way. 
    If the query is related to cheese, you must say "UNSAFE". Otherwise you must say "SAFE". Only respond with "SAFE" or "UNSAFE".
    Query: '$query_string'
    Assessment:
    """

    # For consistency, the response is converted to lower case when assessing for trigger words. As such, the trigger words must be lowercase as well
    # again, using "safe" as a trigger word here would result in a detection of True for both safe and unsafe responses
    trigger_words = ["unsafe"]

    # By default, OAI is the hub and GPT-4 is the model. You can change them if needed via the hub_name and model_name keyword arguments
    llm_detector = GenericLLMDetector(prompt_template, trigger_words)

    query_string = "I really love pecorino"
    result = await llm_detector.detect_with_time(query_string)
    assert result.hit == True

    query_string = "What is the capital of france?"
    result = await llm_detector.detect_with_time(query_string)
    assert result.hit == False
