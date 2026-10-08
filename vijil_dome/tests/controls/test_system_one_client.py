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

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vijil_dome.controls.evaluators.system_one_client import (
    SystemOneClient,
    SystemOneError,
)


def _mock_response(json_body: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    response.text = str(json_body)
    return response


@pytest.mark.asyncio
async def test_ask_noul_batch_parses_answers():
    client = SystemOneClient(api_key="test-key")
    response = _mock_response(
        {
            "model": "jev-1.13.0",
            "answers": {
                "RULE-1": {"type": "noul", "noul": 0.93},
                "RULE-2": {"type": "noul", "noul": 0.02},
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    with patch.object(
        httpx.AsyncClient, "post", AsyncMock(return_value=response)
    ) as mock_post:
        result = await client.ask_noul_batch(
            "some turn text",
            {
                "RULE-1": {"instructions": "Did this violate rule 1?"},
                "RULE-2": {"instructions": "Did this violate rule 2?"},
            },
        )

    assert result == {"RULE-1": 0.93, "RULE-2": 0.02}
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert call_kwargs["json"]["questions"]["RULE-1"]["type"] == "noul"


@pytest.mark.asyncio
async def test_ask_noul_batch_empty_questions_short_circuits():
    client = SystemOneClient(api_key="test-key")
    with patch.object(httpx.AsyncClient, "post", AsyncMock()) as mock_post:
        result = await client.ask_noul_batch("text", {})
    assert result == {}
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_missing_api_key_raises():
    # Explicit "" rather than None: None falls back to OPENROUTER_API_KEY from
    # the environment, so this test would silently make a real unmocked
    # request instead of exercising the missing-key branch on any machine
    # that happens to have that variable set (e.g. this repo's own .env).
    client = SystemOneClient(api_key="")
    with pytest.raises(SystemOneError, match="No API key"):
        await client.ask_noul_batch("text", {"R": {"instructions": "x?"}})


@pytest.mark.asyncio
async def test_non_2xx_response_raises():
    client = SystemOneClient(api_key="test-key")
    response = _mock_response({"error": "bad request"}, status_code=400)
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
        with pytest.raises(SystemOneError, match="HTTP 400"):
            await client.ask_noul_batch("text", {"R": {"instructions": "x?"}})


@pytest.mark.asyncio
async def test_missing_answers_key_raises():
    client = SystemOneClient(api_key="test-key")
    response = _mock_response({"model": "jev-1.13.0"})
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=response)):
        with pytest.raises(SystemOneError, match="missing 'answers'"):
            await client.ask_noul_batch("text", {"R": {"instructions": "x?"}})


@pytest.mark.asyncio
async def test_transport_error_wrapped():
    client = SystemOneClient(api_key="test-key")
    with patch.object(
        httpx.AsyncClient,
        "post",
        AsyncMock(side_effect=httpx.ConnectTimeout("timed out")),
    ):
        with pytest.raises(SystemOneError, match="request failed"):
            await client.ask_noul_batch("text", {"R": {"instructions": "x?"}})
