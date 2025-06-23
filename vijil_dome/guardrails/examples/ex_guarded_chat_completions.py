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

import time
import logging
import asyncio
import traceback
from litellm import completion as litellm_completion
from vijil_dome.guardrails import Guardrail, GuardrailConfig
from vijil_dome.guardrails.config_parser import convert_dict_to_guardrails
from vijil_dome.detectors import MODERATION_FLASHTXT_BANLIST

simple_guard = {
    "simple": {
        "type": "moderation",
        "exit-early": True,
        "methods": [MODERATION_FLASHTXT_BANLIST],
    }
}
default_guardrail_config = {
    "input-guards": [simple_guard],
    "output-guards": [simple_guard],
}

input_guardrail, output_guardrail = convert_dict_to_guardrails(default_guardrail_config)


class SimpleConfig(GuardrailConfig):
    def get_input_guardrail(self) -> Guardrail:
        return input_guardrail

    def get_output_guardrail(self) -> Guardrail:
        return output_guardrail


simple_config = SimpleConfig()

async def wrapped_llm_callback(*args, **kwargs):
    completion = litellm_completion(model="gpt-4o-mini", messages=[{"role": "user", "content": args[0]}])
    for field in completion.__dataclass_fields__.keys():
        if field == "generations":
            continue
        print(f"{field}: {getattr(completion, field)}")
    for gen in completion.generations:
        print(f"{gen}")
    print("\n")

    return completion


async def guarded_chat_completions(callback):
    async def guardrail_completion(*args, **kwargs):
        st_time = time.time()
        completion = litellm_completion(model="gpt-4o-mini", messages=[{"role": "user", "content": args[0]}])
        print(f"Time taken for completion: {time.time() - st_time}\n")
        print("Completion:", completion)
        return completion

    return guardrail_completion


async def main():
    try:
        chat_completions = await guarded_chat_completions(wrapped_llm_callback)
        await chat_completions("What is the capital of France?")
        await chat_completions("What is 2G1C?")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
