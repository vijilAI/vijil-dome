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

from vijil_dome.detectors import (
    JB_LENGTH_PER_PERPLEXITY,
    JB_PREFIX_SUFFIX_PERPLEXITY,
    register_method,
    DetectionMethod,
    DetectionCategory,
)
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Heuristic-based detectors for jailbreaks using GPT-2 perplexity
# See: https://docs.nvidia.com/nemo/guardrails/user_guides/guardrails-library.html#length-per-perplexity
# And: https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/library/jailbreak_detection/heuristics/checks.py


class PerplexityBaseModel(DetectionMethod):
    def __init__(
        self,
        model_id: str = "gpt2-large",
        batch_size: int = 16,
        stride_length: int = 512,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.stride_length = stride_length

    def get_perplexity(self, input_string: str):
        encodings = self.tokenizer(input_string, return_tensors="pt")

        max_length = self.model.config.n_positions
        seq_len = encodings.input_ids.size(1)

        nlls = list()
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride_length):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        perplexity = torch.exp(torch.stack(nlls).mean())

        return perplexity.cpu().detach().numpy().item()


# Length per perplexity heuristic
# See: https://docs.nvidia.com/nemo/guardrails/user_guides/guardrails-library.html#length-per-perplexity
@register_method(DetectionCategory.Security, JB_LENGTH_PER_PERPLEXITY)
class LengthPerPerplexityModel(PerplexityBaseModel):
    def __init__(
        self,
        model_id: str = "gpt2-large",
        batch_size: int = 16,
        stride_length: int = 512,
        threshold: float = 89.79,
    ):
        super().__init__(
            model_id=model_id, batch_size=batch_size, stride_length=stride_length
        )
        self.threshold = threshold
        self.blocked_response_string = f"Method:{JB_LENGTH_PER_PERPLEXITY}"

    async def detect(self, query_string: str):
        # Handle empty inputs
        if not len(query_string):
            return False, {"type": type(self)}
        perplexity = self.get_perplexity(query_string)
        score = len(query_string) / perplexity
        flagged = score > self.threshold
        return flagged, {
            "type": type(self),
            "perplexity": perplexity,
            "length": len(query_string),
            "score": score,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }


# Prefix-Suffix perplexity heuristic
# See: https://docs.nvidia.com/nemo/guardrails/user_guides/guardrails-library.html#prefix-and-suffix-perplexity
@register_method(DetectionCategory.Security, JB_PREFIX_SUFFIX_PERPLEXITY)
class PrefixSuffixPerplexityModel(PerplexityBaseModel):
    def __init__(
        self,
        model_id: str = "gpt2-large",
        batch_size: int = 16,
        stride_length: int = 512,
        prefix_threshold: float = 1845.65,
        suffix_threshold: float = 1845.65,
        prefix_length: int = 20,
        suffix_length: int = 20,
    ):
        super().__init__(
            model_id=model_id, batch_size=batch_size, stride_length=stride_length
        )
        self.prefix_threshold = prefix_threshold
        self.suffix_threshold = suffix_threshold
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.blocked_response_string = f"Method:{JB_PREFIX_SUFFIX_PERPLEXITY}"

    async def detect(self, query_string: str):
        # Handle empty inputs
        if not len(query_string):
            return False, {"type": type(self)}
        # get prefix and suffix
        wordlist = query_string.strip().split()
        prefix = " ".join(wordlist[: max(len(wordlist), self.prefix_length)])
        suffix = " ".join(wordlist[max(0, len(wordlist) - self.suffix_length) :])
        prefix_perplexity_score = self.get_perplexity(prefix)
        suffix_perplexity_score = self.get_perplexity(suffix)
        flagged = False
        if (prefix_perplexity_score > self.prefix_threshold) or (
            suffix_perplexity_score > self.suffix_threshold
        ):
            flagged = True
        return flagged, {
            "type": type(self),
            "prefix_perplexity": prefix_perplexity_score,
            "suffix_perplexity": suffix_perplexity_score,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
