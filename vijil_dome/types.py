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

from typing import List, Optional, Union

from pydantic import BaseModel, model_validator

Embeddings = List[float]

Sentences = Union[str, List[str]]


class DomePayload(BaseModel):
    """Structured input for Dome guardrails.

    Supports plain text (backward compatible), prompt/response pairs,
    and future multimodal content.

    Use ``text`` for legacy plain-string input.
    Use ``prompt`` and/or ``response`` for structured LLM input/output pairs.
    These two modes are mutually exclusive.
    """

    text: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    # This can be augmented with additional fields in the future, e.g. for multimodal content, reasoning etc. 

    @model_validator(mode="after")
    def _validate_has_content(self) -> "DomePayload":
        if self.text is None and self.prompt is None and self.response is None:
            raise ValueError(
                "DomePayload must have at least one of: text, prompt, response"
            )
        if self.text is not None and (
            self.prompt is not None or self.response is not None
        ):
            raise ValueError(
                "Cannot set 'text' alongside 'prompt'/'response'. "
                "Use 'text' for plain strings or 'prompt'/'response' for structured input."
            )
        return self

    @property
    def query_string(self) -> str:
        """Flattened string representation for legacy/text-based detectors."""
        if self.text is not None:
            return self.text
        parts = []
        if self.prompt:
            parts.append(f"Input: {self.prompt}")
        if self.response:
            parts.append(f"Output: {self.response}")
        return "\n".join(parts)

    @classmethod
    def coerce(cls, value: "Union[str, DomePayload]") -> "DomePayload":
        """Accept str or DomePayload, always return DomePayload. This is used to ensure backward compatibility with older versions of Dome that passes plain strings."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value)
        raise TypeError(f"Cannot coerce {type(value)} to DomePayload")
