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

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from vijil_dome.types import Sentences, Embeddings


@dataclass
class EmbeddingsItem:
    text: str
    embeddings: Optional[Embeddings] = None
    meta: Dict = field(default_factory=dict)


class AbstractEmbedder(ABC):
    @abstractmethod
    async def embeddings(self, sentences: Sentences) -> List[Embeddings]:
        pass
