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

# Exposing all method files for ease of import
__all__ = [
    "factcheck_roberta",
    "flashtext_kw_banlist",
    "hhem_hallucination",
    "jb_perplexity_heuristics",
    "llm_models",
    "openai_models",
    "pi_hf_deberta",
    "pi_hf_mbert",
    "pii_presidio",
    "toxicity_deberta",
    "embedding_models",
    "secret_detector",
]

try:
    import googleapiclient  # noqa: F401
    __all__.append("perspective")
except ImportError:
    pass
