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

# Exposing all method files for ease of import.
# Modules that support both local (torch) and remote (no torch) modes
# are always importable. Torch-only modules are gated behind a check
# so that ``pip install vijil-dome`` (without the ``local`` extra)
# still loads the remote / API-only detectors.
__all__ = [
    "flashtext_kw_banlist",
    "llm_models",
    "openai_models",
    "pii_presidio",
    "secret_detector",
    "encoding_heuristics",
    "gpt_oss_safeguard_policy",
    "policy_sections_detector",
    # These three have lazy torch imports and work without torch
    # (remote + safeguard modes stay available).
    "pi_hf_mbert",
    "toxicity_mbert",
    "stereotype_eeoc",
]

# Modules that require torch for all functionality.
try:
    import torch  # noqa: F401

    __all__.extend([
        "factcheck_roberta",
        "hhem_hallucination",
        "jb_perplexity_heuristics",
        "pi_hf_deberta",
        "toxicity_deberta",
        "embedding_models",
    ])
except ImportError:
    pass

try:
    import googleapiclient  # noqa: F401

    __all__.append("perspective")
except ImportError:
    pass
