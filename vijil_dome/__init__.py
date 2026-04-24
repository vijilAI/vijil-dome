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

from .Dome import Dome, create_dome_config, BatchScanResult
from .defaults import get_default_config
from .types import DomePayload

# Trust runtime — lazy imports to avoid pulling in cryptography when not needed
def __getattr__(name: str):
    _lazy = {
        "trust": ("vijil_dome", "trust"),
        "TrustRuntime": ("vijil_dome.trust.runtime", "TrustRuntime"),
        "secure_agent": ("vijil_dome.trust.adapters.auto", "secure_agent"),
    }
    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(module_path)
        return mod if attr == module_path.split(".")[-1] else getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Dome",
    "DomePayload",
    "BatchScanResult",
    "create_dome_config",
    "get_default_config",
    "TrustRuntime",
    "secure_agent",
    "trust",
]
