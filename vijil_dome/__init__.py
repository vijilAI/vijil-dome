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

# Trust runtime — lazy import to avoid pulling in cryptography when not needed
def __getattr__(name: str):
    if name == "trust":
        from vijil_dome import trust as _trust
        return _trust
    if name == "TrustRuntime":
        from vijil_dome.trust.runtime import TrustRuntime
        return TrustRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Dome",
    "DomePayload",
    "BatchScanResult",
    "create_dome_config",
    "get_default_config",
    "TrustRuntime",
    "trust",
]
