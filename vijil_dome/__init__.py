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

# Lazy imports — avoid pulling in heavy deps when not needed
def __getattr__(name: str):
    _lazy = {
        # Trust runtime
        "trust": ("vijil_dome", "trust"),
        "TrustRuntime": ("vijil_dome.trust.runtime", "TrustRuntime"),
        "secure_agent": ("vijil_dome.trust.adapters.auto", "secure_agent"),
        # Controls
        "VijilDome": ("vijil_dome.core", "VijilDome"),
        "ControlEngine": ("vijil_dome.controls.engine", "ControlEngine"),
        "control": ("vijil_dome.controls.decorator", "control"),
        "register_evaluator": ("vijil_dome.controls.evaluators", "register_evaluator"),
        "Evaluator": ("vijil_dome.controls.evaluators.base", "Evaluator"),
        "EvaluatorResult": ("vijil_dome.controls.evaluators.base", "EvaluatorResult"),
        "ControlViolationError": ("vijil_dome.controls.errors", "ControlViolationError"),
        "ControlSteerError": ("vijil_dome.controls.errors", "ControlSteerError"),
        # Adapter registry
        "register_adapter": ("vijil_dome.trust.adapters.base", "register_adapter"),
        "BaseAdapter": ("vijil_dome.trust.adapters.base", "BaseAdapter"),
    }
    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(module_path)
        return mod if attr == module_path.split(".")[-1] else getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Existing
    "Dome",
    "DomePayload",
    "BatchScanResult",
    "create_dome_config",
    "get_default_config",
    "TrustRuntime",
    "secure_agent",
    "trust",
    # Controls
    "VijilDome",
    "ControlEngine",
    "control",
    "register_evaluator",
    "Evaluator",
    "EvaluatorResult",
    "ControlViolationError",
    "ControlSteerError",
    # Adapter registry
    "register_adapter",
    "BaseAdapter",
]
