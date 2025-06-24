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
    PI_MBERT,
    PRIVACY_PRESIDIO,
    MODERATION_DEBERTA,
    MODERATION_FLASHTXT_BANLIST,
)


class DefaultDomeConfig:
    def __init__(self):
        self.security_default = {
            "security_default": {
                "type": "security",
                "methods": [PI_MBERT],
            }
        }
        self.moderation_default = {
            "moderation_default": {
                "type": "moderation",
                "methods": [MODERATION_FLASHTXT_BANLIST, MODERATION_DEBERTA],
            }
        }

        self.privacy_default = {
            "privacy_default": {"type": "privacy", "methods": [PRIVACY_PRESIDIO]}
        }

        self.default_guardrail_config = {
            "input-guards": [self.security_default, self.moderation_default],
            "output-guards": [self.moderation_default, self.privacy_default],
        }


def get_default_config():
    return DefaultDomeConfig().default_guardrail_config
