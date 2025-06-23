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
    DETECT_SECRETS,
    DetectionCategory,
    DetectionMethod,
    register_method,
    DetectionResult,
)
from detect_secrets.core.scan import scan_line, PotentialSecret
from detect_secrets.settings import transient_settings


def _censor_secret(text: str, secret: PotentialSecret):
    if secret.secret_value and len(secret.secret_value) > 1:
        censored = secret.secret_value[0] + (len(secret.secret_value) - 1) * "*"
        text = text.replace(secret.secret_value, censored)
    return text


@register_method(DetectionCategory.Privacy, DETECT_SECRETS)
class SecretDetector(DetectionMethod):
    def __init__(self, censor=True) -> None:
        super().__init__()

        # If censoring is enabled, return a censored version of the response
        self.censor = censor

        plugins = [
            "ArtifactoryDetector",
            "AWSKeyDetector",
            "AzureStorageKeyDetector",
            "BasicAuthDetector",
            "CloudantDetector",
            "DiscordBotTokenDetector",
            "GitHubTokenDetector",
            "GitLabTokenDetector",
            "IbmCloudIamDetector",
            "IbmCosHmacDetector",
            "IPPublicDetector",
            "JwtTokenDetector",
            "KeywordDetector",
            "MailchimpDetector",
            "NpmDetector",
            "OpenAIDetector",
            "PrivateKeyDetector",
            "PypiTokenDetector",
            "SendGridDetector",
            "SlackDetector",
            "SoftlayerDetector",
            "SquareOAuthDetector",
            "StripeDetector",
            "TelegramBotTokenDetector",
            "TwilioKeyDetector",
        ]  # These are the supported plugins to detect secrets
        self._plugins_used = {
            "plugins_used": [{"name": plugin} for plugin in plugins],
        }

        self.blocked_response_string = f"Method:{DETECT_SECRETS}"

    async def detect(self, query_string: str) -> DetectionResult:
        flagged = False
        secrets = []
        with transient_settings(self._plugins_used):
            for secret in scan_line(query_string):
                flagged = True
                secrets.append(secret)

        response_string = query_string
        if self.censor:
            flagged = False
            for secret in secrets:
                response_string = _censor_secret(response_string, secret)

        return flagged, {
            "type": type(self),
            "secrets": secrets,
            "query_string": query_string,
            "response_string": self.blocked_response_string
            if flagged
            else response_string,
        }
