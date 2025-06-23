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
    PRIVACY_PRESIDIO,
    DetectionMethod,
    DetectionResult,
    DetectionCategory,
    register_method,
)
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from typing import Optional, List
import logging
import os

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Privacy, PRIVACY_PRESIDIO)
class PresidioDetector(DetectionMethod):
    def __init__(
        self,
        score_threshold: float = 0.5,
        anonymize: bool = True,
        allow_list_files: Optional[List[str]] = None,
    ) -> None:
        self.allow_list = []
        self.anonymizer = None
        self.anonymization_operators = None
        try:
            if allow_list_files is not None:
                for allow_list_file in allow_list_files:
                    with open(allow_list_file, "r") as f:
                        file_allow_list = [line.rstrip("\n") for line in f.readlines()]
                        self.allow_list.extend(file_allow_list)

            self.score_threhsold = score_threshold

            config_file = os.path.join(
                os.path.dirname(__file__), "configs/presidio", "default_config.yaml"
            )
            provider = NlpEngineProvider(conf_file=config_file)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=["en"]
            )
            self.anonymize = anonymize

            if self.anonymize:
                self.anonymizer = AnonymizerEngine()
                self.anonymization_operators = {
                    "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                }
            self.blocked_response_string = f"Method:{PRIVACY_PRESIDIO}"

        except Exception as e:
            logger.error(f"Error loading Presidio PII detector: {str(e)}")
            raise

    async def detect(self, query_string: str) -> DetectionResult:
        # For now, we only support en, but this could be extended later
        analysis_result = self.analyzer.analyze(
            text=query_string, language="en", allow_list=self.allow_list
        )
        flagged = False
        for possible_pii in analysis_result:
            flagged = possible_pii.score > self.score_threhsold
            if flagged:
                break

        # If anonymization is enabled, detect includes the santized query in the results
        if self.anonymize:
            # This loop is simply to comply with type-checking. Can be removed if we don't care about mypy errors
            # See: https://github.com/microsoft/presidio/issues/1396
            results_for_anonymization = []
            for result in analysis_result:
                results_for_anonymization.append(
                    RecognizerResult(
                        entity_type=result.entity_type,
                        start=result.start,
                        end=result.end,
                        score=result.score,
                    )
                )

            sanitized_query = self.anonymizer.anonymize(
                text=query_string,
                analyzer_results=results_for_anonymization,
                operators=self.anonymization_operators,
            )

            # If anonymized is enabled, PIIs aren't flagged but only sanitized.
            return False, {
                "type": type(self),
                "response_string": sanitized_query.text,
                "pii_analysis": analysis_result,
                "query_string": query_string,
            }

        return flagged, {
            "type": type(self),
            "pii_analysis": analysis_result,
            "query_string": query_string,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
