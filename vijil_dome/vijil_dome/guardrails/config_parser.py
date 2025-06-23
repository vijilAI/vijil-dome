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

from vijil_dome.guardrails import Guard, Guardrail
from vijil_dome.detectors.methods import *  # noqa: F403
from vijil_dome.detectors import DetectionCategory, DetectionFactory
import toml
from typing import Tuple, Dict, Any  # noqa: F401

GUARDRAIL_CATEGORY_MAPPING = {
    "security": DetectionCategory.Security,
    "moderation": DetectionCategory.Moderation,
    "privacy": DetectionCategory.Privacy,
    "integrity": DetectionCategory.Integrity,
}

EARLY_EXIT = "early-exit"
PARALLELIZED = "run-parallel"


def create_detector_for_guard(
    detector_name: str, detector_type: str, detector_config_dict: dict
):
    if detector_type not in GUARDRAIL_CATEGORY_MAPPING:
        raise ValueError(f"Invalid detector type encountered: {detector_type}")
    try:
        detector_instance = DetectionFactory.get_detector(
            GUARDRAIL_CATEGORY_MAPPING[detector_type],
            detector_name,
            **detector_config_dict,
        )
    except Exception as e:
        raise ValueError(
            f"Something broke when creating the detector {detector_name}. You might have passed an invalid parameter. Exception:{e}"
        )
    return detector_instance


# Create a Guard object from a config dict
def create_guard(guard_name: str, guard_config_dict: dict) -> Guard:
    if "type" not in guard_config_dict:
        raise ValueError("No type specified")
    elif guard_config_dict["type"] not in GUARDRAIL_CATEGORY_MAPPING:
        raise ValueError(f"{guard_config_dict["type"]} is not a valid guard type")

    if EARLY_EXIT in guard_config_dict:
        guard_fail_policy = guard_config_dict[EARLY_EXIT]
    else:
        guard_fail_policy = True

    if PARALLELIZED in guard_config_dict:
        guard_parallel_policy = guard_config_dict[PARALLELIZED]
    else:
        guard_parallel_policy = False

    if "methods" not in guard_config_dict:
        raise ValueError("No methods specified")

    detector_list = []
    for detector_name in guard_config_dict["methods"]:
        if detector_name in guard_config_dict:
            detector_config_dict = guard_config_dict[detector_name]
        else:
            detector_config_dict = {}
        detector_list.append(
            create_detector_for_guard(
                detector_name, guard_config_dict["type"], detector_config_dict
            )
        )

    return Guard(guard_name, detector_list, guard_fail_policy, guard_parallel_policy)


# Create a guardrail object from a config dict
def create_guardrail(guardrail_location: str, config_dict: dict) -> Guardrail:
    if guardrail_location not in ["input", "output"]:
        raise ValueError(f"Invalid location for guardrail: {guardrail_location}")

    guard_objects = []
    fail_policy = True
    parallel_policy = False

    guard_level = f"{guardrail_location}-guards"
    fail_fast = f"{guardrail_location}-early-exit"
    run_parallel = f"{guardrail_location}-run-parallel"

    if guard_level in config_dict:  # maybe worth raising a warning if missing?
        guard_list = config_dict[guard_level]
        for guard in guard_list:
            if isinstance(guard, dict):
                if not len(guard.keys()):
                    raise ValueError("Empty guard provided")
                if len(guard.keys()) > 1:
                    raise ValueError("Formatting for the guard appears to be incorrect")
                else:
                    guard_name = list(guard.keys())[0]
                    guard_objects.append(create_guard(guard_name, guard[guard_name]))
            elif isinstance(guard, str):
                if guard not in config_dict:
                    raise ValueError(
                        f"{guardrail_location} guardrail error: Guard {guard} was not specified in the config"
                    )
                guard_objects.append(create_guard(guard, config_dict[guard]))
            else:
                raise ValueError(
                    f"Invalid type for guardgroup {type(guard)}. Must be dict or str."
                )

    if fail_fast in config_dict:
        fail_policy = config_dict[fail_fast]

    if run_parallel in config_dict:
        parallel_policy = config_dict[run_parallel]

    guardrail = Guardrail(
        guardrail_location, guard_objects, fail_policy, parallel_policy
    )

    return guardrail


# Helper function - extract a value if present in the toml dict, else return the default
def extract_field_from_toml(field_level, field_name, default_value, toml_dict):
    field_value = default_value
    full_field_name = f"{field_level}-{field_name}"
    if full_field_name in toml_dict["guardrail"]:
        field_value = toml_dict["guardrail"][full_field_name]
    return field_value


# Convert a toml config file into a guardrail dict
def convert_toml_to_guardrail_dict(path_to_toml: str):
    with open(path_to_toml, "r") as file:
        toml_config_dict = toml.load(file)

    if "guardrail" not in toml_config_dict:
        raise ValueError("No [guardrail] in config file!")

    raw_config_dict = dict()  # type: Dict[str, Any]
    raw_config_dict["input-guards"] = extract_field_from_toml(
        "input", "guards", [], toml_config_dict
    )
    raw_config_dict["input-early-exit"] = extract_field_from_toml(
        "input", "early-exit", True, toml_config_dict
    )
    raw_config_dict["input-run-parallel"] = extract_field_from_toml(
        "input", "run-parallel", False, toml_config_dict
    )
    raw_config_dict["output-guards"] = extract_field_from_toml(
        "output", "guards", [], toml_config_dict
    )
    raw_config_dict["output-early-exit"] = extract_field_from_toml(
        "output", "early-exit", True, toml_config_dict
    )
    raw_config_dict["output-run-parallel"] = extract_field_from_toml(
        "output", "run-parallel", False, toml_config_dict
    )

    for groupname in raw_config_dict["input-guards"]:
        raw_config_dict[groupname] = toml_config_dict[groupname]
    for groupname in raw_config_dict["output-guards"]:
        raw_config_dict[groupname] = toml_config_dict[groupname]

    return raw_config_dict


# Convert a dictionary into the corresponding guardrails
def convert_dict_to_guardrails(config_dict: dict) -> Tuple[Guardrail, Guardrail]:
    input_guardrail = create_guardrail("input", config_dict)
    output_guardrail = create_guardrail("output", config_dict)
    return input_guardrail, output_guardrail


# convert a toml file into its corresponding guardrails
def convert_toml_to_guardrails(path_to_toml: str) -> Tuple[Guardrail, Guardrail]:
    config_dict = convert_toml_to_guardrail_dict(path_to_toml)
    return convert_dict_to_guardrails(config_dict)
