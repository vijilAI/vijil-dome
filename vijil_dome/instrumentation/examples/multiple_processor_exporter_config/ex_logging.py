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

import os
import toml
import logging
import time
from pathlib import Path
from vijil_dome.instrumentation.logging import (
    setup_vijil_logging,
    set_log_level,
)


def test_logging_config():
    # Load the configuration
    CONFIG_PATH = Path(__file__).parent / "config.toml"
    try:
        with open(CONFIG_PATH, "r") as f:
            config = toml.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {CONFIG_PATH}")
        return
    except toml.TomlDecodeError as e:
        print(f"Error parsing config file: {e}")
        return

    # Setup Vijil logging with the loaded configuration
    try:
        logger_provider = setup_vijil_logging(config)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return

    # Get a logger
    logger = logging.getLogger("test_logger")

    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test logging with extra data
    extra_data = {"user_id": 12345, "action": "login"}
    logger.info("User logged in", extra=extra_data)

    # Test batch processing by sending multiple logs quickly
    for i in range(5):
        logger.info(f"Batch log message {i}")

    # Allow time for batch processing
    time.sleep(2)

    # Test changing log levels
    set_log_level(logging.DEBUG)
    logger.debug("This debug message should now appear")

    # Clean up
    logger_provider.shutdown()

    # Clean up
    logger_provider.shutdown()

    # Read from the log file
    log_file_path = "/tmp/debug_logs.log"
    try:
        with open(log_file_path, "r") as f:
            log_contents = f.read()
        print("\nContents of the log file:")
        print(log_contents)
    except FileNotFoundError:
        print(f"Log file not found at {log_file_path}")
    except Exception as e:
        print(f"Error reading log file: {e}")
    finally:
        if os.path.exists(log_file_path):
            os.remove(log_file_path)


if __name__ == "__main__":
    test_logging_config()
