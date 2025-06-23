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

from vijil_dome.detectors import DetectionMethod, DetectionTimingResult
import asyncio
from asyncio import Task  # noqa: F401
import concurrent.futures
from typing import List, Dict, Tuple, Union, Any  # noqa: F401
from abc import ABC, abstractmethod
import time
from pydantic import BaseModel
import logging


DEFAULT_GUARDRAIL_BLOCKED_PATTERNS = [
    "Blocked by input guardrail",
    "Blocked by output guardrail",
]

logger = logging.getLogger("vijil.dome")


class GuardResult(BaseModel):
    triggered: bool
    details: Dict[str, DetectionTimingResult]
    exec_time: float
    response: str

    def __str__(self):
        result_dict = {
            "triggered": self.triggered,
            "details": self.details,
            "exec_time": self.exec_time,
            "response": self.response,
        }
        return str(result_dict)

    def __repr__(self):
        return self.__str__()


class GuardrailResult(BaseModel):
    flagged: bool
    guardrail_response_message: str
    exec_time: float
    guard_exec_details: Dict[str, GuardResult]

    def __str__(self):
        result_dict = {
            "flagged": self.flagged,
            "guardrail_response_message": self.guardrail_response_message,
            "exec_time": self.exec_time,
            "guard_exec_details": self.guard_exec_details,
        }
        return str(result_dict)

    def __repr__(self):
        return self.__str__()


# Enable detectors that have blocking sync_detect methods to run on another thread
async def run_detector_via_executor(executor, detector, query_string):
    start_time = time.time()
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(executor, detector.sync_detect, query_string)
    try:
        result = await future
        execution_time = round((time.time() - start_time) * 1000, 3)
        result_with_timing = DetectionTimingResult(
            hit=result[0], result=result[1], exec_time=execution_time
        )
        return result_with_timing
    except asyncio.CancelledError:
        future.cancel()
        raise


class Guard:
    def __init__(
        self,
        guard_name: str,
        detector_list: List[DetectionMethod],
        early_exit: bool = True,
        run_in_parallel: bool = False,
    ):
        self.guard_name = guard_name
        self.detector_list = detector_list
        self.early_exit = early_exit
        self.run_in_parallel = run_in_parallel
        self.blocked_response_string = f"Guard:{self.guard_name} "

    # Sequential guard
    async def sequential_guard(self, query_string: str) -> GuardResult:
        st_time = time.time()
        flagged = False
        detector_results = {}
        response_string = query_string
        for detector in self.detector_list:
            detector_scan_result = await detector.detect_with_time(query_string)
            detector_results[type(detector).__name__] = detector_scan_result

            if detector_scan_result.hit:
                response_string = (
                    self.blocked_response_string
                    + detector_scan_result.result["response_string"]
                )
                flagged = True
                if self.early_exit:
                    break
            if not flagged:
                if detector_scan_result.result["response_string"] != query_string:
                    response_string = detector_scan_result.result[
                        "response_string"
                    ]  # For guards with passthroughs
        exec_time = time.time() - st_time
        return GuardResult(
            triggered=flagged,
            details=detector_results,
            exec_time=exec_time,
            response=response_string,
        )

    # Parallel Guard
    async def parallel_guard(self, query_string: str, executor) -> GuardResult:
        st_time = time.time()
        detector_results = {}
        flagged = False
        response_string = query_string

        # Helper function - enable parallel execution of guards which can stop once a single guard is triggered
        async def detector_wrapper(
            detector: DetectionMethod, query_string: str, executor
        ):
            asyncio_timeout_limit = 5  # Timeout limit for any detector - 5 seconds
            if hasattr(detector, "run_in_executor") and detector.run_in_executor:
                if hasattr(detector, "sync_detect") and callable(detector.sync_detect):
                    try:
                        async with asyncio.timeout(asyncio_timeout_limit):
                            result = await run_detector_via_executor(
                                executor, detector, query_string
                            )
                    except TimeoutError:
                        result = DetectionTimingResult(
                            hit=False,
                            result={"error": "Detection method timed out"},
                            exec_time=asyncio_timeout_limit,
                        )
                else:
                    raise ValueError(
                        f"Attempting to use a detector of type {type(detector).__name__} in an executor without defining a sync detection method."
                    )
            else:
                try:
                    async with asyncio.timeout(asyncio_timeout_limit):
                        result = await detector.detect_with_time(query_string)
                except TimeoutError:
                    logger.warn(
                        f"Detection method {detector.__class__.__name__} timed out."
                    )
                    result = DetectionTimingResult(
                        hit=False,
                        result={"error": "Detection method timed out"},
                        exec_time=asyncio_timeout_limit,
                    )
            detector_results[type(detector).__name__] = result
            return result

        tasks = []  # type: Union[set[Task[Any]], list[Task[Any]]]
        for detector in self.detector_list:
            task = asyncio.create_task(
                detector_wrapper(detector, query_string, executor)
            )
            task.set_name(f"{self.guard_name}{detector.__class__.__name__}_detect")
            tasks.append(task)

        while tasks:
            if self.early_exit:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.ALL_COMPLETED
                )

            for task in done:
                if not task.cancelled() and not task.exception():
                    if task.result().hit:
                        response_string = (
                            self.blocked_response_string
                            + task.result().hitData["response_string"]
                        )
                        flagged = True
                        for p in pending:
                            p.cancel()
                        break
                    if not flagged:
                        if "response_string" not in task.result().hitData:
                            logger.warn(
                                f"No response string was found from result in task {task.get_name()}"
                            )
                        else:
                            if task.result().hitData["response_string"] != query_string:
                                response_string = task.result().hitData[
                                    "response_string"
                                ]
            tasks = pending
        exec_time = time.time() - st_time
        return GuardResult(
            triggered=flagged,
            details=detector_results,
            exec_time=exec_time,
            response=response_string,
        )

    # Sync method that runs the regular guard or parallel guard based on config
    # This is never invoked by the guardrail class, but is useful for debugging
    def scan(self, query_string: str, executor) -> GuardResult:
        if self.run_in_parallel:
            return asyncio.run(self.parallel_guard(query_string, executor))
        else:
            return asyncio.run(self.sequential_guard(query_string))

    # Async scan
    async def async_scan(self, query_string: str, executor) -> GuardResult:
        if self.run_in_parallel:
            return await self.parallel_guard(query_string, executor)
        else:
            return await self.sequential_guard(query_string)


class Guardrail:
    def __init__(
        self,
        level: str,
        guard_list: List[Guard],
        early_exit: bool = True,
        run_in_parallel: bool = False,
    ):
        if level not in ["input", "output"]:
            raise ValueError("Guardrail level must be 'input' or 'output'")
        self.level = level
        self.guard_list = guard_list
        self.early_exit = early_exit
        self.run_in_parallel = run_in_parallel
        self.blocked_response_string = f"Blocked by {self.level} guardrail at "
        self.executor = concurrent.futures.ThreadPoolExecutor()

    async def sequential_guard(self, query_string: str) -> GuardrailResult:
        st_time = time.time()
        flagged = False
        guard_results = {}
        response_string = query_string
        for guard in self.guard_list:
            guard_scan_result = await guard.async_scan(query_string, self.executor)
            guard_results[guard.guard_name] = guard_scan_result
            if guard_scan_result.triggered:
                response_string = (
                    self.blocked_response_string + guard_scan_result.response
                )
                flagged = True
                if self.early_exit:
                    break
            if not flagged:
                if guard_scan_result.response != query_string:
                    response_string = guard_scan_result.response
        exec_time = time.time() - st_time
        return GuardrailResult(
            flagged=flagged,
            guardrail_response_message=response_string,
            exec_time=exec_time,
            guard_exec_details=guard_results,
        )

    # Parallel guard
    async def parallel_guard(self, query_string: str) -> GuardrailResult:
        guard_results = {}
        flagged = False
        response_string = query_string
        st_time = time.time()

        # Helper function - enable parallel execution of guards which can stop once a single guard is triggered
        async def guard_wrapper(guard: Guard, query_string: str, executor):
            result = await guard.async_scan(query_string, executor)
            guard_results[guard.guard_name] = result
            return result

        tasks = [
            asyncio.create_task(guard_wrapper(guard, query_string, self.executor))
            for guard in self.guard_list
        ]  # type: Union[set[Task[Any]], list[Task[Any]]]
        while tasks:
            if self.early_exit:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
            else:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.ALL_COMPLETED
                )

            for task in done:
                if not task.cancelled() and not task.exception():
                    if task.result().triggered:
                        response_string = (
                            self.blocked_response_string + task.result().response
                        )
                        flagged = True
                        for p in pending:
                            p.cancel()
                        break
                    if not flagged:
                        if task.result().response != query_string:
                            response_string = task.result().response
            tasks = pending
        exec_time = time.time() - st_time
        return GuardrailResult(
            flagged=flagged,
            guardrail_response_message=response_string,
            exec_time=exec_time,
            guard_exec_details=guard_results,
        )

    # Sync method that runs the regular guard or parallel guard based on config
    def scan(self, query_string: str) -> GuardrailResult:
        if self.run_in_parallel:
            return asyncio.run(self.parallel_guard(query_string))
        else:
            return asyncio.run(self.sequential_guard(query_string))

    # Async scan
    async def async_scan(self, query_string: str) -> GuardrailResult:
        if self.run_in_parallel:
            logger.info(f'Scanning "{query_string}" through guards in parallel.')
            return await self.parallel_guard(query_string)
        else:
            logger.info(f'Scanning "{query_string}" through guards sequentially.')
            return await self.sequential_guard(query_string)


# A generic Guardrail Config Class
class GuardrailConfig(ABC):
    @abstractmethod
    def get_input_guardrail(self) -> Guardrail:
        pass

    @abstractmethod
    def get_output_guardrail(self) -> Guardrail:
        pass
