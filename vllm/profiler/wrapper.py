# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, suppress
from typing import Literal

import torch
from typing_extensions import override

from vllm.config import ProfilerConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class WorkerProfiler(ABC):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        self._delay_iters = profiler_config.delay_iterations
        if self._delay_iters > 0:
            logger.info_once(
                "GPU profiling will start "
                f"{self._delay_iters} steps after start_profile."
            )

        self._max_iters = profiler_config.max_iterations
        if self._max_iters > 0:
            logger.info_once(
                "GPU profiling will stop "
                f"after {self._max_iters} worker steps, "
                "or when stop_profile is received."
            )

        # Track when the profiler gets triggered by start_profile
        self._active_iteration_count = 0
        self._active = False

        # Track when the profiler is actually running
        self._profiling_for_iters = 0
        self._running = False

    @abstractmethod
    def _start(self) -> None:
        """Start the profiler."""
        pass

    @abstractmethod
    def _stop(self) -> None:
        """Stop the profiler."""
        pass

    def _call_start(self) -> None:
        """Call _start with error handling but no safeguards."""
        try:
            self._start()
            self._running = True  # Only mark as running if start succeeds
        except Exception as e:
            logger.warning("Failed to start profiler: %s", e)

    def _call_stop(self) -> None:
        """Call _stop with error handling but no safeguards."""
        try:
            self._stop()
            logger.info_once("Profiler stopped successfully.", scope="local")
        except Exception as e:
            logger.warning("Failed to stop profiler: %s", e)
        self._running = False  # Always mark as not running, assume stop worked

    def start(self) -> None:
        """Attempt to start the profiler, accounting for delayed starts."""
        if self._active:
            logger.debug(
                "start_profile received when profiler is already active. "
                "Ignoring request."
            )
            return
        self._active = True
        if self._delay_iters == 0:
            self._call_start()

    def step(self) -> None:
        """Update the profiler state at each worker step,
        to handle delayed starts and max iteration limits."""
        if not self._active:
            return

        self._active_iteration_count += 1

        if (
            not self._running
            and self._delay_iters > 0
            and self._active_iteration_count == self._delay_iters
        ):
            logger.info_once("Starting profiler after delay...", scope="local")
            self._call_start()

        if self._running:
            self._profiling_for_iters += 1

        if (
            self._max_iters > 0
            and self._running
            and self._profiling_for_iters > self._max_iters
        ):
            # Automatically stop the profiler after max iters
            # will be marked as not running, but leave as active so that stop
            # can clean up properly
            logger.info_once(
                "Max profiling iterations reached. Stopping profiler...", scope="local"
            )
            self._call_stop()
            return

    def stop(self) -> None:
        """Attempt to stop the profiler, accounting for overlapped calls."""
        if not self._active:
            logger.debug(
                "stop_profile received when profiler is not active. Ignoring request."
            )
            return
        self._active = False
        self._active_iteration_count = 0
        self._profiling_for_iters = 0

        if self._running:
            self._call_stop()

    def shutdown(self) -> None:
        """Ensure profiler is stopped when shutting down."""
        logger.info_once("Shutting down profiler", scope="local")
        if self._running:
            self.stop()

    def annotate_context_manager(self, name: str):
        """Return a context manager to annotate profiler traces."""
        return nullcontext()


TorchProfilerActivity = Literal["CPU", "CUDA", "XPU"]
TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
}


class TorchProfilerWrapper(WorkerProfiler):
    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity],
    ) -> None:
        super().__init__(profiler_config)

        self.local_rank = local_rank
        self.profiler_config = profiler_config
        self.activities = activities  # Store for use in _stop()
        torch_profiler_trace_dir = profiler_config.torch_profiler_dir
        if local_rank in (None, 0):
            logger.info_once(
                "Torch profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
                scope="local",
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                profiler_config.torch_profiler_record_shapes,
                profiler_config.torch_profiler_with_memory,
                profiler_config.torch_profiler_with_stack,
                profiler_config.torch_profiler_with_flops,
            )

        self.dump_cpu_time_total = "CPU" in activities and len(activities) == 1
        self.torch_profiler_trace_dir = torch_profiler_trace_dir
        self.worker_name = worker_name
        self.use_gzip = profiler_config.torch_profiler_use_gzip
        self.accumulated_traces = []

        def accumulate_trace_handler(prof):
            """Accumulate trace data from each step for merging."""
            import json
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                temp_file = f.name
            prof.export_chrome_trace(temp_file)
            try:
                with open(temp_file) as f:
                    self.accumulated_traces.append(json.load(f))
            except Exception as e:
                logger.debug("Could not read trace: %s", e)
            finally:
                with suppress(Exception):
                    os.remove(temp_file)

        # Schedule is required for GPU profiling to work
        # active=1 ensures GPU events are captured for each step
        self.profiler = torch.profiler.profile(
            activities=[TorchProfilerActivityMap[activity] for activity in activities],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
            record_shapes=profiler_config.torch_profiler_record_shapes,
            profile_memory=profiler_config.torch_profiler_with_memory,
            with_stack=profiler_config.torch_profiler_with_stack,
            with_flops=profiler_config.torch_profiler_with_flops,
            on_trace_ready=accumulate_trace_handler,
        )

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def step(self) -> None:
        """Advance the torch profiler schedule to enable GPU event recording."""
        super().step()
        if self._running:
            self.profiler.step()

    @override
    def _stop(self) -> None:
        if "CUDA" in self.activities:
            torch.cuda.synchronize()
        self.profiler.stop()

        # Merge all accumulated traces into a single file with continuous timestamps
        try:
            import gzip
            import json
            import os
            import shutil

            trace_file = os.path.join(
                self.torch_profiler_trace_dir, f"{self.worker_name}.pt.trace.json"
            )

            if self.accumulated_traces:
                merged_trace = {
                    "traceEvents": [],
                    "displayTimeUnit": "ms",
                    "otherData": {},
                }
                current_max_time = None

                for trace in self.accumulated_traces:
                    if "traceEvents" in trace and trace["traceEvents"]:
                        trace_min_time = min(
                            (e.get("ts") for e in trace["traceEvents"] if "ts" in e),
                            default=None,
                        )
                        trace_max_time = max(
                            (e.get("ts") for e in trace["traceEvents"] if "ts" in e),
                            default=None,
                        )

                        if trace_min_time is not None:
                            time_offset = (
                                0
                                if current_max_time is None
                                else current_max_time - trace_min_time
                            )
                            current_max_time = (
                                trace_max_time + time_offset
                                if trace_max_time
                                else current_max_time
                            )

                            for event in trace["traceEvents"]:
                                adjusted_event = event.copy()
                                if "ts" in adjusted_event:
                                    adjusted_event["ts"] += time_offset
                                merged_trace["traceEvents"].append(adjusted_event)
                        else:
                            merged_trace["traceEvents"].extend(trace["traceEvents"])

                    if "otherData" in trace:
                        merged_trace["otherData"].update(trace["otherData"])

                merged_trace["traceEvents"].sort(key=lambda x: x.get("ts", 0))

                with open(trace_file, "w") as f:
                    json.dump(merged_trace, f)

                if self.use_gzip:
                    with (
                        open(trace_file, "rb") as f_in,
                        gzip.open(trace_file + ".gz", "wb") as f_out,
                    ):
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(trace_file)
            else:
                self.profiler.export_chrome_trace(trace_file)
                if self.use_gzip:
                    with (
                        open(trace_file, "rb") as f_in,
                        gzip.open(trace_file + ".gz", "wb") as f_out,
                    ):
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(trace_file)
        except Exception as e:
            logger.warning("Could not export trace: %s", e)

        profiler_config = self.profiler_config
        rank = self.local_rank
        if profiler_config.torch_profiler_dump_cuda_time_total:
            profiler_dir = profiler_config.torch_profiler_dir
            profiler_out_file = f"{profiler_dir}/profiler_out_{rank}.txt"
            sort_key = "self_cuda_time_total"
            table = self.profiler.key_averages().table(sort_by=sort_key)

            with open(profiler_out_file, "w") as f:
                print(table, file=f)

            # only print profiler results on rank 0
            if rank == 0:
                print(table)
        if self.dump_cpu_time_total and rank == 0:
            logger.info(
                self.profiler.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=50
                )
            )

    @override
    def annotate_context_manager(self, name: str):
        @contextmanager
        def sync_context():
            with torch.profiler.record_function(name):
                yield
            if "CUDA" in self.activities:
                torch.cuda.synchronize()

        return sync_context()


class CudaProfilerWrapper(WorkerProfiler):
    def __init__(self, profiler_config: ProfilerConfig) -> None:
        super().__init__(profiler_config)
        # Note: lazy import to avoid dependency issues if CUDA is not available.
        import torch.cuda.profiler as cuda_profiler

        self._cuda_profiler = cuda_profiler

    @override
    def _start(self) -> None:
        self._cuda_profiler.start()

    @override
    def _stop(self) -> None:
        self._cuda_profiler.stop()

    @override
    def annotate_context_manager(self, name: str):
        return torch.cuda.nvtx.range(name)
