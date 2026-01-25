# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import threading
import time

import requests
from global_vars import *


class vllmServer:
    def __init__(self):
        self.process = None

    def wait_for_server(self, timeout: int = 120) -> bool:
        """Wait for the server to be ready and return the actual model name."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try /v1/models endpoint (vLLM's standard endpoint)
                response = requests.get(f"{BASE_URL}/v1/models", timeout=2)
                if response.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def start_server(
        self,
        use_spec_decode: bool,
        disable_cuda_graphs: bool,
        seq_low_high: int,
        port: int = PORT,
    ):
        """Start vLLM server with specified configuration.

        Returns:
            sub process: process - The server process
        """
        cmd = [
            "vllm",
            "serve",
            MODEL,
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            str(seq_low_high),
            "--gpu-memory-utilization",
            "0.95",
        ]

        if disable_cuda_graphs:
            cmd.extend(["-cc.cudagraph_mode=NONE"])
        else:
            cmd.extend(["-cc.cudagraph_mode=FULL_AND_PIECEWISE"])

        if use_spec_decode:
            speculative_config = json.dumps(
                {
                    "model": DRAFT_MODEL,
                    "method": "eagle3",
                    "num_speculative_tokens": 3,
                    "draft_tensor_parallel_size": 1,
                }
            )
            cmd.extend(
                [
                    "--speculative-config",
                    speculative_config,
                ]
            )
        if PROFILE:
            profiler_dir = os.path.abspath("./log/vllm_profile/serve_1_test/")
            os.makedirs(profiler_dir, exist_ok=True)
            profiler_config = json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": profiler_dir,
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_memory": False,
                    "torch_profiler_dump_cuda_time_total": True,
                }
            )
            cmd.extend(
                [
                    "--profiler-config",
                    profiler_config,
                ]
            )

        print(f"\n{'=' * 80}")
        print(
            f"Starting Server with Config : {'Spec' if use_spec_decode else 'No Spec'},\
            {'CUDAGraph' if not disable_cuda_graphs else 'No CUDAGraph'}, SeqLowHigh \
            {seq_low_high}"
        )
        print(f"Command: {' '.join(cmd)}")
        print(f"{'=' * 80}\n")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Start a thread to read and print server output in real-time
        def read_output():
            """Read server output and print it."""
            if self.process and self.process.stdout:
                try:
                    for line in iter(self.process.stdout.readline, ""):
                        if line:
                            print(f"[vLLM Server] {line.rstrip()}")
                except Exception:
                    pass

        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()

        # Wait for server to be ready and get actual model name
        print(f"Waiting for server to be ready... : {cmd}")
        is_ready = self.wait_for_server()
        if is_ready:
            print("✓ Server is ready!")
            return self.process
        else:
            print("✗ Server failed to start within timeout")
            # Check process status
            return_code = self.process.poll()
            if return_code is not None:
                print(f"\nServer process terminated with return code: {return_code}")
                # Try to read any remaining output
                try:
                    remaining_output, _ = self.process.communicate(timeout=2)
                    if remaining_output:
                        print("\n--- Remaining Server Output ---")
                        print(remaining_output)
                        print("--- End Server Output ---\n")
                except subprocess.TimeoutExpired:
                    pass
            else:
                print(
                    "\nServer process is still running \
                        but not responding to HTTP requests."
                )
                print("This might indicate:")
                print("  - Server is still initializing (model loading, etc.)")
                print("  - Port conflict or network issue")
                print("  - Server error that prevents HTTP endpoint from starting")

            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            raise RuntimeError("Server failed to start")

    def stop_server(self):
        """Stop the vLLM server."""
        print("\nStopping server...")
        try:
            self.process.terminate()
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        print("✓ Server stopped")
