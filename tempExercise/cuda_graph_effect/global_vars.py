# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path

DEBUG = True
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
if not DEBUG:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
else:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
PROFILE = True
# Set RPC timeout for profiler (30 minutes) - needed for stop_profile to flush traces
if PROFILE:
    os.environ["VLLM_RPC_TIMEOUT"] = "1800000"  # 30 minutes in milliseconds
# Server Params
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
RUN_LOC = "local"
if RUN_LOC == "local":
    MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    DRAFT_MODEL = "nm-testing/Llama3_2_1B_speculator.eagle3"
elif RUN_LOC == "server":
    MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # 30B MoE model
    # MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"  # Dense Model
    DRAFT_MODEL = "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"

# Client Params
LOG_DIR = Path(os.path.join(os.getcwd(), "log", "spec_script"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_INPUT_LEN = 100
RANDOM_OUTPUT_LEN = 128
REQUEST_RATE = 2.0
WARMUP_REQUESTS = 5
NUM_PROMPTS = 50
