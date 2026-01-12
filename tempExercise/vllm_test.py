# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time

import torch

from vllm import LLM, SamplingParams

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_RPC_TIMEOUT"] = "6000"

target_model = "meta-llama/Llama-3.2-1B-Instruct"

profiler_dir = os.path.abspath("./log/vllm_profile2")
os.makedirs(profiler_dir, exist_ok=True)
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": profiler_dir,
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_memory": False,
    "torch_profiler_dump_cuda_time_total": True,
}

llm = LLM(
    model=target_model,
    dtype="bfloat16",
    max_model_len=1024,
    max_num_seqs=2,
    gpu_memory_utilization=0.95,
    disable_log_stats=False,  # To get metrics
    enable_prefix_caching=True,  # Clean Benchmarking
    enforce_eager=True,
    profiler_config=profiler_config,
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.8,
    max_tokens=10,
)

prompts = [
    "What is the difference between CPU and GPU? Explain \
        their respective strengths and use cases.",
]

llm.start_profile()
output = llm.generate(prompts=prompts, sampling_params=sampling_params)
llm.stop_profile()

print(output[0].outputs[0].text)

time.sleep(50)
del llm
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
