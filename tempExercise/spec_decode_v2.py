# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from typing import Any

import numpy as np
import torch

from vllm import LLM, SamplingParams

MODE = "profiling"  # "benchmarking"
RUN_LOC = "local"  # "server"
DEBUG = False

os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
if not DEBUG:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
else:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def get_model(model_type: str, enable_profiler: bool = False) -> LLM:
    if RUN_LOC == "local":
        target_model = "meta-llama/Llama-3.2-1B-Instruct"
        draft_model = "nm-testing/Llama3_2_1B_speculator.eagle3"
    elif RUN_LOC == "server":
        target_model = "Qwen/Qwen3-8B"
        draft_model = "RedHatAI/Qwen3-8B-speculator.eagle3"

    profiler_config = None
    if enable_profiler:
        profiler_dir = os.path.abspath("./log/vllm_profile")
        os.makedirs(profiler_dir, exist_ok=True)
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": profiler_dir,
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": False,
            "torch_profiler_dump_cuda_time_total": True,
        }

    if model_type == "original":
        return LLM(
            model=target_model,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=32,
            disable_log_stats=False,  # To get metrics
            enable_prefix_caching=True,  # Clean Benchmarking
            gpu_memory_utilization=0.95,
            profiler_config=profiler_config,
        )
    elif model_type == "SpecDecode":
        return LLM(
            model=target_model,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=32,
            gpu_memory_utilization=0.95,
            disable_log_stats=False,  # To get metrics
            enable_prefix_caching=False,  # Clean Benchmarking
            enforce_eager=True,
            speculative_config={
                "model": draft_model,
                "dtype": "bfloat16",
                "method": "eagle3",
                "draft_tensor_parallel_size": 1,  # Draft model must use TP=1
                "num_speculative_tokens": 4,  # Number of speculative tokens
            },
            profiler_config=profiler_config,
        )


def warmup(llm: LLM, prompts: list[str], sampling_params, warmup_iters):
    print("WARM UP START")
    assert warmup_iters <= len(prompts)
    for i in range(warmup_iters):
        _ = llm.generate(prompts[i], sampling_params)
    print("WARM UP DONE")


def print_single_request_metrics(metrics_dict: dict[str, Any]):
    """Print benchmarking metrics for a single request."""
    print("\n" + "=" * 70)
    print("BENCHMARKING METRICS (Single Request)")
    print("=" * 70)
    print(f"Prompt Tokens:  {metrics_dict['num_prompt_tokens']}")
    print(f"Output Tokens:  {metrics_dict['num_output_tokens']}")
    print(
        f"Total Tokens: \
        {metrics_dict['num_prompt_tokens'] + metrics_dict['num_output_tokens']}"
    )
    print(
        f"TTFT:{metrics_dict['ttft']:.4f} seconds \
                {metrics_dict['ttft'] * 1000:.2f} ms"
    )
    print(
        f"Prefill Time:{metrics_dict['prefill_time']:.4f} seconds \
                {metrics_dict['prefill_time'] * 1000:.2f} ms"
    )
    print(
        f"Decode Time:{metrics_dict['decode_time']:.4f} seconds \
                {metrics_dict['decode_time'] * 1000:.2f} ms"
    )
    print(
        f"Tokens/sec:    \
        {metrics_dict['tokens_per_sec']:.2f} tokens/second"
    )
    print(
        f"\nTotal Time:  \
        {metrics_dict['wall_clock_time']:.4f} seconds"
    )
    if metrics_dict["Efficiency"] is not None:
        print(
            f"\nSpec Decode Efficiency:  \
        {metrics_dict['Efficiency']:.4f}"
        )
        print(
            f"\nSpec Decode Draft tokens:  \
        {metrics_dict['Draft tokens']:.4f}"
        )
        print(
            f"\nSpec Decode Accepted tokens:  \
        {metrics_dict['Accepted tokens']:.4f}"
        )

    print("=" * 70)


def benchmark_single(
    metrics, num_output_tokens, num_prompt_tokens, start_time, end_time
) -> dict[str, Any]:
    ret_val = {}

    prefill_time = metrics.first_token_ts - metrics.scheduled_ts
    ttft = metrics.first_token_latency
    decode_time = metrics.last_token_ts - metrics.first_token_ts
    tokens_per_sec = num_output_tokens / decode_time
    wall_clock_time = end_time - start_time

    ret_val["num_prompt_tokens"] = num_prompt_tokens
    ret_val["num_output_tokens"] = num_output_tokens
    ret_val["ttft"] = ttft
    ret_val["prefill_time"] = prefill_time
    ret_val["decode_time"] = decode_time
    ret_val["tokens_per_sec"] = tokens_per_sec
    ret_val["wall_clock_time"] = wall_clock_time

    # print_single_request_metrics(ret_val)

    return ret_val


def benchmark_multi(
    benchmark_metrics: list[dict[str, Any]], scheduler_stats: list | None = None
):
    print("\n" + "=" * 70)
    print("AGGREGATED BENCHMARKING METRICS")
    print("=" * 70)
    print(f"Number of Requests:   {len(benchmark_metrics)}")

    # Extract arrays for each metric
    ttfts = [m["ttft"] for m in benchmark_metrics]
    prefill_times = [m["prefill_time"] for m in benchmark_metrics]
    decode_times = [m["decode_time"] for m in benchmark_metrics]
    tokens_per_sec_list = [m["tokens_per_sec"] for m in benchmark_metrics]
    wall_clock_time_list = [m["wall_clock_time"] for m in benchmark_metrics]

    # Spec Decoding Stats from Prometheus metrics (aggregated across all requests)
    spec_decode_accepted_tokens = None
    spec_decode_draft_tokens = None
    spec_decode_num_drafts = None

    if scheduler_stats:
        for scheduler_stat in scheduler_stats:
            if scheduler_stat.name == "vllm:spec_decode_num_accepted_tokens":
                spec_decode_accepted_tokens = scheduler_stat.value
            elif scheduler_stat.name == "vllm:spec_decode_num_draft_tokens":
                spec_decode_draft_tokens = scheduler_stat.value
            elif scheduler_stat.name == "vllm:spec_decode_num_drafts":
                spec_decode_num_drafts = scheduler_stat.value

    # Calculate efficiency as acceptance rate: accepted / draft tokens
    spec_decode_efficiency = (
        spec_decode_accepted_tokens / spec_decode_draft_tokens
        if spec_decode_draft_tokens and spec_decode_draft_tokens > 0
        else None
    )

    # Calculate statistics
    print("\n--- Time To First Token (TTFT) ---")
    print(
        f"Mean:    \
                {np.mean(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Median:  \
                {np.median(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Std Dev: \
                {np.std(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Min:     \
                {np.min(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Max:     \
                {np.max(ttfts) * 1000:.2f} ms"
    )

    print("\n--- Prefill Time ---")
    print(
        f"Mean:    \
                {np.mean(prefill_times) * 1000:.2f} ms"
    )
    print(
        f"Median:  \
                {np.median(prefill_times) * 1000:.2f} ms"
    )
    print(
        f"Std Dev: \
                {np.std(prefill_times) * 1000:.2f} ms"
    )

    print("\n--- Decode Time ---")
    print(
        f"Mean:     \
                {np.mean(decode_times) * 1000:.2f} ms"
    )
    print(
        f"Median:   \
                {np.median(decode_times) * 1000:.2f} ms"
    )
    print(
        f"Std Dev:  \
                {np.std(decode_times) * 1000:.2f} ms"
    )

    print("\n--- Tokens/Second ---")
    print(f"  Mean:                {np.mean(tokens_per_sec_list):.2f} tokens/second")
    print(f"  Median:              {np.median(tokens_per_sec_list):.2f} tokens/second")
    print(f"  Std Dev:             {np.std(tokens_per_sec_list):.2f} tokens/second")
    print(f"  Min:                 {np.min(tokens_per_sec_list):.2f} tokens/second")
    print(f"  Max:                 {np.max(tokens_per_sec_list):.2f} tokens/second")

    print("\n--- Total Time (Wall Clock) ---")
    print(f"  Mean:                {np.mean(wall_clock_time_list):.4f} seconds")
    print(f"  Median:              {np.median(wall_clock_time_list):.4f} seconds")
    print(f"  Std Dev:             {np.std(wall_clock_time_list):.4f} seconds")

    # Print speculative decoding statistics if available
    if spec_decode_accepted_tokens is not None:
        print("\n--- Speculative Decoding Statistics ---")
        print(f"\nAccepted Tokens: {spec_decode_accepted_tokens}")
        print(f"\nDraft Tokens:    {spec_decode_draft_tokens}")
        print(f"\nEfficiency:      {spec_decode_efficiency}")
        print(f"\nNum Drafts:      {spec_decode_num_drafts}")

        # Calculate acceptance length per draft step
        if spec_decode_num_drafts and spec_decode_num_drafts > 0:
            acceptance_length = spec_decode_accepted_tokens / spec_decode_num_drafts
            print(
                f"\nAcceptance Length (avg tokens per draft step): \
                {acceptance_length:.3f}"
            )
            print("Expected for Eagle3: ~2.811 (with temp=0.0, top_p=1.0)")
            if acceptance_length < 2.0:
                print("⚠️  WARNING: Acceptance length is lower than expected!")
                print(
                    f"Current: {acceptance_length:.3f} \
                    tokens/step (expected ~2.811)"
                )
                print(
                    f"Efficiency: {spec_decode_efficiency * 100:.1f}% \
                    (ideal: >90%)"
                )

    print("=" * 70)


def main():
    # Enable vLLM's built-in profiler which runs in worker processes
    llm = get_model("SpecDecode", enable_profiler=True)

    if MODE == "profiling":
        max_tokens = 10
    elif MODE == "benchmarking":
        max_tokens = 128
    else:
        raise NotImplementedError("Check MODE")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.8,
        max_tokens=max_tokens,
    )

    prompts = [
        "Explain the concept of machine learning in simple terms. \
            Cover the basic idea, \
            how it differs from traditional programming, \
            and give a practical example.",
        "Describe the process of photosynthesis. Include the key steps, \
            what inputs are needed, and what outputs are produced.",
        "Write a brief explanation of how neural networks work. \
            Discuss neurons, layers, and the learning process.",
        "Explain the difference between supervised and unsupervised learning. \
            Provide examples of each type.",
        "Describe the water cycle. Include all the major stages and \
            how water moves through the system.",
        "What are the main advantages and disadvantages of renewable \
            energy sources? Discuss at least three different types.",
        "Explain how a compiler works. Describe the main stages \
            from source code to executable program.",
        "What is the difference between CPU and GPU? Explain \
            their respective strengths and use cases.",
    ]

    all_metrics = []

    # Start profiling - this enables profiling in worker processes
    warmup(llm=llm, prompts=prompts, sampling_params=sampling_params, warmup_iters=2)

    for i, prompt in enumerate(prompts):
        if i > 0:
            llm.reset_prefix_cache(reset_running_requests=False)
        start_time = time.time()

        if i == 2:
            llm.start_profile()
        outputs = llm.generate(prompt, sampling_params)
        if i == 2:
            llm.stop_profile()
        end_time = time.time()

        for output in outputs:
            all_metrics.append(
                benchmark_single(
                    output.metrics,
                    len(output.outputs[0].token_ids),
                    len(output.prompt_token_ids),
                    start_time,
                    end_time,
                )
            )

    # Stop profiling

    benchmark_multi(all_metrics, llm.get_metrics())

    # Wait for profiler to finish writing traces
    print("\nWaiting for profiler to finish writing traces...")
    time.sleep(5)
    print("Profiler traces saved to ./log/vllm_profile/")

    time.sleep(1)
    del llm

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
