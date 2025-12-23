# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any

import numpy as np

from vllm import LLM, SamplingParams


def get_model(model_type: str) -> LLM:
    target_model = "meta-llama/Llama-3.2-1B-Instruct"
    draft_model = "nm-testing/Llama3_2_1B_speculator.eagle3"
    if model_type == "original":
        return LLM(
            model=target_model,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_seqs=32,
            gpu_memory_utilization=0.95,
        )
    elif model_type == "SpecDecode":
        return LLM(
            model=target_model,
            max_model_len=8192,
            max_num_seqs=32,
            gpu_memory_utilization=0.95,
            disable_log_stats=False,  # To get metrics
            enable_prefix_caching=False,  # Clean Benchmarking
            speculative_config={
                "model": draft_model,
                "dtype": "bfloat16",
                "method": "eagle3",
                "draft_tensor_parallel_size": 1,  # Draft model must use TP=1
                "num_speculative_tokens": 5,  # Number of speculative tokens
            },
        )


def warmup(llm: LLM, prompts: list[str], sampling_params, warmup_iters):
    print("WARM UP START")
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


def benchmark_multi(benchmark_metrics: list[dict[str, Any]]):
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

    # Calculate statistics
    print("\n--- Time To First Token (TTFT) ---")
    print(
        f"Mean:{np.mean(ttfts):.4f} seconds \
                {np.mean(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Median:{np.median(ttfts):.4f} seconds \
                {np.median(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Std Dev:{np.std(ttfts):.4f} seconds \
                {np.std(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Min:{np.min(ttfts):.4f} seconds \
                {np.min(ttfts) * 1000:.2f} ms"
    )
    print(
        f"Max:{np.max(ttfts):.4f} seconds \
                {np.max(ttfts) * 1000:.2f} ms"
    )

    print("\n--- Prefill Time ---")
    print(
        f"Mean:{np.mean(prefill_times):.4f} seconds \
                {np.mean(prefill_times) * 1000:.2f} ms"
    )
    print(
        f"Median:{np.median(prefill_times):.4f} seconds \
                {np.median(prefill_times) * 1000:.2f} ms"
    )
    print(
        f"Std Dev:{np.std(prefill_times):.4f} seconds \
                {np.std(prefill_times) * 1000:.2f} ms"
    )

    print("\n--- Decode Time ---")
    print(
        f"Mean:{np.mean(decode_times):.4f} seconds \
                {np.mean(decode_times) * 1000:.2f} ms"
    )
    print(
        f"Median:{np.median(decode_times):.4f} seconds \
                {np.median(decode_times) * 1000:.2f} ms"
    )
    print(
        f"Std Dev:{np.std(decode_times):.4f} seconds \
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

    print("=" * 70)


def main():
    llm = get_model("SpecDecode")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=500,
    )

    prompts = [
        "Winter is here. How to enjoy it?",
        "How to become LLM inference performance optimization tzar?",
        "The capital of France is",
        "Explain quantum computing in simple terms:",
        "Write a short story about a robot learning to paint.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How does machine learning differ from traditional programming?",
        "Describe the water cycle.",
    ]

    all_metrics = []

    for i, prompt in enumerate(prompts):
        # print(f'\n\nPrompt : {output.prompt}')
        # print(f'Completion : {output.outputs[0].text}')
        if i > 0:
            llm.reset_prefix_cache(reset_running_requests=False)
        start_time = time.time()
        outputs = llm.generate(prompt, sampling_params)
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

    benchmark_multi(all_metrics)

    time.sleep(1)
    del llm


if __name__ == "__main__":
    main()
