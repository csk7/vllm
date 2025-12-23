# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from typing import Any

import numpy as np

from vllm import LLM, SamplingParams


def extract_metrics_from_output(output, start_time, end_time) -> dict[str, Any]:
    """Extract benchmarking metrics from a single output."""
    # Get token counts
    num_output_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
    num_prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0

    # Total time (wall clock)
    total_time = end_time - start_time

    # Get metrics from output
    metrics = output.metrics
    if metrics is None:
        raise ValueError(
            "Metrics are None! Make sure to set \
                disable_log_stats=False when creating LLM."
        )

    # RequestStateStats from v1 engine
    ttft = metrics.first_token_latency
    prefill_time = metrics.first_token_ts - metrics.scheduled_ts
    decode_time = metrics.last_token_ts - metrics.first_token_ts
    tokens_per_sec = (
        metrics.num_generation_tokens / decode_time if decode_time > 0 else 0
    )

    return {
        "num_prompt_tokens": num_prompt_tokens,
        "num_output_tokens": num_output_tokens,
        "total_time": total_time,
        "ttft": ttft,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "tokens_per_sec": tokens_per_sec,
    }


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
        {metrics_dict['total_time']:.4f} seconds"
    )
    print("=" * 70)


def print_aggregated_metrics(all_metrics: list[dict[str, Any]]):
    """Print aggregated statistics (mean, median) across all requests."""
    if not all_metrics:
        return

    print("\n" + "=" * 70)
    print("AGGREGATED BENCHMARKING METRICS")
    print("=" * 70)
    print(f"Number of Requests:   {len(all_metrics)}")

    # Extract arrays for each metric
    ttfts = [m["ttft"] for m in all_metrics]
    prefill_times = [m["prefill_time"] for m in all_metrics]
    decode_times = [m["decode_time"] for m in all_metrics]
    tokens_per_sec_list = [m["tokens_per_sec"] for m in all_metrics]
    total_times = [m["total_time"] for m in all_metrics]

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
    print(f"  Mean:                {np.mean(total_times):.4f} seconds")
    print(f"  Median:              {np.median(total_times):.4f} seconds")
    print(f"  Std Dev:             {np.std(total_times):.4f} seconds")

    print("=" * 70)


def main():
    # ============================================================================
    # CORRECT MODEL COMBINATIONS FOR EAGLE3 SPECULATIVE DECODING
    # ============================================================================
    # Eagle3 requires specialized Eagle3 draft models, not just any smaller model.
    # Below are verified combinations that work with vLLM, ordered by size:
    #
    # Option 1: Llama-3.2-1B (Target) + Llama-3.2-1B Eagle3 (Draft) - SMALLEST!
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    target_model = "meta-llama/Llama-3.2-1B-Instruct"
    draft_model = "nm-testing/Llama3_2_1B_speculator.eagle3"
    trust_remote_code = False  # Llama models don't need trust_remote_code
    #
    # Option 2: Qwen3-8B (Target) + Qwen3-8B Eagle3 (Draft)
    # target_model = "Qwen/Qwen3-8B"
    # draft_model = "AngelSlim/Qwen3-8B_eagle3"
    #
    # Option 3: Qwen3-14B (Target) + Qwen3-14B Eagle3 (Draft)
    # target_model = "Qwen/Qwen3-14B"
    # draft_model = "AngelSlim/Qwen3-14B_eagle3"
    #
    # Option 4: Qwen3-32B (Target) + Qwen3-32B Eagle3 (Draft)
    # target_model = "Qwen/Qwen3-32B"
    # draft_model = "AngelSlim/Qwen3-32B_eagle3"
    # trust_remote_code = True
    #
    # Option 6: Llama-3.1-8B (Target) + Llama-3.1-8B Eagle3 (Draft)
    # target_model = "meta-llama/Llama-3.1-8B-Instruct"
    # draft_model = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    # trust_remote_code = False
    # ============================================================================

    # Configuration for benchmarking (set before creating LLM)
    clear_cache_between_requests = (
        True  # Clear KV cache between each request for fair benchmarking
    )
    enable_prefix_caching = (
        False  # Disable prefix caching for fair benchmarking (no cache reuse)
    )

    # Create the LLM engine with Eagle3 speculative decoding
    # NOTE: disable_log_stats=False is required to enable metrics collection
    # NOTE: enable_prefix_caching=False ensures no KV cache reuse between requests
    llm = LLM(
        model=target_model,
        max_model_len=512,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        trust_remote_code=trust_remote_code,  # Set based on model type
        enforce_eager=True,
        disable_log_stats=False,  # Enable metrics collection for benchmarking
        enable_prefix_caching=enable_prefix_caching,  # Disable for fair benchmarking
        speculative_config={
            # Eagle3 draft model (specialized, not a regular model)
            "model": draft_model,
            "dtype": "bfloat16",
            "method": "eagle3",  # Use Eagle3 speculative decoding method
            "draft_tensor_parallel_size": 1,  # Draft model must use TP=1
            "num_speculative_tokens": 2,  # Number of speculative tokens
        },
    )

    # Sampling configuration
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=400,
    )

    # Configuration for benchmarking
    num_warmup_requests = 3  # Number of warmup requests to discard
    num_benchmark_requests = 10  # Number of requests to benchmark

    # Create multiple prompts for benchmarking
    prompts = [
        "Winter is coming.",
        "The capital of France is",
        "Explain quantum computing in simple terms:",
        "Write a short story about a robot learning to paint.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
        "How does machine learning differ from traditional programming?",
        "What is the significance of the Renaissance period?",
        "Explain the theory of relativity.",
        "Describe the water cycle.",
    ]

    # Ensure we have enough prompts
    if len(prompts) < num_benchmark_requests:
        # Repeat prompts if needed
        prompts = (prompts * ((num_benchmark_requests // len(prompts)) + 1))[
            :num_benchmark_requests
        ]

    print("=" * 70)
    print("BENCHMARKING CONFIGURATION")
    print("=" * 70)
    print(f"Warmup Requests:              {num_warmup_requests}")
    print(f"Benchmark Requests:           {num_benchmark_requests}")
    print(
        f"Clear Cache Between Requests: \
        {'Yes' if clear_cache_between_requests else 'No'}"
    )
    print("=" * 70)

    # ===== WARMUP PHASE =====
    print("\n[WARMUP PHASE]")
    print("Running warmup requests (these will be discarded)...")
    warmup_prompts = prompts[:num_warmup_requests]
    for i, prompt in enumerate(warmup_prompts, 1):
        print(f"  Warmup {i}/{num_warmup_requests}...", end=" ", flush=True)
        _ = llm.generate([prompt], sampling_params)
        print("Done")

    print("\n[WARMUP COMPLETE]")
    time.sleep(0.5)  # Brief pause after warmup

    # ===== BENCHMARK PHASE =====
    print("\n[BENCHMARK PHASE]")
    print("Running benchmark requests...")
    all_metrics = []
    benchmark_prompts = prompts[
        num_warmup_requests : num_warmup_requests + num_benchmark_requests
    ]

    for i, prompt in enumerate(benchmark_prompts, 1):
        print(f"  Request {i}/{num_benchmark_requests}...", end=" ", flush=True)

        # Clear cache before each request if enabled (for fair benchmarking)
        if clear_cache_between_requests and i > 1:  # Don't clear before first request
            llm.reset_prefix_cache(reset_running_requests=False)

        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()

        for output in outputs:
            metrics_dict = extract_metrics_from_output(output, start_time, end_time)
            all_metrics.append(metrics_dict)

        print(
            f"Done (TTFT: {metrics_dict['ttft'] * 1000:.1f}ms, "
            f"Tokens/sec: {metrics_dict['tokens_per_sec']:.1f})"
        )

    # ===== RESULTS =====
    print("\n[BENCHMARK COMPLETE]")

    # Print aggregated statistics
    print_aggregated_metrics(all_metrics)

    # Optionally print individual request details (uncomment if needed)
    # print("\n[INDIVIDUAL REQUEST DETAILS]")
    # for i, (output, metrics_dict) in enumerate(zip(outputs, all_metrics), 1):
    #     print(f"\n--- Request {i} ---")
    #     print(f"Prompt: {output.prompt[:50]}...")
    #     print_single_request_metrics(metrics_dict)

    time.sleep(1)
    del llm


if __name__ == "__main__":
    main()
