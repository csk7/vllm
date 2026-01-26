# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import time
from pathlib import Path

from global_vars import *


def run_benchmark(
    scenario_name: str,
    use_spec_decode: bool,
    client_max_concurrency: int,
    result_file: Path,
    num_prompts: int = None,
    num_warmups: int = None,
) -> dict:
    """Run benchmark and return results.

    Args:
        scenario_name: Name of the benchmark scenario
        use_spec_decode: Whether to use speculative decoding
        client_max_concurrency: Maximum client concurrency
        result_file: Path to save results
        num_prompts: Number of prompts to run (defaults to NUM_PROMPTS)
        num_warmups: Number of warmup requests (defaults to WARMUP_REQUESTS)
    """
    config_type = "spec_decode" if use_spec_decode else "vanilla"
    model_name = MODEL

    # Use provided values or default to global constants
    # When called normally (not from phased),
    # always uses NUM_PROMPTS and WARMUP_REQUESTS
    prompts_to_run = NUM_PROMPTS if num_prompts is None else num_prompts
    warmups_to_run = WARMUP_REQUESTS if num_warmups is None else num_warmups

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model_name,
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",  # Explicitly set endpoint
        "--dataset-name",
        "random",
        "--random-input-len",
        str(RANDOM_INPUT_LEN),
        "--random-output-len",
        str(RANDOM_OUTPUT_LEN),
        "--num-prompts",
        str(prompts_to_run),
        "--max-concurrency",
        str(client_max_concurrency),
        "--request-rate",
        str(REQUEST_RATE),
        "--num-warmups",
        str(warmups_to_run),
        "--save-result",
        "--result-dir",
        str(LOG_DIR),
        "--result-filename",
        result_file.name,
        "--percentile-metrics",
        "tpot",
        "--metric-percentiles",
        "90",
    ]

    print(f"\n{'=' * 80}")
    print(f"Running benchmark: {scenario_name} - {config_type}")
    print(f"Client max_concurrency: {client_max_concurrency}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Load results from JSON
        if result_file.exists():
            with open(result_file) as f:
                benchmark_results = json.load(f)
            return benchmark_results
        else:
            print(f"Warning: Result file {result_file} not found")
            print(f"Current directory: {os.getcwd()}")
            return {}
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {}


# Read results and store it in a pandas dataframe
def extract_tpot_metrics(results: dict) -> dict[str, float]:
    """Extract TPOT metrics from benchmark results."""
    metrics = {}

    # Mean TPOT
    if "mean_tpot_ms" in results:
        metrics["mean_tpot_ms"] = results["mean_tpot_ms"]

    return metrics


def run_profiling_only(
    scenario_name: str,
    use_spec_decode: bool,
    client_max_concurrency: int,
) -> None:
    """Run a separate profiling run with 10% of prompts.

    The profiling run's TPOT is printed but NOT included in final results.
    This is called after the main benchmark to collect profiling traces
    without affecting benchmark accuracy.
    """
    import requests

    total_prompts = NUM_PROMPTS
    profile_count = int(total_prompts * PROFILE_PERCENTAGE)

    print(f"\n{'=' * 80}")
    print(f"[Profiling Run] Starting profiler and running {profile_count} prompts...")
    print("NOTE: Profiling TPOT will be printed but NOT included in final results.")
    print(f"{'=' * 80}\n")

    try:
        print("Starting profiler...")
        response = requests.post(f"{BASE_URL}/start_profile", timeout=10)
        if response.status_code == 200:
            print("✓ Profiler started successfully")
        else:
            print(f"⚠ Warning: Failed to start profiler: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠ Warning: Could not start profiler: {e}")

    profile_file = LOG_DIR / f"{scenario_name}_profile_results.json"
    profile_results = run_benchmark(
        scenario_name=f"{scenario_name}_profile",
        use_spec_decode=use_spec_decode,
        client_max_concurrency=client_max_concurrency,
        result_file=profile_file,
        num_prompts=profile_count,
        num_warmups=WARMUP_REQUESTS,  # Use same warmup as
        # main benchmark for consistency
    )

    # Print profiling TPOT but don't include in final results
    if profile_results:
        profile_metrics = extract_tpot_metrics(profile_results)
        profile_tpot = profile_metrics.get("mean_tpot_ms")
        if profile_tpot is not None:
            print(f"\n{'=' * 80}")
            print(f"Profiling Run TPOT: {profile_tpot:.2f} ms")
            print("(This value is NOT included in final benchmark results)")
            print(
                "NOTE: Profiling overhead may cause this to be higher than actual TPOT"
            )
            print(f"{'=' * 80}\n")

    # Delete profile file after use
    try:
        if profile_file.exists():
            profile_file.unlink()
            print(f"Deleted temporary file: {profile_file}")
    except Exception as e:
        print(f"Warning: Could not delete {profile_file}: {e}")

    # Stop profiling
    try:
        print("\nStopping profiler (this may take several minutes to flush traces)...")
        response = requests.post(f"{BASE_URL}/stop_profile", timeout=1800)
        if response.status_code == 200:
            print("✓ Profiler stopped and traces flushed successfully")
        else:
            print(f"⚠ Warning: Failed to stop profiler: HTTP {response.status_code}")
    except requests.exceptions.Timeout:
        print(
            "⚠ Warning: Stop profiler request timed out (traces may still be flushing)"
        )
        print(
            "  This is normal for large workloads - \
                traces may continue flushing in background"
        )
    except Exception as e:
        print(f"⚠ Warning: Could not stop profiler: {e}")
        print("  Traces may still be flushing in the background")

    print("Waiting for profiler to finish writing traces...")
    time.sleep(10)  # Give extra time for trace flushing


def combine_benchmark_results(results_list: list[dict], total_prompts: int) -> dict:
    """Combine results from multiple benchmark phases into a single result."""
    if not results_list:
        return {}

    # Extract all TPOT values for percentile calculation
    all_tpot_values = []
    total_successful = 0
    total_failed = 0
    weighted_tpot_sum = 0.0
    total_weight = 0

    for result in results_list:
        if not result:
            continue

        # Get number of prompts in this phase
        phase_prompts = result.get("num_prompts", 0)
        if phase_prompts == 0:
            # Try to infer from successful requests
            phase_prompts = result.get("successful_requests", 0)

        successful = result.get("successful_requests", 0)
        failed = result.get("failed_requests", 0)
        total_successful += successful
        total_failed += failed

        # Get TPOT values
        mean_tpot = result.get("mean_tpot_ms")
        if mean_tpot is not None and phase_prompts > 0:
            weighted_tpot_sum += mean_tpot * phase_prompts
            total_weight += phase_prompts
        else:
            print("Something wrong")

        # Collect individual TPOT values if available
        if "tpot_ms" in result:
            all_tpot_values.extend(result["tpot_ms"])

    # Calculate combined metrics
    combined = {}

    if total_weight > 0:
        combined["mean_tpot_ms"] = weighted_tpot_sum / total_weight

    # Calculate percentiles if we have individual values
    if all_tpot_values:
        all_tpot_values.sort()
        n = len(all_tpot_values)
        if n > 0:
            # P90
            p90_idx = int(0.9 * n)
            if p90_idx >= n:
                p90_idx = n - 1
            combined["p90_tpot_ms"] = all_tpot_values[p90_idx]

            # Median
            median_idx = n // 2
            combined["median_tpot_ms"] = all_tpot_values[median_idx]

    # Add request counts
    combined["num_requests"] = total_prompts
    combined["successful_requests"] = total_successful
    combined["failed_requests"] = total_failed

    return combined
