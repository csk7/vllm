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
    use_profile_flag: bool = False,
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

    if use_profile_flag:
        cmd.append("--profile")

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

    Uses the built-in --profile flag which handles profiling timing correctly
    (starts after warmup, stops after benchmark).
    """
    total_prompts = NUM_PROMPTS
    profile_count = int(total_prompts * PROFILE_PERCENTAGE)

    profiler_type = "nsys" if NSYS_PROFILE else "torch"
    print(f"\n{'=' * 80}")
    print(
        f"[Profiling Run] Running {profile_count} \
            prompts with {profiler_type} profiling..."
    )
    print(
        "NOTE: Profiling TPOT will be printed \
        but NOT included in final results."
    )
    if NSYS_PROFILE:
        print("Using nsys profiling (lower overhead, ~5-20% vs torch ~30-60%).")
    else:
        print(
            "Using built-in --profile flag with torch \
                profiler (profiling starts after warmup)."
        )
    print(f"{'=' * 80}\n")

    profile_file = LOG_DIR / f"{scenario_name}_profile_results.json"
    profile_results = run_benchmark(
        scenario_name=f"{scenario_name}_profile",
        use_spec_decode=use_spec_decode,
        client_max_concurrency=client_max_concurrency,
        result_file=profile_file,
        num_prompts=profile_count,
        num_warmups=WARMUP_REQUESTS,  # Warmup needed for stable profiling results
        use_profile_flag=True,  # Use built-in --profile
        # flag instead of manual HTTP calls
    )

    # Print profiling TPOT but don't include in final results
    if profile_results:
        profile_metrics = extract_tpot_metrics(profile_results)
        profile_tpot = profile_metrics.get("mean_tpot_ms")
        if profile_tpot is not None:
            profiler_type = "nsys" if NSYS_PROFILE else "torch"
            overhead_range = "5-20%" if NSYS_PROFILE else "30-60%"
            print(f"\n{'=' * 80}")
            print(f"Profiling Run TPOT ({profiler_type}): {profile_tpot:.2f} ms")
            print("(This value is NOT included in final benchmark results)")
            print(
                f"WARNING: Profiling adds overhead (typically {overhead_range}). "
                "Profiling TPOT will always be higher than actual TPOT."
            )
            if NSYS_PROFILE:
                print("nsys profiling has lower overhead than torch profiler.")
            else:
                print(
                    "This is expected behavior - profiling \
                        instrumentation slows down inference."
                )
            print(f"{'=' * 80}\n")

    # Delete profile file after use
    try:
        if profile_file.exists():
            profile_file.unlink()
            print(f"Deleted temporary file: {profile_file}")
    except Exception as e:
        print(f"Warning: Could not delete {profile_file}: {e}")

    # Note: When using --profile flag,
    #  profiling is automatically stopped after benchmark
    # The profiler will flush traces in the background
    profiler_type = "nsys" if NSYS_PROFILE else "torch"
    print(f"\n{profiler_type.capitalize()} profiling completed.")
    if NSYS_PROFILE:
        # The actual filename includes scenario_name, but we show the base path here
        print(
            "nsys profile saved to: ./log/vllm_profile/serve_1_test/\
                nsys_profile_<scenario>.nsys-rep"
        )
        print(
            "View with: nsys-ui ./log/vllm_profile/serve_1_test/\
                nsys_profile_<scenario>.nsys-rep"
        )
    else:
        print("Traces are being flushed in the background...")
        print(
            "(This may take several minutes - traces continue flushing after benchmark)"
        )
        time.sleep(5)  # Brief wait for initial trace flushing
