#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script to compare Vanilla vs Speculative Decoding performance
for different server/client concurrency scenarios.

Scenarios:
1. Server max_num_seqs=32, Client max_concurrency=4
2. Server max_num_seqs=4, Client max_concurrency=None (unlimited)

Usage:
    python benchmark_spec_decode_comparison.py

Requirements:
    - vllm installed and in PATH
    - requests, matplotlib, numpy packages
    - Model and draft model accessible (default: Qwen/Qwen3-8B
    and RedHatAI/Qwen3-8B-speculator.eagle3)

Output:
    - JSON result files in /log/ directory
    - Comparison plots: tpot_comparison.png and tpot_summary_table.png in /log/
    - Results summary: benchmark_results_summary.json in /log/
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Warning: requests not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Warning: matplotlib/numpy not installed. Installing...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "matplotlib", "numpy"]
    )
    import matplotlib.pyplot as plt
    import numpy as np

# Configuration

RUN_LOC = "local"
if RUN_LOC == "local":
    MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    DRAFT_MODEL = "nm-testing/Llama3_2_1B_speculator.eagle3"
elif RUN_LOC == "server":
    MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # 30B MoE model
    DRAFT_MODEL = "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"

PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
LOG_DIR = Path(os.path.join(os.getcwd(), "log", "spec_script"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark parameters
NUM_PROMPTS = 50
RANDOM_INPUT_LEN = 100
RANDOM_OUTPUT_LEN = 128
REQUEST_RATE = 2.0
WARMUP_REQUESTS = 5

# Scenarios
SCENARIOS = [
    {
        "name": "Scenario1_Server32_Client4",
        "server_max_num_seqs": 32,
        "client_max_concurrency": 4,
    },
    {
        "name": "Scenario2_Server4_Client4",
        "server_max_num_seqs": 4,
        "client_max_concurrency": 4,
    },
]

# Results storage
results: dict[str, dict[str, dict]] = {}


def wait_for_server(timeout: int = 120) -> bool:
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
    use_spec_decode: bool,
    max_num_seqs: int,
    port: int = PORT,
) -> subprocess.Popen:
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
        str(max_num_seqs),
        "--gpu-memory-utilization",
        "0.95",
        # "--disable-log-stats", "False",
    ]

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

    print(f"\n{'=' * 80}")
    print(f"Starting {'Spec Decode' if use_spec_decode else 'Vanilla'} server...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for server to be ready and get actual model name
    print("Waiting for server to be ready...")
    is_ready = wait_for_server()
    if is_ready:
        print("✓ Server is ready!")
        return process
    else:
        print("✗ Server failed to start within timeout")
        process.terminate()
        raise RuntimeError("Server failed to start")


def stop_server(process: subprocess.Popen):
    """Stop the vLLM server."""
    print("\nStopping server...")
    try:
        process.terminate()
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    print("✓ Server stopped")


def run_benchmark(
    scenario_name: str,
    use_spec_decode: bool,
    client_max_concurrency: int | None,
    result_file: Path,
) -> dict:
    """Run benchmark and return results."""
    config_type = "spec_decode" if use_spec_decode else "vanilla"
    model_name = MODEL

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
        str(NUM_PROMPTS),
        "--request-rate",
        str(REQUEST_RATE),
        "--num-warmups",
        str(WARMUP_REQUESTS),
        "--save-result",
        "--result-dir",
        str(LOG_DIR),
        "--result-filename",
        result_file.name,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
    ]

    if client_max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(client_max_concurrency)])

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
            # Try to find the file in current directory (vLLM default)
            current_dir_file = Path(result_file.name)
            if current_dir_file.exists():
                print(f"Found result file in current directory: {current_dir_file}")
                with open(current_dir_file) as f:
                    benchmark_results = json.load(f)
                # Move to log directory
                current_dir_file.rename(result_file)
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


def extract_tpot_metrics(results: dict) -> dict[str, float]:
    """Extract TPOT metrics from benchmark results."""
    metrics = {}

    # Mean TPOT
    if "mean_tpot_ms" in results:
        metrics["mean_tpot_ms"] = results["mean_tpot_ms"]

    # Median TPOT
    if "median_tpot_ms" in results:
        metrics["median_tpot_ms"] = results["median_tpot_ms"]

    # Percentiles
    for key in ["p50_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms"]:
        if key in results:
            metrics[key] = results[key]

    # Individual TPOT values (if available)
    if "tpots" in results:
        metrics["tpots"] = results["tpots"]

    return metrics


def create_comparison_plot(results: dict, output_path: Path):
    """Create matplotlib comparison plots for TPOT metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Vanilla vs Speculative Decoding TPOT Comparison",
        fontsize=16,
        fontweight="bold",
    )

    scenarios = list(results.keys())
    metrics_to_plot = [
        ("mean_tpot_ms", "Mean TPOT (ms)", axes[0, 0]),
        ("median_tpot_ms", "Median TPOT (ms)", axes[0, 1]),
        ("p95_tpot_ms", "P95 TPOT (ms)", axes[1, 0]),
        ("p99_tpot_ms", "P99 TPOT (ms)", axes[1, 1]),
    ]

    for metric_key, metric_title, ax in metrics_to_plot:
        vanilla_values = []
        spec_decode_values = []
        scenario_labels = []

        for scenario_name in scenarios:
            if scenario_name in results:
                scenario_labels.append(scenario_name.replace("_", "\n"))
                vanilla_data = (
                    results[scenario_name].get("vanilla", {}).get(metric_key, 0)
                )
                spec_data = (
                    results[scenario_name].get("spec_decode", {}).get(metric_key, 0)
                )
                vanilla_values.append(vanilla_data)
                spec_decode_values.append(spec_data)

        x = np.arange(len(scenario_labels))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            vanilla_values,
            width,
            label="Vanilla",
            color="#2E86AB",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            spec_decode_values,
            width,
            label="Spec Decode",
            color="#A23B72",
            alpha=0.8,
        )

        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {output_path}")

    # Also create a summary table plot
    create_summary_table(results, output_path.parent / "tpot_summary_table.png")


def create_summary_table(results: dict, output_path: Path):
    """Create a summary table with all TPOT metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")

    # Prepare data
    table_data = []
    headers = ["Scenario", "Config", "Mean", "Median", "P95", "P99"]

    for scenario_name in sorted(results.keys()):
        scenario_data = results[scenario_name]
        for config_type in ["vanilla", "spec_decode"]:
            if config_type in scenario_data:
                metrics = scenario_data[config_type]
                row = [
                    scenario_name.replace("_", " "),
                    config_type.replace("_", " ").title(),
                    f"{metrics.get('mean_tpot_ms', 0):.2f}",
                    f"{metrics.get('median_tpot_ms', 0):.2f}",
                    f"{metrics.get('p95_tpot_ms', 0):.2f}",
                    f"{metrics.get('p99_tpot_ms', 0):.2f}",
                ]
                table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4A90E2")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F0F0F0")

    plt.title("TPOT Metrics Summary (ms)", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Summary table saved to: {output_path}")


def main():
    """Main benchmark execution."""
    print("=" * 80)
    print("Vanilla vs Speculative Decoding Benchmark")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Draft Model: {DRAFT_MODEL}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("=" * 80)

    server_process = None

    try:
        for scenario in SCENARIOS:
            scenario_name = scenario["name"]
            server_max_seqs = scenario["server_max_num_seqs"]
            client_max_concurrency = scenario["client_max_concurrency"]

            results[scenario_name] = {}

            # Test Vanilla configuration
            print(f"\n{'#' * 80}")
            print(f"# Testing VANILLA: {scenario_name}")
            print(f"{'#' * 80}")

            server_process = start_server(
                use_spec_decode=False,
                max_num_seqs=server_max_seqs,
            )

            result_file_vanilla = LOG_DIR / f"{scenario_name}_vanilla_results.json"
            vanilla_results = run_benchmark(
                scenario_name=scenario_name,
                use_spec_decode=False,
                client_max_concurrency=client_max_concurrency,
                result_file=result_file_vanilla,
            )

            stop_server(server_process)
            server_process = None
            time.sleep(5)  # Wait before starting next server

            # Extract TPOT metrics
            results[scenario_name]["vanilla"] = extract_tpot_metrics(vanilla_results)

            # Test Spec Decode configuration
            print(f"\n{'#' * 80}")
            print(f"# Testing SPEC DECODE: {scenario_name}")
            print(f"{'#' * 80}")

            server_process = start_server(
                use_spec_decode=True,
                max_num_seqs=server_max_seqs,
            )

            result_file_spec = LOG_DIR / f"{scenario_name}_spec_decode_results.json"
            spec_results = run_benchmark(
                scenario_name=scenario_name,
                use_spec_decode=True,
                client_max_concurrency=client_max_concurrency,
                result_file=result_file_spec,
            )

            stop_server(server_process)
            server_process = None
            time.sleep(5)  # Wait before starting next server

            # Extract TPOT metrics
            results[scenario_name]["spec_decode"] = extract_tpot_metrics(spec_results)

            # Print summary for this scenario
            print(f"\n{'=' * 80}")
            print(f"Summary for {scenario_name}:")
            print(f"{'=' * 80}")
            print(
                f"Vanilla Mean TPOT: {
                    results[scenario_name]['vanilla'].get('mean_tpot_ms', 'N/A')
                } ms"
            )
            print(
                f"Spec Decode Mean TPOT: {
                    results[scenario_name]['spec_decode'].get('mean_tpot_ms', 'N/A')
                } ms"
            )
            if (
                "mean_tpot_ms" in results[scenario_name]["vanilla"]
                and "mean_tpot_ms" in results[scenario_name]["spec_decode"]
            ):
                speedup = (
                    results[scenario_name]["vanilla"]["mean_tpot_ms"]
                    / results[scenario_name]["spec_decode"]["mean_tpot_ms"]
                )
                print(f"Speedup: {speedup:.2f}x")
            print(f"{'=' * 80}\n")

        # Create plots
        print("\n" + "=" * 80)
        print("Creating comparison plots...")
        print("=" * 80)

        plot_path = LOG_DIR / "tpot_comparison.png"
        create_comparison_plot(results, plot_path)

        # Save results summary
        summary_path = LOG_DIR / "benchmark_results_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results summary saved to: {summary_path}")

        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        if server_process:
            stop_server(server_process)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()
        if server_process:
            stop_server(server_process)
        sys.exit(1)


if __name__ == "__main__":
    main()
