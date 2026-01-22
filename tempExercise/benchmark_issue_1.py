# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# imports and global info
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests

spec_decode = [True, False]
CUDA_graph_en = [True, False]
concur_num_seq = [(4, [4, 32]), (32, [32, 128])]

# Server Params
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
RUN_LOC = "server"
if RUN_LOC == "local":
    MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    DRAFT_MODEL = "nm-testing/Llama3_2_1B_speculator.eagle3"
elif RUN_LOC == "server":
    MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # 30B MoE model
    DRAFT_MODEL = "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"

# Client Params
LOG_DIR = Path(os.path.join(os.getcwd(), "log", "spec_script"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_INPUT_LEN = 100
RANDOM_OUTPUT_LEN = 128
REQUEST_RATE = 2.0
WARMUP_REQUESTS = 5
NUM_PROMPTS = 50

# Server Configs --> [Spec, No Spec] x
#   [Cuda graphs, no cuda graphs] x [--max-num-seqs 4, 32 (Concur)]
# Bench Mark configs ---> [--max-num-seqs --> (Concur)],
#   [--num-prompts ---> (4/32) or (32/128)]
# Start Server


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
            cmd.extend(["--enforce-eager"])

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
                    for line in iter(self.process.stdout.readline, ''):
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
                print("\nServer process is still running but not responding to HTTP requests.")
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


# Start BenchMark
def run_benchmark(
    scenario_name: str,
    use_spec_decode: bool,
    client_max_concurrency: int,
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
        "--max-concurrency",
        str(client_max_concurrency),
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


def main():
    """Main benchmark execution."""
    print("=" * 80)
    print("Vanilla vs Speculative Decoding Benchmark")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Draft Model: {DRAFT_MODEL}")
    print("=" * 80)

    # Store results for DataFrame
    results_data = []

    # Test all combinations
    for use_spec_decode in spec_decode:
        for disable_cuda_graphs in CUDA_graph_en:
            for scenario in concur_num_seq:
                client_max_concurrency = scenario[0]
                seq_low_high_list = scenario[1]
                seq_low = seq_low_high_list[0]
                seq_high = seq_low_high_list[1]

                tpot_low = None
                tpot_high = None

                # Test both low and high seq values
                for max_num_seq in seq_low_high_list:
                    server = None
                    try:
                        server_max_seqs = max_num_seq
                        spec_str = "spec" if use_spec_decode else "nospec"
                        cuda_str = "disabled" if disable_cuda_graphs else "enabled"

                        print(f"\n{'#' * 80}")
                        print(
                            f"Case: Spec={spec_str}, CUDAGraph={cuda_str}, "
                            f"Concur={client_max_concurrency}, \
                                Max_num_seq={server_max_seqs}"
                        )
                        scenario_name = (
                            f"Case:_Spec_{spec_str}_CUDAGraph_{cuda_str}_"
                            f"Concur_{client_max_concurrency}_num_seq_{server_max_seqs}"
                        )
                        print(f"{'#' * 80}")

                        server = vllmServer()
                        server.start_server(
                            use_spec_decode=use_spec_decode,
                            disable_cuda_graphs=disable_cuda_graphs,
                            seq_low_high=server_max_seqs,
                        )

                        result_file = LOG_DIR / f"{scenario_name}_results.json"
                        benchmark_results = run_benchmark(
                            scenario_name=scenario_name,
                            use_spec_decode=use_spec_decode,
                            client_max_concurrency=client_max_concurrency,
                            result_file=result_file,
                        )

                        server.stop_server()
                        server = None
                        time.sleep(5)  # Wait before starting next server

                        # Extract TPOT metrics
                        metrics = extract_tpot_metrics(benchmark_results)
                        tpot_value = metrics.get("mean_tpot_ms")

                        if max_num_seq == seq_low:
                            tpot_low = tpot_value
                        elif max_num_seq == seq_high:
                            tpot_high = tpot_value

                        print(f"{scenario_name} : TPOT = {tpot_value} ms")

                    except KeyboardInterrupt:
                        print("\n\nBenchmark interrupted by user")
                        if server:
                            server.stop_server()
                        sys.exit(1)
                    except Exception as e:
                        print(f"\n\nError during benchmark: {e}")
                        import traceback

                        traceback.print_exc()
                        if server:
                            server.stop_server()
                        # Continue with next test instead of exiting
                        continue

                # Calculate speedup and improvement if we have both values
                if tpot_low is not None and tpot_high is not None and tpot_high > 0:
                    speedup = tpot_low / tpot_high
                    improvement = (speedup - 1) * 100

                    # Add row to results
                    results_data.append(
                        {
                            "Spec": "spec" if use_spec_decode else "nospec",
                            "CUDAGraph": "enabled"
                            if not disable_cuda_graphs
                            else "disabled",
                            "Concur": client_max_concurrency,
                            "SeqsLow": seq_low,
                            "SeqsHigh": seq_high,
                            "TPOT_Low(ms)": round(tpot_low, 2),
                            "TPOT_High(ms)": round(tpot_high, 2),
                            "Speedup": round(speedup, 4),
                            "Improvement": round(improvement, 2),
                        }
                    )
                elif tpot_low is not None or tpot_high is not None:
                    print(
                        f"Warning: Missing TPOT values for Spec={use_spec_decode}, "
                        f"CUDAGraph={not disable_cuda_graphs}, \
                            Concur={client_max_concurrency}"
                    )

    # Create pandas DataFrame
    if results_data:
        df = pd.DataFrame(results_data)

        # Sort DataFrame to match the format in komal_results.md
        # Order: spec enabled, spec disabled, nospec enabled, nospec disabled
        # Within each, by Concur (4, 32)
        df["Spec_order"] = df["Spec"].map({"spec": 0, "nospec": 1})
        df["CUDAGraph_order"] = df["CUDAGraph"].map({"enabled": 0, "disabled": 1})
        df = df.sort_values(["Spec_order", "CUDAGraph_order", "Concur"]).drop(
            ["Spec_order", "CUDAGraph_order"], axis=1
        )

        # Display results
        print("\n" + "=" * 80)
        print(f"Model: {MODEL}")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        # Save to CSV
        csv_file = LOG_DIR / "benchmark_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")

        return df
    else:
        print("\nNo results collected!")
        return pd.DataFrame()


if __name__ == "__main__":
    main()
