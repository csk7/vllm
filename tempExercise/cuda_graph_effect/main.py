# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# imports and global info
import sys
import time

import pandas as pd
import requests
import torch
from global_vars import *
from server_ops import vllmServer

from tools import extract_tpot_metrics, run_benchmark


def start_server(
    server, use_spec_decode, disable_cuda_graphs, server_max_seqs, scenario_name
):
    server.start_server(
        use_spec_decode=use_spec_decode,
        disable_cuda_graphs=disable_cuda_graphs,
        seq_low_high=server_max_seqs,
    )

    result_file = LOG_DIR / f"{scenario_name}_results.json"
    # Start profiling via HTTP API
    if PROFILE:
        try:
            print("Starting profiler...")
            response = requests.post(f"{BASE_URL}/start_profile", timeout=10)
            if response.status_code == 200:
                print("✓ Profiler started successfully")
            else:
                print(
                    f"⚠ Warning: Failed to start profiler: \
                        HTTP {response.status_code}"
                )
                if response.text:
                    print(f"  Response: {response.text}")
        except requests.exceptions.Timeout:
            print("⚠ Warning: Start profiler request timed out")
        except Exception as e:
            print(f"⚠ Warning: Could not start profiler: {e}")

    return result_file


def benchmark(scenario_name, use_spec_decode, client_max_concurrency, result_file):
    benchmark_results = run_benchmark(
        scenario_name=scenario_name,
        use_spec_decode=use_spec_decode,
        client_max_concurrency=client_max_concurrency,
        result_file=result_file,
    )

    # Stop profiling via HTTP API
    # Note: stop_profile can take a long time
    # (up to 10+ minutes) to flush traces
    if PROFILE:
        try:
            print(
                "\nStopping profiler (this may take \
                    several minutes to flush traces)..."
            )
            # Use a very long timeout (30 minutes)
            # as recommended in vLLM docs
            response = requests.post(f"{BASE_URL}/stop_profile", timeout=1800)
            if response.status_code == 200:
                print(
                    "✓ Profiler stopped and traces \
                        flushed successfully"
                )
            else:
                print(
                    f"⚠ Warning: Failed to stop profiler:\
                        HTTP {response.status_code}"
                )
                if response.text:
                    print(f"  Response: {response.text}")
        except requests.exceptions.Timeout:
            print(
                "⚠ Warning: Stop profiler request \
                    timed out (traces may still be flushing)"
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

    return benchmark_results


def write_result_json(
    results_data,
    tpot_low,
    tpot_high,
    use_spec_decode,
    disable_cuda_graphs,
    client_max_concurrency,
    seq_low,
    seq_high,
):
    if tpot_low is not None and tpot_high is not None and tpot_high > 0:
        speedup = tpot_low / tpot_high
        improvement = (speedup - 1) * 100

        # Add row to results
        results_data.append(
            {
                "Spec": "spec" if use_spec_decode else "nospec",
                "CUDAGraph": "enabled" if not disable_cuda_graphs else "disabled",
                "Concur": client_max_concurrency,
                "SeqsLow": seq_low,
                "SeqsHigh": seq_high if seq_high is not None else "N/A",
                "TPOT_Low(ms)": round(tpot_low, 2),
                "TPOT_High(ms)": round(tpot_high, 2),
                "Speedup": round(speedup, 4),
                "Improvement": round(improvement, 2),
            }
        )

    elif tpot_low is not None:
        # Handle case where we only have one value (no high)
        results_data.append(
            {
                "Spec": "spec" if use_spec_decode else "nospec",
                "CUDAGraph": "enabled" if not disable_cuda_graphs else "disabled",
                "Concur": client_max_concurrency,
                "SeqsLow": seq_low,
                "SeqsHigh": "N/A",
                "TPOT_Low(ms)": round(tpot_low, 2),
                "TPOT_High(ms)": "N/A",
                "Speedup": "N/A",
                "Improvement": "N/A",
            }
        )

    elif tpot_high is not None:
        print(
            f"Warning: Missing TPOT values for Spec={use_spec_decode}, "
            f"CUDAGraph={not disable_cuda_graphs}, \
                Concur={client_max_concurrency}"
        )

    return results_data


def main():
    spec_decode = [False]
    # CUDA_graph_en = [True, False]
    # concur_num_seq = [(4, [4, 32]), (32, [32, 128])]
    CUDA_graph_en = [False, True]
    concur_num_seq = [(4, [4, 32])]

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
                seq_high = seq_low_high_list[1] if len(seq_low_high_list) > 1 else None

                tpot_low = None
                tpot_high = None

                # Test both low and high seq values
                for max_num_seq in seq_low_high_list:
                    server = None
                    try:
                        # Set scenario variables and logs
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
                        # Start server

                        server = vllmServer()

                        result_file = start_server(
                            server,
                            use_spec_decode,
                            disable_cuda_graphs,
                            server_max_seqs,
                            scenario_name,
                        )

                        # Benchmark
                        benchmark_results = benchmark(
                            scenario_name,
                            use_spec_decode,
                            client_max_concurrency,
                            result_file,
                        )

                        # Stop server
                        server.stop_server()
                        server = None
                        time.sleep(5)  # Wait before starting next server

                        # Extract TPOT metrics
                        metrics = extract_tpot_metrics(benchmark_results)
                        tpot_value = metrics.get("mean_tpot_ms")

                        if max_num_seq == seq_low:
                            tpot_low = tpot_value
                        elif seq_high is not None and max_num_seq == seq_high:
                            tpot_high = tpot_value

                        results_data = write_result_json(
                            results_data,
                            tpot_low,
                            tpot_high,
                            use_spec_decode,
                            disable_cuda_graphs,
                            client_max_concurrency,
                            seq_low,
                            seq_high,
                        )

                        print(f"Done: {scenario_name} : TPOT = {tpot_value} ms")

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
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
        torch.cuda.empty_cache()
    main()
