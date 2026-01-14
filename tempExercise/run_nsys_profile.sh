#!/bin/bash
# Script to run vLLM with NVIDIA Nsight Systems profiling
# Usage: ./run_nsys_profile.sh

# Set required environment variable for nsys
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Run with nsys profiling
# --trace-fork-before-exec=true: Required for multiprocessing
# --cuda-graph-trace=node: Captures operations inside CUDA graphs
# --output=nsys_profile: Output file name (will be nsys_profile.nsys-rep)
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --output=nsys_profile \
    --force-overwrite true \
    python spec_decode_v2_nsys.py

echo ""
echo "Profiling complete!"
echo "Profile saved as: nsys_profile.nsys-rep"
echo ""
echo "To view the profile:"
echo "  nsys-ui nsys_profile.nsys-rep"
echo ""
echo "To get summary statistics:"
echo "  nsys stats nsys_profile.nsys-rep"
