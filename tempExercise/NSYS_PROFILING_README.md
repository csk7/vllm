# NVIDIA Nsight Systems Profiling for vLLM

This guide explains how to use NVIDIA Nsight Systems (nsys) to profile your vLLM inference with GPU traces.

## Prerequisites

1. **Install nsys** (if not already installed):
   ```bash
   # Ubuntu/Debian
   apt update
   apt install -y --no-install-recommends gnupg
   echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
   apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   apt update
   apt install nsight-systems-cli
   ```

2. **Verify installation**:
   ```bash
   nsys --version
   ```

## Quick Start

### Option 1: Use the provided shell script (Recommended)

```bash
cd /home/siva/sivaHome/vLLM/v1/vllm/tempExercise
./run_nsys_profile.sh
```

### Option 2: Run manually

```bash
cd /home/siva/sivaHome/vLLM/v1/vllm/tempExercise
export VLLM_WORKER_MULTIPROC_METHOD=spawn
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --output=nsys_profile \
    python spec_decode_v2_nsys.py
```

## Key Differences from Torch Profiler

1. **Profiler Config**: Changed from `"profiler": "torch"` to `"profiler": "cuda"` in the script
2. **Environment Variable**: Added `VLLM_WORKER_MULTIPROC_METHOD=spawn` (required for nsys)
3. **No trace directory needed**: nsys saves directly to `.nsys-rep` file

## What Gets Profiled

- **LLM Configuration**: Your model config, dtype, max_model_len, etc.
- **Sampling Parameters**: temperature, top_p, max_tokens
- **GPU Operations**: All CUDA kernels, memory transfers, CUDA graph operations
- **Speculative Decoding**: Eagle3 draft model operations

The prompts themselves don't affect what gets profiled - only the LLM config and sampling params matter.

## Viewing Results

### GUI (Recommended)
```bash
nsys-ui nsys_profile.nsys-rep
```

### CLI Summary
```bash
nsys stats nsys_profile.nsys-rep
```

This will show:
- CUDA GPU Kernel Summary (with timing)
- Memory operations
- API calls
- And more detailed metrics

## Customizing the Profile

### Change output file name:
```bash
nsys profile --output=my_profile ... python spec_decode_v2_nsys.py
```

### Add more detailed tracing:
```bash
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx,osrt \
    --output=nsys_profile \
    python spec_decode_v2_nsys.py
```

### Profile specific time range:
```bash
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    --output=nsys_profile \
    python spec_decode_v2_nsys.py
```

## Important Notes

1. **CUDA Graphs**: The `--cuda-graph-trace=node` flag is crucial for seeing operations inside CUDA graphs (which vLLM uses by default)

2. **Multiprocessing**: `VLLM_WORKER_MULTIPROC_METHOD=spawn` is required for nsys to properly trace worker processes

3. **File Size**: nsys profiles can be large (several GB). Consider profiling only a few requests

4. **Performance Impact**: nsys profiling adds overhead. Use for profiling, not benchmarking

## Troubleshooting

### "nsys: command not found"
- Install nsys using the commands above
- Or download from: https://developer.nvidia.com/nsight-systems

### No GPU traces visible
- Ensure `--cuda-graph-trace=node` is included
- Check that CUDA is available: `nvidia-smi`
- Verify profiler config uses `"profiler": "cuda"`

### Profile file is empty or too small
- Make sure `llm.start_profile()` and `llm.stop_profile()` are called
- Check that actual inference happens between start/stop
- Increase the number of tokens or requests being profiled
