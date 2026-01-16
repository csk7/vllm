# How to Add NVTX Markers in vLLM

This guide explains the different ways to add NVTX (NVIDIA Tools Extension) markers in vLLM for profiling with tools like Nsight Systems, Nsight Compute, and PyTorch Profiler.

## Overview

NVTX markers create named ranges in your code that appear in profiling tools, making it easier to identify different phases of execution. vLLM provides several methods to add NVTX markers:

1. **Environment Variable Approach** (Recommended for existing code)
2. **Direct NVTX Usage** (For custom code)
3. **Using `record_function_or_nullcontext`** (vLLM's built-in helper)
4. **Layerwise NVTX Hooks** (Automatic per-layer markers)

---

## Method 1: Environment Variable Approach

The easiest way to enable NVTX markers in existing vLLM code is to set an environment variable. This automatically converts all `record_function_or_nullcontext()` calls to use NVTX.

### Usage

```python
import os

# Enable NVTX markers globally
os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"

# Your existing vLLM code will automatically use NVTX
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
```

### How It Works

When `VLLM_NVTX_SCOPES_FOR_PROFILING=1` is set, vLLM's `record_function_or_nullcontext()` function automatically uses `nvtx.annotate()` instead of a no-op context manager. This means all existing profiling markers in vLLM (like `"llm_engine step: get_output"`, `"gpu_model_runner: forward"`, etc.) will appear in your profiler.

### Example

```python
import os
os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"

from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
prompts = ["What is machine learning?"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

outputs = llm.generate(prompts, sampling_params)
```

When profiling with nsys, you'll see markers like:
- `llm_engine step: get_output`
- `llm_engine step: process_outputs`
- `gpu_model_runner: forward`
- `gpu_model_runner: preprocess`
- etc.

---

## Method 2: Direct NVTX Usage

For custom code or when you need fine-grained control, you can use NVTX directly.

### Import Options

```python
# Option 1: Use torch's NVTX (recommended)
import torch.cuda.nvtx as nvtx

# Option 2: Use standalone nvtx package
import nvtx
```

### Using Context Managers

```python
import torch.cuda.nvtx as nvtx

# Simple context manager
with nvtx.annotate("my_custom_operation"):
    # Your code here
    result = some_expensive_operation()

# With color (for better visualization in profiler)
with nvtx.annotate("my_operation", color="green"):
    result = some_expensive_operation()

# Using torch.cuda.nvtx.range (alternative)
with torch.cuda.nvtx.range("my_operation"):
    result = some_expensive_operation()
```

### Manual Push/Pop

```python
import torch.cuda.nvtx as nvtx

# Push a marker
nvtx.range_push("start_processing")

try:
    # Your code here
    result = process_data()
finally:
    # Pop the marker
    nvtx.range_pop()
```

### Example: Adding Custom Markers to Your Code

```python
import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
prompts = ["What is machine learning?"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

with nvtx.annotate("user_code: setup"):
    # Setup code
    pass

with nvtx.annotate("user_code: generation"):
    outputs = llm.generate(prompts, sampling_params)

with nvtx.annotate("user_code: postprocess"):
    # Post-processing
    for output in outputs:
        print(output.outputs[0].text)
```

---

## Method 3: Using `record_function_or_nullcontext`

This is vLLM's built-in helper function that automatically uses NVTX when the environment variable is set, or falls back to a no-op otherwise.

### Usage

```python
from vllm.v1.utils import record_function_or_nullcontext

# This will use NVTX if VLLM_NVTX_SCOPES_FOR_PROFILING=1
# Otherwise, it's a no-op (no overhead)
with record_function_or_nullcontext("my_custom_scope"):
    result = expensive_operation()
```

### Example: Adding to Custom Code

```python
import os
os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"

from vllm.v1.utils import record_function_or_nullcontext
from vllm import LLM, SamplingParams

def my_custom_function():
    with record_function_or_nullcontext("custom: data_preparation"):
        # Prepare data
        prompts = ["What is AI?"]
    
    with record_function_or_nullcontext("custom: model_inference"):
        llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
        outputs = llm.generate(prompts, SamplingParams(max_tokens=10))
    
    with record_function_or_nullcontext("custom: result_processing"):
        # Process results
        return outputs
```

---

## Method 4: Layerwise NVTX Hooks

For automatic per-layer NVTX markers in PyTorch models, vLLM provides layerwise NVTX hooks.

### Enable via Configuration

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    enable_layerwise_nvtx_tracing=True,  # Enable automatic layer markers
)
```

### Manual Usage

```python
from vllm.utils.nvtx_pytorch_hooks import PytHooks, layerwise_nvtx_marker_context

# Option 1: Register hooks for entire model
hooks = PytHooks()
hooks.register_hooks(your_model)

# Option 2: Use context manager for specific module
with layerwise_nvtx_marker_context("Module:MyLayer", layer_module, in_tensor=inputs):
    output = layer_module(inputs)
```

### Important Notes

- **CUDA Graphs**: Layerwise NVTX tracing does NOT work with CUDA graphs enabled
- **Performance**: Adds overhead, use only for profiling, not production
- **Automatic**: When enabled, each layer gets its own NVTX marker with tensor information

---

## Complete Example: Adding NVTX to Your Profiling Script

Here's a complete example combining multiple methods:

```python
import os
import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams

# Method 1: Enable NVTX for vLLM's internal markers
os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"

def main():
    # Method 2: Add custom markers
    with nvtx.annotate("main: initialization"):
        llm = LLM(
            model="meta-llama/Llama-3.2-1B-Instruct",
            dtype="bfloat16",
            max_model_len=8192,
            # Method 4: Enable layerwise markers (optional)
            # enable_layerwise_nvtx_tracing=True,  # Uncomment if needed
        )
    
    prompts = ["What is machine learning?"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    with nvtx.annotate("main: generation"):
        outputs = llm.generate(prompts, sampling_params)
    
    with nvtx.annotate("main: cleanup"):
        del llm

if __name__ == "__main__":
    main()
```

---

## Profiling with Nsight Systems

Once you've added NVTX markers, profile with nsys:

```bash
# Basic profiling
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --trace=cuda,nvtx,osrt \
    --output=profile \
    python your_script.py

# View results
nsys-ui profile.nsys-rep
```

### Key Flags

- `--trace-fork-before-exec=true`: Required for multiprocessing (vLLM workers)
- `--cuda-graph-trace=node`: Shows operations inside CUDA graphs
- `--trace=cuda,nvtx,osrt`: Includes NVTX markers in trace
- `--output=profile`: Output file name

---

## Best Practices

1. **Use Environment Variable for Existing Code**: Set `VLLM_NVTX_SCOPES_FOR_PROFILING=1` to enable markers in vLLM's internal code without modification.

2. **Add Custom Markers for Your Code**: Use `nvtx.annotate()` or `record_function_or_nullcontext()` for your custom functions.

3. **Use Descriptive Names**: Use clear, hierarchical names like `"module: function"` or `"phase: operation"`.

4. **Keep Markers Focused**: Don't add too many markers - focus on important phases and operations.

5. **Disable in Production**: NVTX markers add overhead. Only enable during profiling.

6. **Layerwise Tracing Limitations**: Remember that layerwise NVTX tracing doesn't work with CUDA graphs.

---

## Troubleshooting

### Markers Not Appearing in Profiler

1. **Check Environment Variable**: Ensure `VLLM_NVTX_SCOPES_FOR_PROFILING=1` is set before importing vLLM
2. **Check nsys Flags**: Include `--trace=nvtx` in your nsys command
3. **Verify Import**: Make sure `nvtx` is available: `python -c "import torch.cuda.nvtx; print('OK')"`

### Performance Impact

- NVTX markers have minimal overhead, but disable them in production
- Layerwise tracing adds more overhead - use only when needed

### CUDA Graphs Conflict

- Layerwise NVTX tracing cannot be used with CUDA graphs
- Use regular NVTX markers instead when CUDA graphs are enabled

---

## References

- vLLM profiling documentation: `docs/contributing/profiling.md`
- NVTX utilities: `vllm/utils/nvtx_pytorch_hooks.py`
- Profiler wrapper: `vllm/profiler/wrapper.py`
- vLLM utils: `vllm/v1/utils.py` (see `record_function_or_nullcontext`)
