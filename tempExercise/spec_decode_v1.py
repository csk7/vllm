# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time

from vllm import LLM, SamplingParams


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

    # Create the LLM engine with Eagle3 speculative decoding
    llm = LLM(
        model=target_model,
        max_model_len=512,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        trust_remote_code=trust_remote_code,  # Set based on model type
        enforce_eager=True,
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
        max_tokens=64,
    )

    # Prompt
    prompt = "Winter is coming."

    # Run inference
    outputs = llm.generate([prompt], sampling_params)

    # Print result
    for output in outputs:
        print("Prompt:", output.prompt)
        print("Completion:", output.outputs[0].text)

    time.sleep(1)
    del llm


if __name__ == "__main__":
    main()
