# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, schedule
from torch.profiler import profile as torch_profile
from torch.profiler import record_function as torch_record_function

device = "cuda" if torch.cuda.is_available() else "cpu"


class leNet(nn.Module):
    def __init__(
        self, in_channels=3, hidden_channels=32, out_channels=8, num_h_layers=4
    ):
        super().__init__()
        self.cascaded_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="zeros",
            ),
            *[
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    padding_mode="zeros",
                )
                for _ in range(num_h_layers)
            ],
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="zeros",
            ),
        )

    def forward(self, x):
        return self.cascaded_conv(x)


def main():
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(2026)
    B = 1
    Cin = 3
    H = 1080
    W = 1920
    input = torch.randn((B, Cin, H, W), dtype=torch.float32, device=device)
    le_net = leNet().to(device)
    print(f"Model device: {next(le_net.parameters()).device}")

    # Add 1 forward pass with random weights
    for param in le_net.parameters():
        param.data = torch.randn_like(param.data)

    # Warmup run to initialize CUDA
    if device == "cuda":
        _ = le_net.forward(input)
        torch.cuda.synchronize()

    # Profile CPU and GPU activity during forward pass
    # Use torch.profiler with schedule for better CUDA support
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def trace_handler(prof):
        profiler_dir = os.path.abspath("./log/vllm_profile")
        os.makedirs(profiler_dir, exist_ok=True)
        store_path = os.path.join(profiler_dir, "trace_test.json")
        prof.export_chrome_trace(store_path)
        print(
            f"\nProfile exported to {store_path}. \
            Open it in chrome://tracing"
        )

    with torch_profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        prof.start()
        with torch_record_function("forward_pass"):
            _ = le_net.forward(input)

        # Synchronize GPU to ensure all operations
        #   complete before profiler ends
        if device == "cuda":
            torch.cuda.synchronize()

        prof.step()

    # Print profiling results
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
            row_limit=20,
        )
    )


if __name__ == "__main__":
    main()
