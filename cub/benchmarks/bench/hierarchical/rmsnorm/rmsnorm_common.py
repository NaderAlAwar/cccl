# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

import cuda.bench as bench

RMS_NORM_EPS = 1e-5

BATCH_SIZES = [64, 800, 150000]
HIDDEN_SIZES = [
    512,
    768,
    896,
    1024,
    1152,
    1280,
    1536,
    1600,
    2048,
    2304,
    2560,
    2880,
    3072,
    3584,
    3840,
    4096,
    4608,
    4868,
    5120,
    6144,
    6656,
    7168,
    8192,
    9736,
    12288,
    12980,
    16384,
    18432,
    19472,
    39572,
]

DTYPE_AXIS = ["F32", "F16", "BF16"]


def require_torch(state: bench.State):
    try:
        import torch
    except ImportError:
        state.skip("Skipping: PyTorch is not installed.")
        return None

    if not torch.cuda.is_available():
        state.skip("Skipping: CUDA is not available to PyTorch.")
        return None

    return torch


def torch_dtype(torch: Any, dtype_name: str):
    if dtype_name == "F32":
        return torch.float32
    if dtype_name == "F16":
        return torch.float16
    if dtype_name == "BF16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype axis value: {dtype_name}")


def as_torch_stream(torch: Any, cs: bench.CudaStream, device: int):
    return torch.cuda.ExternalStream(cs.addressof(), device=device)


def is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def allocate_rmsnorm_tensors(
    state: bench.State,
    torch: Any,
    batch_size: int,
    hidden_size: int,
    dtype: Any,
    zero_data: bool,
):
    device_id = state.get_device()
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    stream = as_torch_stream(torch, state.get_stream(), device_id)

    with torch.cuda.device(device), torch.cuda.stream(stream):
        if zero_data:
            x = torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
            weight = torch.zeros((hidden_size,), dtype=dtype, device=device)
        else:
            x = torch.empty((batch_size, hidden_size), dtype=dtype, device=device)
            weight = torch.empty((hidden_size,), dtype=dtype, device=device)
            x.uniform_(-1.0, 1.0)
            weight.uniform_(-1.0, 1.0)

        out = torch.empty_like(x)

    return x, weight, out


def add_rmsnorm_counters(state: bench.State, x: Any, weight: Any, out: Any) -> None:
    state.add_element_count(x.numel())
    state.add_global_memory_reads(x.numel() * x.element_size(), "Input")
    state.add_global_memory_reads(weight.numel() * weight.element_size(), "Weight")
    state.add_global_memory_writes(out.numel() * out.element_size(), "Output")


def add_rmsnorm_axes(b: bench.Benchmark) -> bench.Benchmark:
    b.add_string_axis("T{ct}", DTYPE_AXIS)
    b.add_int64_axis("BatchSize", BATCH_SIZES)
    b.add_int64_axis("ZeroData", [0, 1])
    b.add_int64_axis("HiddenSize", HIDDEN_SIZES)
    return b
