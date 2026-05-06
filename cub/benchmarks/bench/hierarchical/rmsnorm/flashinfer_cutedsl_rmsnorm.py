# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for the direct FlashInfer CuTe DSL RMSNorm kernel.

This intentionally calls flashinfer.norm.kernels.rmsnorm.rmsnorm_cute instead of
the public flashinfer.norm.rmsnorm dispatcher, so the measured backend is
unambiguous.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cuda.bench as bench
from rmsnorm_common import (  # noqa: E402
    RMS_NORM_EPS,
    add_rmsnorm_axes,
    add_rmsnorm_counters,
    allocate_rmsnorm_tensors,
    as_torch_stream,
    is_oom_error,
    require_torch,
    torch_dtype,
)


def require_flashinfer_cutedsl(state: bench.State):
    try:
        from flashinfer.norm.kernels.rmsnorm import rmsnorm_cute
    except (ImportError, AttributeError):
        state.skip("Skipping: FlashInfer CuTe DSL RMSNorm is not available.")
        return None
    return rmsnorm_cute


def bench_flashinfer_cutedsl_rmsnorm(state: bench.State) -> None:
    torch = require_torch(state)
    if torch is None:
        return

    rmsnorm_cute = require_flashinfer_cutedsl(state)
    if rmsnorm_cute is None:
        return

    dtype_name = state.get_string("T{ct}")
    dtype = torch_dtype(torch, dtype_name)
    batch_size = int(state.get_int64("BatchSize"))
    hidden_size = int(state.get_int64("HiddenSize"))
    zero_data = state.get_int64("ZeroData") != 0

    try:
        x, weight, out = allocate_rmsnorm_tensors(
            state, torch, batch_size, hidden_size, dtype, zero_data
        )
    except RuntimeError as exc:
        if is_oom_error(exc):
            state.skip("Skipping: out of memory.")
            return
        raise

    add_rmsnorm_counters(state, x, weight, out)

    try:
        stream = as_torch_stream(torch, state.get_stream(), state.get_device())
        with torch.cuda.stream(stream):
            rmsnorm_cute(
                x, weight, out, RMS_NORM_EPS, weight_bias=0.0, enable_pdl=False
            )
        torch.cuda.synchronize(state.get_device())
    except RuntimeError as exc:
        if is_oom_error(exc):
            state.skip("Skipping: out of memory.")
            return
        raise

    def launcher(launch: bench.Launch) -> None:
        stream = as_torch_stream(torch, launch.get_stream(), state.get_device())
        with torch.cuda.stream(stream):
            rmsnorm_cute(
                x, weight, out, RMS_NORM_EPS, weight_bias=0.0, enable_pdl=False
            )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_flashinfer_cutedsl_rmsnorm)
    b.set_name("flashinfer_cutedsl_rmsnorm")
    add_rmsnorm_axes(b)
    bench.run_all_benchmarks(sys.argv)
