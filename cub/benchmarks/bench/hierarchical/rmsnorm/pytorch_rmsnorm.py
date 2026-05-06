# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for PyTorch RMSNorm.

C++ comparison targets:
- cub/benchmarks/bench/hierarchical/rmsnorm/hierarchical_rmsnorm.cu
- cub/benchmarks/bench/hierarchical/rmsnorm/cub_rmsnorm.cu
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


def bench_pytorch_rmsnorm(state: bench.State) -> None:
    torch = require_torch(state)
    if torch is None:
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

    # Keep the most recent output alive until after the timed launch.
    result = [out]

    stream = as_torch_stream(torch, state.get_stream(), state.get_device())
    with torch.cuda.stream(stream):
        result[0] = torch.nn.functional.rms_norm(
            x,
            (hidden_size,),
            weight=weight,
            eps=RMS_NORM_EPS,
        )
    torch.cuda.synchronize(state.get_device())

    def launcher(launch: bench.Launch) -> None:
        stream = as_torch_stream(torch, launch.get_stream(), state.get_device())
        with torch.cuda.stream(stream):
            result[0] = torch.nn.functional.rms_norm(
                x,
                (hidden_size,),
                weight=weight,
                eps=RMS_NORM_EPS,
            )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_pytorch_rmsnorm)
    b.set_name("pytorch_rmsnorm")
    add_rmsnorm_axes(b)
    bench.run_all_benchmarks(sys.argv)
