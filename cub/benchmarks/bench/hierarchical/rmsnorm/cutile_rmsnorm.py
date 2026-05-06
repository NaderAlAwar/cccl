# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for CuTile RMSNorm variants.

The kernel shapes follow NVIDIA/cutile-python's RMSNorm benchmark:
https://github.com/NVIDIA/cutile-python/blob/c0eba4365b2ddbe3d1049150f2f24a1d27105834/test/bench_rms_norm.py
"""

from __future__ import annotations

import functools
import sys
from math import ceil
from pathlib import Path

import numpy as np

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


def next_power_of_2(n: int) -> int:
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@functools.cache
def get_cutile_kernels():
    import cuda.tile as ct

    @ct.kernel(occupancy=ct.ByTarget(sm_100=16))
    def rms_norm_kernel(
        x,
        w,
        out,
        rstd,
        n: ct.Constant[int],
        eps: ct.Constant[float],
        tile_size: ct.Constant[int],
    ):
        row = ct.bid(0)
        rms = ct.full((1, tile_size), 0.0, dtype=np.float32)
        num_tiles = ct.cdiv(x.shape[1], tile_size)
        for j in range(0, num_tiles):
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, tile_size),
                allow_tma=False,
                latency=1,
                padding_mode=ct.PaddingMode.ZERO,
            )
            xj = ct.astype(xj, np.float32)
            rms += xj * xj

        rms = ct.rsqrt(ct.sum(rms, axis=1, keepdims=False) / n + eps)
        ct.store(rstd, index=(row,), tile=rms)

        for j in range(0, num_tiles):
            wj = ct.load(
                w,
                index=(j,),
                shape=(tile_size,),
                allow_tma=False,
                latency=1,
                padding_mode=ct.PaddingMode.ZERO,
            )
            wj = ct.astype(wj, np.float32)
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, tile_size),
                allow_tma=False,
                latency=1,
                padding_mode=ct.PaddingMode.ZERO,
            )
            xj = ct.astype(xj, np.float32)
            yj = ct.astype(xj * rms * wj, x.dtype)
            ct.store(out, index=(row, j), tile=yj, allow_tma=False, latency=1)

    @ct.kernel
    def rms_norm_kernel_gather(
        x,
        w,
        out,
        rstd,
        n: ct.Constant[int],
        eps: ct.Constant[float],
        tile_size: ct.Constant[int],
    ):
        row = ct.bid(0)
        rms = ct.full((tile_size,), 0.0, dtype=np.float32)
        num_tiles = ct.cdiv(n, tile_size)
        offsets = ct.arange(tile_size, dtype=np.int32)
        for j in range(0, num_tiles):
            offs = j * tile_size + offsets
            xj = ct.gather(x, (row, offs), latency=1)
            xj = ct.astype(xj, np.float32)
            rms += xj * xj

        rms = ct.rsqrt(ct.sum(rms, axis=0, keepdims=False) / n + eps)
        ct.scatter(rstd, row, rms)

        for j in range(0, num_tiles):
            offs = j * tile_size + offsets
            wj = ct.gather(w, offs, latency=1)
            wj = ct.astype(wj, np.float32)
            xj = ct.gather(x, (row, offs), latency=1)
            xj = ct.astype(xj, np.float32)
            yj = ct.astype(xj * rms * wj, x.dtype)
            ct.scatter(out, (row, offs), yj, latency=1)

    @ct.kernel
    def rms_norm_kernel_static_persistent(
        x,
        out,
        w,
        tile_size_m: ct.Constant[int],
        tile_size_n: ct.Constant[int],
        eps: ct.Constant[float],
    ):
        bid = ct.bid(0)
        m = x.shape[0]
        n = x.shape[1]
        upper_bound = (m + tile_size_m - 1) // tile_size_m
        w_tile = ct.load(
            w,
            index=(0,),
            shape=(tile_size_n,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        w_tile = ct.astype(w_tile, np.float32)

        num_tile_blocks = ct.num_blocks(0)
        for current_bid in range(bid, upper_bound, num_tile_blocks):
            x_tile = ct.load(
                x,
                index=(current_bid, 0),
                shape=(tile_size_m, tile_size_n),
                latency=10,
                padding_mode=ct.PaddingMode.ZERO,
            )
            x_tile = ct.astype(x_tile, np.float32)
            x2_sum = ct.sum(ct.mul(x_tile, x_tile), axis=1, keepdims=True)
            n_f32 = ct.full((tile_size_m, 1), n * 1.0, dtype=np.float32)
            eps_tensor = ct.full((tile_size_m, 1), eps, dtype=np.float32)
            rsqrt_var = ct.rsqrt(ct.truediv(x2_sum, n_f32) + eps_tensor)
            w_broadcasted = ct.reshape(w_tile, (1, tile_size_n))
            y = ct.astype(ct.mul(ct.mul(x_tile, rsqrt_var), w_broadcasted), x.dtype)
            ct.store(out, index=(current_bid, 0), tile=y, allow_tma=False, latency=3)

    return (
        ct,
        rms_norm_kernel,
        rms_norm_kernel_gather,
        rms_norm_kernel_static_persistent,
    )


def require_cutile(state: bench.State):
    try:
        return get_cutile_kernels()
    except (ImportError, AttributeError):
        state.skip("Skipping: cuda.tile is not available.")
        return None


def launch_cutile_rmsnorm(torch, x, weight, out, mode: str) -> None:
    cutile = get_cutile_kernels()
    ct, rms_norm_kernel, rms_norm_kernel_gather, rms_norm_kernel_static_persistent = (
        cutile
    )
    m, n = x.shape

    if mode == "static_persistent":
        device_prop = torch.cuda.get_device_properties(x.device)
        num_sms = device_prop.multi_processor_count
        tile_size_n = next_power_of_2(n)
        tile_size_m = 2 if device_prop.major == 8 else 16
        if device_prop.major >= 9:
            if tile_size_n <= 1024:
                tile_size_m = 16
            elif tile_size_n >= 16384:
                tile_size_m = 2
            else:
                tile_size_m = 4
        grid_size = min(num_sms, ceil(m / tile_size_m) * ceil(n / tile_size_n))
        ct.launch(
            torch.cuda.current_stream(),
            (grid_size,),
            rms_norm_kernel_static_persistent,
            (x, out, weight, tile_size_m, tile_size_n, RMS_NORM_EPS),
        )
        return

    rstd = torch.empty((m,), dtype=torch.float32, device=x.device)
    max_fused_size = 2048 // x.element_size()
    tile_size = min(max_fused_size, next_power_of_2(n))
    kernel = rms_norm_kernel_gather if mode == "gather" else rms_norm_kernel
    ct.launch(
        torch.cuda.current_stream(),
        (m,),
        kernel,
        (x, weight, out, rstd, n, RMS_NORM_EPS, tile_size),
    )


def bench_cutile_rmsnorm(state: bench.State) -> None:
    torch = require_torch(state)
    if torch is None:
        return

    if require_cutile(state) is None:
        return

    dtype_name = state.get_string("T{ct}")
    dtype = torch_dtype(torch, dtype_name)
    mode = state.get_string("Mode{ct}")
    batch_size = int(state.get_int64("BatchSize"))
    hidden_size = int(state.get_int64("HiddenSize"))
    zero_data = state.get_int64("ZeroData") != 0

    if mode == "static_persistent" and hidden_size >= 16384:
        state.skip(
            "Skipping: CuTile static-persistent RMSNorm uses too much memory for this hidden size."
        )
        return

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
            launch_cutile_rmsnorm(torch, x, weight, out, mode)
        torch.cuda.synchronize(state.get_device())
    except RuntimeError as exc:
        if is_oom_error(exc):
            state.skip("Skipping: out of memory.")
            return
        raise

    def launcher(launch: bench.Launch) -> None:
        stream = as_torch_stream(torch, launch.get_stream(), state.get_device())
        with torch.cuda.stream(stream):
            launch_cutile_rmsnorm(torch, x, weight, out, mode)

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_cutile_rmsnorm)
    b.set_name("cutile_rmsnorm")
    b.add_string_axis("Mode{ct}", ["standard", "gather", "static_persistent"])
    add_rmsnorm_axes(b)
    bench.run_all_benchmarks(sys.argv)
