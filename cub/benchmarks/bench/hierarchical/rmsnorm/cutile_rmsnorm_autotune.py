# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for autotuned CuTile RMSNorm variants.

This is intentionally separate from cutile_rmsnorm.py: the baseline benchmark
uses fixed heuristics, while this file measures a finite config space once per
shape/dtype/mode and benchmarks the cached winner.
"""

from __future__ import annotations

import functools
import itertools
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import cuda.bench as bench
from rmsnorm_common import (  # noqa: E402
    RMS_NORM_EPS,
    add_rmsnorm_axes,
    add_rmsnorm_counters,
    allocate_rmsnorm_tensors,
    as_torch_stream,
    check_rmsnorm_correctness,
    is_oom_error,
    require_torch,
    torch_dtype,
)

STANDARD_TILE_SIZES = (2**7, 2**8, 2**9, 2**10, 2**11, 2**12)
STATIC_TILE_SIZE_M = (2, 4, 8, 16)
STATIC_TILE_SIZE_N = (2**9, 2**10, 2**11, 2**12, 2**13, 2**14)
NUM_CTAS = (1, 2)
OCCUPANCIES = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class StandardConfig:
    tile_size: int
    num_ctas: int
    occupancy: int


@dataclass(frozen=True)
class StaticPersistentConfig:
    tile_size_m: int
    tile_size_n: int
    num_ctas: int
    occupancy: int


@dataclass(frozen=True)
class TunedKernel:
    kernel: Any
    config: StandardConfig | StaticPersistentConfig


_tuning_cache: dict[tuple[Any, ...], TunedKernel] = {}


def get_cutile_module():
    import cuda.tile as ct

    return ct


def get_exhaustive_search():
    from cuda.tile.tune import exhaustive_search

    return exhaustive_search


def require_cutile_autotune(state: bench.State):
    try:
        ct = get_cutile_module()
        get_exhaustive_search()
    except (ImportError, AttributeError):
        state.skip("Skipping: cuda.tile autotuning support is not available.")
        return None
    return ct


@functools.cache
def get_standard_kernel():
    ct = get_cutile_module()

    @ct.kernel(occupancy=ct.ByTarget(sm_100=16))
    def rms_norm_kernel(
        x,
        w,
        out,
        rstd,
        N: ct.Constant[int],
        eps: ct.Constant[float],
        TILE_SIZE: ct.Constant[int],
    ):
        row = ct.bid(0)
        rms = ct.full((1, TILE_SIZE), 0.0, dtype=np.float32)
        num_tiles = ct.cdiv(x.shape[1], TILE_SIZE)

        for j in range(0, num_tiles):
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=1,
                padding_mode=ct.PaddingMode.ZERO,
            )
            xj = ct.astype(xj, np.float32)
            rms += xj * xj

        rms = ct.rsqrt(ct.sum(rms, axis=1, keepdims=False) / N + eps)
        ct.store(rstd, index=(row,), tile=rms)

        for j in range(0, num_tiles):
            wj = ct.load(
                w,
                index=(j,),
                shape=(TILE_SIZE,),
                allow_tma=False,
                latency=1,
                padding_mode=ct.PaddingMode.ZERO,
            )
            wj = ct.astype(wj, np.float32)
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE_SIZE),
                allow_tma=False,
                latency=1,
                padding_mode=ct.PaddingMode.ZERO,
            )
            xj = ct.astype(xj, np.float32)
            yj = ct.astype(xj * rms * wj, x.dtype)
            ct.store(out, index=(row, j), tile=yj, allow_tma=False, latency=1)

    return rms_norm_kernel


@functools.cache
def get_gather_kernel():
    ct = get_cutile_module()

    @ct.kernel
    def rms_norm_kernel_gather(
        x,
        w,
        out,
        rstd,
        N: ct.Constant[int],
        eps: ct.Constant[float],
        TILE_SIZE: ct.Constant[int],
    ):
        row = ct.bid(0)
        rms = ct.full((TILE_SIZE,), 0.0, dtype=np.float32)
        num_tiles = ct.cdiv(N, TILE_SIZE)
        offsets = ct.arange(TILE_SIZE, dtype=np.int32)

        for j in range(0, num_tiles):
            offs = j * TILE_SIZE + offsets
            xj = ct.gather(x, (row, offs), latency=1)
            xj = ct.astype(xj, np.float32)
            rms += xj * xj

        rms = ct.rsqrt(ct.sum(rms, axis=0, keepdims=False) / N + eps)
        ct.scatter(rstd, row, rms)

        for j in range(0, num_tiles):
            offs = j * TILE_SIZE + offsets
            wj = ct.gather(w, offs, latency=1)
            wj = ct.astype(wj, np.float32)
            xj = ct.gather(x, (row, offs), latency=1)
            xj = ct.astype(xj, np.float32)
            yj = ct.astype(xj * rms * wj, x.dtype)
            ct.scatter(out, (row, offs), yj, latency=1)

    return rms_norm_kernel_gather


@functools.cache
def get_static_persistent_kernel():
    ct = get_cutile_module()

    @ct.kernel
    def rms_norm_kernel_static_persistent(
        x,
        out,
        w,
        TILE_SIZE_M: ct.Constant[int],
        TILE_SIZE_N: ct.Constant[int],
        eps: ct.Constant[float],
    ):
        bid = ct.bid(0)
        m = x.shape[0]
        n = x.shape[1]
        upper_bound = (m + TILE_SIZE_M - 1) // TILE_SIZE_M

        w_tile = ct.load(
            w,
            index=(0,),
            shape=(TILE_SIZE_N,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        w_tile = ct.astype(w_tile, np.float32)

        num_tile_blocks = ct.num_blocks(0)
        for current_bid in range(bid, upper_bound, num_tile_blocks):
            x_tile = ct.load(
                x,
                index=(current_bid, 0),
                shape=(TILE_SIZE_M, TILE_SIZE_N),
                latency=10,
                padding_mode=ct.PaddingMode.ZERO,
            )
            x_tile = ct.astype(x_tile, np.float32)
            x2_sum = ct.sum(ct.mul(x_tile, x_tile), axis=1, keepdims=True)
            n_f32 = ct.full((TILE_SIZE_M, 1), n * 1.0, dtype=np.float32)
            eps_tensor = ct.full((TILE_SIZE_M, 1), eps, dtype=np.float32)
            rsqrt_var = ct.rsqrt(ct.truediv(x2_sum, n_f32) + eps_tensor)
            w_broadcasted = ct.reshape(w_tile, (1, TILE_SIZE_N))
            y = ct.astype(ct.mul(ct.mul(x_tile, rsqrt_var), w_broadcasted), x.dtype)
            ct.store(out, index=(current_bid, 0), tile=y, allow_tma=False, latency=3)

    return rms_norm_kernel_static_persistent


def _standard_autotune_configs():
    for tile_size, num_ctas, occupancy in itertools.product(
        STANDARD_TILE_SIZES, NUM_CTAS, OCCUPANCIES
    ):
        yield StandardConfig(tile_size, num_ctas, occupancy)


def _static_persistent_autotune_configs(hidden_size: int):
    for tile_size_m, tile_size_n, num_ctas, occupancy in itertools.product(
        STATIC_TILE_SIZE_M, STATIC_TILE_SIZE_N, NUM_CTAS, OCCUPANCIES
    ):
        if hidden_size * 2 > tile_size_n >= hidden_size:
            yield StaticPersistentConfig(
                tile_size_m,
                tile_size_n,
                num_ctas,
                occupancy,
            )


def _static_persistent_grid(torch, x, cfg: StaticPersistentConfig):
    device_prop = torch.cuda.get_device_properties(x.device)
    m, n = x.shape
    grid_size = min(
        device_prop.multi_processor_count,
        ceil(m / cfg.tile_size_m) * ceil(n / cfg.tile_size_n),
    )
    return (grid_size,)


def _cache_key(torch, x, mode: str):
    device_prop = torch.cuda.get_device_properties(x.device)
    return (
        mode,
        str(x.dtype),
        int(x.shape[0]),
        int(x.shape[1]),
        int(device_prop.major),
        int(device_prop.minor),
        int(device_prop.multi_processor_count),
    )


def _tune_standard_or_gather(torch, x, weight, out, mode: str) -> TunedKernel:
    exhaustive_search = get_exhaustive_search()
    m, n = x.shape
    rstd = torch.empty((m,), dtype=torch.float32, device=x.device)
    tune_out = torch.empty_like(out)
    tune_rstd = torch.empty_like(rstd)
    kernel = get_gather_kernel() if mode == "gather" else get_standard_kernel()
    result = exhaustive_search(
        list(_standard_autotune_configs()),
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (m,),
        kernel=kernel,
        args_fn=lambda cfg: (
            x,
            weight,
            tune_out,
            tune_rstd,
            n,
            RMS_NORM_EPS,
            cfg.tile_size,
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        quiet=True,
    )
    cfg = result.best.config
    tuned_kernel = kernel.replace_hints(
        num_ctas=cfg.num_ctas,
        occupancy=cfg.occupancy,
    )
    return TunedKernel(tuned_kernel, cfg)


def _tune_static_persistent(torch, x, weight, out) -> TunedKernel:
    exhaustive_search = get_exhaustive_search()
    search_space = list(_static_persistent_autotune_configs(int(x.shape[1])))
    tune_out = torch.empty_like(out)
    kernel = get_static_persistent_kernel()
    result = exhaustive_search(
        search_space,
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: _static_persistent_grid(torch, x, cfg),
        kernel=kernel,
        args_fn=lambda cfg: (
            x,
            tune_out,
            weight,
            cfg.tile_size_m,
            cfg.tile_size_n,
            RMS_NORM_EPS,
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        quiet=True,
    )
    cfg = result.best.config
    tuned_kernel = kernel.replace_hints(
        num_ctas=cfg.num_ctas,
        occupancy=cfg.occupancy,
    )
    return TunedKernel(tuned_kernel, cfg)


def tune_cutile_rmsnorm(torch, x, weight, out, mode: str) -> TunedKernel:
    key = _cache_key(torch, x, mode)
    tuned = _tuning_cache.get(key)
    if tuned is not None:
        return tuned

    if mode == "static_persistent":
        tuned = _tune_static_persistent(torch, x, weight, out)
    else:
        tuned = _tune_standard_or_gather(torch, x, weight, out, mode)

    _tuning_cache[key] = tuned
    return tuned


def launch_tuned_cutile_rmsnorm(torch, x, weight, out, tuned: TunedKernel, rstd=None):
    ct = get_cutile_module()
    m, n = x.shape
    cfg = tuned.config

    if isinstance(cfg, StaticPersistentConfig):
        ct.launch(
            torch.cuda.current_stream(),
            _static_persistent_grid(torch, x, cfg),
            tuned.kernel,
            (x, out, weight, cfg.tile_size_m, cfg.tile_size_n, RMS_NORM_EPS),
        )
        return

    if rstd is None:
        rstd = torch.empty((m,), dtype=torch.float32, device=x.device)
    ct.launch(
        torch.cuda.current_stream(),
        (m,),
        tuned.kernel,
        (x, weight, out, rstd, n, RMS_NORM_EPS, cfg.tile_size),
    )


def bench_cutile_rmsnorm_autotune(state: bench.State) -> None:
    torch = require_torch(state)
    if torch is None:
        return

    if require_cutile_autotune(state) is None:
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
    rstd = None
    if mode != "static_persistent":
        rstd = torch.empty((batch_size,), dtype=torch.float32, device=x.device)

    try:
        stream = as_torch_stream(torch, state.get_stream(), state.get_device())
        with torch.cuda.stream(stream):
            tuned = tune_cutile_rmsnorm(torch, x, weight, out, mode)
            launch_tuned_cutile_rmsnorm(torch, x, weight, out, tuned, rstd)
        torch.cuda.synchronize(state.get_device())
    except RuntimeError as exc:
        if is_oom_error(exc):
            state.skip("Skipping: out of memory.")
            return
        raise
    except ValueError as exc:
        state.skip(f"Skipping: CuTile autotuning failed: {exc}")
        return

    check_rmsnorm_correctness(torch, x, weight, out, dtype_name)

    def launcher(launch: bench.Launch) -> None:
        stream = as_torch_stream(torch, launch.get_stream(), state.get_device())
        with torch.cuda.stream(stream):
            launch_tuned_cutile_rmsnorm(torch, x, weight, out, tuned, rstd)

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_cutile_rmsnorm_autotune)
    b.set_name("cutile_rmsnorm_autotune")
    b.add_string_axis("Mode{ct}", ["standard", "gather", "static_persistent"])
    add_rmsnorm_axes(b)
    bench.run_all_benchmarks(sys.argv)
