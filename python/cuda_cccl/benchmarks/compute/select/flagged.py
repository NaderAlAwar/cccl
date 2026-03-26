# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for select_flagged using cuda.compute.

C++ equivalent: cub/benchmarks/bench/select/flagged.cu

Notes:
- The C++ benchmark generates both input values and bool flags with the same
  Entropy axis.
- Entropy for bool flags mirrors nvbench_helper generation in the range [0, 1],
  then the benchmark selects entries whose flag evaluates to true.
- InPlace axis controls whether output can alias input (not exposed in Python
  API).
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    FUNDAMENTAL_TYPES as TYPE_MAP,
)
from utils import as_cupy_stream, generate_data_with_entropy

import cuda.bench as bench
from cuda.compute import make_select_flagged


def bench_select_flagged(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    entropy_str = state.get_string("Entropy")

    alloc_stream = as_cupy_stream(state.get_stream())

    d_in = generate_data_with_entropy(num_elements, dtype, entropy_str, alloc_stream)
    d_flags_u8 = generate_data_with_entropy(
        num_elements,
        np.uint8,
        entropy_str,
        alloc_stream,
        min_val=0,
        max_val=1,
    )

    with alloc_stream:
        d_flags = d_flags_u8.astype(cp.bool_, copy=False)
        selected_elements = int(cp.count_nonzero(d_flags).get())
        d_out = cp.empty(selected_elements, dtype=dtype)
        d_num_selected = cp.zeros(1, dtype=np.int64)

    alloc_stream.synchronize()

    def flag_is_set(flag):
        return flag != 0

    selector = make_select_flagged(d_in, d_flags, d_out, d_num_selected, flag_is_set)

    temp_storage_bytes = selector(
        None,
        d_in,
        d_out,
        d_num_selected,
        flag_is_set,
        num_elements,
        d_flags,
    )
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_reads(num_elements * d_flags.dtype.itemsize)
    state.add_global_memory_writes(selected_elements * d_out.dtype.itemsize)
    state.add_global_memory_writes(1 * d_num_selected.dtype.itemsize)

    def launcher(launch: bench.Launch):
        selector(
            temp_storage,
            d_in,
            d_out,
            d_num_selected,
            flag_is_set,
            num_elements,
            d_flags,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_select_flagged)
    b.set_name("base")

    b.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b.add_int64_power_of_two_axis("Elements{io}", range(16, 29, 4))
    b.add_string_axis("Entropy", ["1.000", "0.544", "0.000"])
    # Note: InPlace and OffsetT axes from C++ are not exposed in Python API

    bench.run_all_benchmarks(sys.argv)
