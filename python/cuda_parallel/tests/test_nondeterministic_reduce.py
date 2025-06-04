# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba.cuda
import numba.types
import numpy as np
import pytest

import cuda.parallel.experimental.algorithms as algorithms


def random_int(shape, dtype):
    return np.random.randint(0, 5, size=shape).astype(dtype)


def type_to_problem_sizes(dtype):
    if dtype in [np.uint8, np.int8]:
        return [2, 4, 5, 6]
    elif dtype in [np.uint16, np.int16]:
        return [4, 8, 12, 14]
    elif dtype in [np.uint32, np.int32]:
        return [16, 20, 24, 26]
    elif dtype in [np.uint64, np.int64]:
        return [16, 20, 24, 25]
    else:
        raise ValueError("Unsupported dtype")


dtype_size_pairs = [
    (dt, 2**log_size)
    for dt in [np.uint8, np.uint16, np.uint32, np.uint64]
    for log_size in type_to_problem_sizes(dt)
]


@pytest.mark.parametrize("dtype,num_items", dtype_size_pairs)
def test_device_reduce(dtype, num_items):
    def op(a, b):
        return a + b

    init_value = 42
    h_init = np.array([init_value], dtype=dtype)
    d_output = numba.cuda.device_array(1, dtype=dtype)
    reduce_into = algorithms.nondeterministic_reduce_into(
        d_output, d_output, op, h_init
    )

    h_input = random_int(num_items, dtype)
    d_input = numba.cuda.to_device(h_input)
    temp_storage_size = reduce_into(None, d_input, d_output, d_input.size, h_init)
    d_temp_storage = numba.cuda.device_array(temp_storage_size, dtype=np.uint8)
    reduce_into(d_temp_storage, d_input, d_output, d_input.size, h_init)
    h_output = d_output.copy_to_host()
    assert h_output[0] == sum(h_input) + init_value
