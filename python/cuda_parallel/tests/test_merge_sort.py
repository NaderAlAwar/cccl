# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numba.cuda
import numba.types
import numpy as np
import pytest
from conftest import random_array, type_to_problem_sizes

import cuda.parallel.experimental.algorithms as algorithms
from cuda.parallel.experimental.struct import gpu_struct


def merge_sort_device(
    d_in_keys, d_in_items, d_out_keys, d_out_items, op, num_items, stream=None
):
    merge_sort = algorithms.merge_sort(
        d_in_keys, d_in_items, d_out_keys, d_out_items, op
    )

    temp_storage_size = merge_sort(
        None, d_in_keys, d_in_items, d_out_keys, d_out_items, num_items
    )
    d_temp_storage = numba.cuda.device_array(
        temp_storage_size, dtype=np.uint8, stream=stream.ptr if stream else 0
    )
    merge_sort(
        d_temp_storage, d_in_keys, d_in_items, d_out_keys, d_out_items, num_items
    )


def compare_op(lhs, rhs):
    return np.uint8(lhs < rhs)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_device_merge_sort_keys(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2

        h_in_keys = random_array(num_items, dtype)

        d_in_keys = numba.cuda.to_device(h_in_keys)

        merge_sort_device(d_in_keys, None, d_in_keys,
                          None, compare_op, num_items)

        h_out_keys = d_in_keys.copy_to_host()
        h_in_keys.sort()

        np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_device_merge_sort_pairs(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2

        h_in_keys = random_array(num_items, dtype)
        h_in_items = random_array(num_items, np.float32)

        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_in_items = numba.cuda.to_device(h_in_items)

        merge_sort_device(
            d_in_keys, d_in_items, d_in_keys, d_in_items, compare_op, num_items
        )

        h_out_keys = d_in_keys.copy_to_host()
        h_out_items = d_in_items.copy_to_host()

        argsort = np.argsort(h_in_keys, stable=True)
        h_in_keys = np.array(h_in_keys)[argsort]
        h_in_items = np.array(h_in_items)[argsort]

        np.testing.assert_array_equal(h_out_keys, h_in_keys)
        np.testing.assert_array_equal(h_out_items, h_in_items)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_device_merge_sort_keys_copy(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2

        h_in_keys = random_array(num_items, dtype)
        h_out_keys = np.empty(num_items, dtype=dtype)

        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_out_keys = numba.cuda.to_device(h_out_keys)

        merge_sort_device(d_in_keys, None, d_out_keys,
                          None, compare_op, num_items)

        h_out_keys = d_out_keys.copy_to_host()
        h_in_keys.sort()

        np.testing.assert_array_equal(h_out_keys, h_in_keys)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_device_merge_sort_pairs_copy(dtype):
    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2

        h_in_keys = random_array(num_items, dtype)
        h_in_items = random_array(num_items, np.float32)
        h_out_keys = np.empty(num_items, dtype=dtype)
        h_out_items = np.empty(num_items, dtype=np.float32)

        d_in_keys = numba.cuda.to_device(h_in_keys)
        d_in_items = numba.cuda.to_device(h_in_items)
        d_out_keys = numba.cuda.to_device(h_out_keys)
        d_out_items = numba.cuda.to_device(h_out_items)

        merge_sort_device(
            d_in_keys, d_in_items, d_out_keys, d_out_items, compare_op, num_items
        )

        h_out_keys = d_out_keys.copy_to_host()
        h_out_items = d_out_items.copy_to_host()

        argsort = np.argsort(h_in_keys, stable=True)
        h_in_keys = np.array(h_in_keys)[argsort]
        h_in_items = np.array(h_in_items)[argsort]

        np.testing.assert_array_equal(h_out_keys, h_in_keys)
        np.testing.assert_array_equal(h_out_items, h_in_items)


def test_device_merge_sort_pairs_struct_type():
    @gpu_struct
    class key_pair:
        a: np.int8
        b: np.uint64

    @gpu_struct
    class item_pair:
        a: np.int32
        b: np.float32

    def struct_compare_op(lhs, rhs):
        return np.uint8(lhs.b < rhs.b if lhs.a == rhs.a else lhs.a < rhs.a)

    num_items = 10000

    a_keys = np.random.randint(0, 100, num_items, dtype=np.int8)
    b_keys = np.random.randint(0, 100, num_items, dtype=np.uint64)

    a_items = np.random.randint(0, 100, num_items, dtype=np.int32)
    b_items = np.random.rand(num_items).astype(np.float32)

    h_in_keys = np.empty(num_items, dtype=key_pair.dtype)
    h_in_items = np.empty(num_items, dtype=item_pair.dtype)

    h_in_keys["a"] = a_keys
    h_in_keys["b"] = b_keys

    h_in_items["a"] = a_items
    h_in_items["b"] = b_items

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_items = numba.cuda.to_device(h_in_items)

    merge_sort_device(
        d_in_keys, d_in_items, d_in_keys, d_in_items, struct_compare_op, num_items
    )

    h_out_keys = d_in_keys.copy_to_host()
    h_out_items = d_in_items.copy_to_host()

    argsort = np.argsort(h_in_keys, stable=True)
    h_in_keys = np.array(h_in_keys)[argsort]
    h_in_items = np.array(h_in_items)[argsort]

    np.testing.assert_array_equal(h_out_keys, h_in_keys)
    np.testing.assert_array_equal(h_out_items, h_in_items)
